from functools import partial

import ConfigSpace
import ConfigSpace.hyperparameters
import ConfigSpace.util
import numpy as np
import torch
import torch.nn as nn
from pybnn.bohamiann import Bohamiann

from hpbandster.core.base_config_generator import base_config_generator
from hpbandster.optimizers.acquisition_functions.acquisition_functions import expected_improvement, thompson_sampling
from hpbandster.optimizers.acquisition_functions.local_search import local_search


def get_default_network(input_dimensionality: int) -> torch.nn.Module:
    class AppendLayer(nn.Module):
        def __init__(self, bias=True, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if bias:
                self.bias = nn.Parameter(torch.FloatTensor(1, 1))
            else:
                self.register_parameter('bias', None)

        def forward(self, x):
            return torch.cat((x, self.bias * torch.ones_like(x)), dim=1)

    def init_weights(module):
        if type(module) == AppendLayer:
            nn.init.constant_(module.bias, val=np.log(1e-3))
        elif type(module) == nn.Linear:
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="linear")
            nn.init.constant_(module.bias, val=0.0)

    return nn.Sequential(
        nn.Linear(input_dimensionality, 50), nn.Tanh(),
        nn.Linear(50, 1),
        AppendLayer()
    ).apply(init_weights)


class BNNCG(base_config_generator):
    def __init__(self, configspace, min_points_in_model=None, acquisition_func="ei",
                 top_n_percent=15, num_samples=64, random_fraction=1 / 3,
                 **kwargs):
        """
            Fits for each given budget a kernel density estimator on the best N percent of the
            evaluated configurations on this budget.


            Parameters:
            -----------
            configspace: ConfigSpace
                Configuration space object
            top_n_percent: int
                Determines the percentile of configurations that will be used as training data
                for the kernel density estimator, e.g if set to 10 the 10% best configurations will be considered
                for training.
            min_points_in_model: int
                minimum number of datapoints needed to fit a model
            num_samples: int
                number of samples drawn to optimize EI via sampling
            random_fraction: float
                fraction of random configurations returned


        """
        super().__init__(**kwargs)
        self.top_n_percent = top_n_percent
        self.configspace = configspace
        self.acquisition_func = acquisition_func

        self.min_points_in_model = min_points_in_model
        if min_points_in_model is None:
            self.min_points_in_model = len(self.configspace.get_hyperparameters()) + 1

        self.num_samples = num_samples
        self.random_fraction = random_fraction

        self.configs = dict()
        self.losses = dict()

        self.bnn_models = dict()
        self.is_training = False
        self.n_update = 1

    def largest_budget_with_model(self):
        if len(self.bnn_models) == 0:
            return -np.inf
        return max(self.bnn_models.keys())

    def get_config(self, budget):
        """
            Function to sample a new configuration

            This function is called inside Hyperband to query a new configuration


            Parameters:
            -----------
            budget: float
                the budget for which this configuration is scheduled

            returns: config
                should return a valid configuration

        """
        sample = None
        info_dict = {}

        # If no model is available, sample from prior
        # also mix in a fraction of random configs
        if len(self.bnn_models.keys()) == 0:  # or np.random.rand() < self.random_fraction:
            sample = self.configspace.sample_configuration()
            info_dict['model_based_pick'] = False

        if sample is None:
            # try:
            # sample from largest budget
            budget = max(self.bnn_models.keys())
            if self.acquisition_func == "ts":
                idx = np.random.randint(len(self.bnn_models[budget].sampled_weights))
                acquisition = partial(thompson_sampling, model=self.bnn_models[budget], idx=idx)
            # elif args.acquisition == "ucb":
            # acquisition = partial(lcb, model=self.bnn_models[budget])
            elif self.acquisition_func == "ei":
                acquisition = partial(expected_improvement, model=self.bnn_models[budget],
                                      y_star=np.min(self.bnn_models[budget].y))
            candidates = []
            cand_values = []
            for n in range(10):
                x_new, acq_val = local_search(acquisition,
                                              x_init=self.configspace.sample_configuration(),
                                              n_steps=10)
                candidates.append(x_new)
                cand_values.append(acq_val)

            best = np.argmax(cand_values)
            sample = candidates[best]

            sample = ConfigSpace.util.deactivate_inactive_hyperparameters(
                configuration_space=self.configspace,
                configuration=sample.get_dictionary()
            )
            info_dict['model_based_pick'] = True

        return sample.get_dictionary(), info_dict

    def impute_conditional_data(self, array):

        return_array = np.empty_like(array)

        for i in range(array.shape[0]):
            datum = np.copy(array[i])
            nan_indices = np.argwhere(np.isnan(datum)).flatten()

            while (np.any(nan_indices)):
                nan_idx = nan_indices[0]
                valid_indices = np.argwhere(np.isfinite(array[:, nan_idx])).flatten()

                if len(valid_indices) > 0:
                    # pick one of them at random and overwrite all NaN values
                    row_idx = np.random.choice(valid_indices)
                    datum[nan_indices] = array[row_idx, nan_indices]

                else:
                    # no good point in the data has this value activated, so fill it with a valid but random value
                    t = self.vartypes[nan_idx]
                    if t == 0:
                        datum[nan_idx] = np.random.rand()
                    else:
                        datum[nan_idx] = np.random.randint(t)

                nan_indices = np.argwhere(np.isnan(datum)).flatten()
            return_array[i, :] = datum
        return (return_array)

    def new_result(self, job, update_model=True):
        """
            function to register finished runs

            Every time a run has finished, this function should be called
            to register it with the result logger. If overwritten, make
            sure to call this method from the base class to ensure proper
            logging.


            Parameters:
            -----------
            job: hpbandster.distributed.dispatcher.Job object
                contains all the info about the run
        """

        super().new_result(job)
        if job.result is None:
            # One could skip crashed results, but we decided
            # assign a +inf loss and count them as bad configurations
            # TODO: this might be a potential issue with BNNs
            loss = np.inf
        else:
            loss = job.result["loss"]

        budget = job.kwargs["budget"]

        if budget not in self.configs.keys():
            self.configs[budget] = []
            self.losses[budget] = []

        if len(self.configs.keys()) == 1:
            min_num_points = 10
        else:
            min_num_points = self.min_points_in_model

        # We want to get a numerical representation of the configuration in the original space

        conf = ConfigSpace.Configuration(self.configspace, job.kwargs["config"]).get_array().tolist()

        if conf in self.configs[budget]:
            i = self.configs[budget].index(conf)
            self.losses[budget][i].append(loss)
            print('-' * 50)
            print('ran config %s with loss %f again' % (conf, loss))
        else:
            self.configs[budget].append(conf)
            self.losses[budget].append([loss])

        # skip model building:

        # a) if not enough points are available

        tmp = np.array([np.mean(r) for r in self.losses[budget]])
        if np.sum(np.isfinite(tmp)) < min_num_points:
            self.logger.debug(
                "Only %i successful run(s) for budget %f available, need more than %s -> can't build model!" % (
                    np.sum(np.isfinite(tmp)), budget, min_num_points))
            return

        # b) during warnm starting when we feed previous results in and only update once
        if not update_model:
            return

        x_train = np.array(self.configs[budget])

        l = [li[0] for li in self.losses[budget]]
        y_train = np.array(l)

        if y_train.shape[0] % self.n_update == 0:

            if not budget in self.bnn_models:
                self.bnn_models[budget] = Bohamiann(get_network=get_default_network, use_double_precision=True)

                self.bnn_models[budget].train(x_train, y_train, verbose=False, lr=1e-2,
                                          num_burn_in_steps=x_train.shape[0] * 10,
                                          num_steps=x_train.shape[0] * 10 + 200, keep_every=10,
                                          continue_training=False)

            else:
                self.bnn_models[budget].train(x_train, y_train, verbose=False, lr=1e-2,
                                          num_burn_in_steps=0,
                                          num_steps=200, keep_every=10,
                                          continue_training=True)
        # update probs for the categorical parameters for later sampling
        self.logger.debug(
            'done building a new model for budget %f based on %d data points \n\n\n\n\n' % (
                budget, x_train.shape[0]))
