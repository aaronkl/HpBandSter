import collections
import random
import traceback
from copy import deepcopy
from functools import partial

import ConfigSpace
import ConfigSpace.hyperparameters
import ConfigSpace.util
import numpy as np
import torch
import torch.nn as nn
from hpbandster.core.base_config_generator import base_config_generator
from pybnn.bohamiann import Bohamiann


def thompson_sampling(candidates, model, budget):
    x = np.concatenate((candidates, np.ones([candidates.shape[0], 1]) * budget), axis=1)

    mu, var, samples = model.predict(x, return_individual_predictions=True)
    idx = np.random.randint(samples.shape[0])

    return samples[idx]


class Model(object):
    def __init__(self):
        self.arch = None
        self.accuracy = None

    def __str__(self):
        """Prints a readable version of this bitstring."""
        return '{0:b}'.format(self.arch)


def mutate_arch(parent_arch, cs):
    # pick random dimension
    dim = np.random.randint(len(cs.get_hyperparameters()))
    hyper = cs.get_hyperparameters()[dim]

    if type(hyper) == ConfigSpace.OrdinalHyperparameter:
        choices = list(hyper.sequence)
    else:
        choices = list(hyper.choices)
    # drop current values from potential choices
    choices.remove(parent_arch[hyper.name])

    # flip hyperparameter
    idx = np.random.randint(len(choices))

    child_arch = deepcopy(parent_arch)
    child_arch[hyper.name] = choices[idx]
    return child_arch


def regularized_evolution(acq, cs, cycles, population_size, sample_size):
    population = collections.deque()
    history = []  # Not used by the algorithm, only used to report results.

    # Initialize the population with random models.
    while len(population) < population_size:
        model = Model()
        model.arch = cs.sample_configuration()
        model.accuracy = acq(model.arch.get_array()[None, :])
        population.append(model)
        history.append(model)

    # Carry out evolution in cycles. Each cycle produces a model and removes
    # another.
    while len(history) < cycles:
        # Sample randomly chosen models from the current population.
        sample = []
        while len(sample) < sample_size:
            # Inefficient, but written this way for clarity. In the case of neural
            # nets, the efficiency of this line is irrelevant because training neural
            # nets is the rate-determining step.
            candidate = random.choice(list(population))
            sample.append(candidate)

        # The parent is the best model in the sample.
        parent = max(sample, key=lambda i: i.accuracy)

        # Create the child model and store it.
        child = Model()
        child.arch = mutate_arch(parent.arch, cs)
        child.accuracy = acq(child.arch.get_array()[None, :])
        population.append(child)
        history.append(child)

        # Remove the oldest model.
        population.popleft()
    cands_value = [i.accuracy for i in history]
    best = np.argmax(cands_value)
    x_new = history[best].arch
    return x_new


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


class H2BNN(base_config_generator):
    def __init__(self, configspace, min_points_in_model=None,
                 top_n_percent=15, num_samples=64, random_fraction=1 / 3,
                 min_bandwidth=1e-3, bw_estimator='scott', fully_dimensional=True,
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
            bw_estimator: string
                how the bandwidths is estimated. Possible values are 'scott' and 'mlcv' for maximum likelihood estimation
            min_bandwidth: float
                to keep diversity, even when all (good) samples have the same value for one of the parameters,
                a minimum bandwidth (Default: 1e-3) is used instead of zero.
            fully_dimensional: bool
                if true, the KDE is uses factored kernel across all dimensions, otherwise the PDF is a product of 1d PDFs

        """
        super().__init__(**kwargs)
        self.top_n_percent = top_n_percent
        self.configspace = configspace
        self.bw_estimator = bw_estimator
        self.min_bandwidth = min_bandwidth
        self.fully_dimensional = fully_dimensional

        self.min_points_in_model = min_points_in_model
        if min_points_in_model is None:
            self.min_points_in_model = len(self.configspace.get_hyperparameters()) + 1

        # if self.min_points_in_model < len(self.configspace.get_hyperparameters())+1:
        #	self.logger.warning('Invalid min_points_in_model value. Setting it to %i'%(len(self.configspace.get_hyperparameters())+1))
        #	self.min_points_in_model =len(self.configspace.get_hyperparameters())+1

        self.num_samples = num_samples
        self.random_fraction = random_fraction

        self.configs = dict()
        self.losses = dict()

        self.bnn = None

    # def largest_budget_with_model(self):
    #     if len(self.kde_models) == 0:
    #         return (-np.inf)
    #     return (max(self.kde_models.keys()))

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
        if self.bnn is None or np.random.rand() < self.random_fraction:
            sample = self.configspace.sample_configuration()
            info_dict['model_based_pick'] = False

        if sample is None:
            try:

                # Thompson sampling
                # if args.acquisition == "ts":
                acquisition = partial(thompson_sampling, model=self.bnn, budget=budget)
                # elif args.acquisition == "ucb":
                # acquisition = partial(ucb, model=bnn)
                # elif args.acquisition == "ei":
                # acquisition = partial(expected_improvement, model=bnn, y_star=np.argmax(y))

                sample = regularized_evolution(acq=acquisition, cs=self.configspace,
                                               cycles=1000, population_size=100, sample_size=10)

                sample = ConfigSpace.util.deactivate_inactive_hyperparameters(
                    configuration_space=self.configspace,
                    configuration=sample.get_dictionary()
                )
                info_dict['model_based_pick'] = True

            except Exception as e:
                self.logger.warning(("=" * 50 + "\n") * 3 + \
                                    "Error sampling a configuration!\n" + \
                                    "\n here is a traceback:" + \
                                    traceback.format_exc())

                for b, l in self.losses.items():
                    self.logger.debug("budget: {}\nlosses:{}".format(b, l))

                sample = self.configspace.sample_configuration()
                info_dict['model_based_pick'] = False

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
                    np.sum(np.isfinite(self.losses[budget])), budget, min_num_points))
            return

        # b) during warnm starting when we feed previous results in and only update once
        if not update_model:
            return

        x_train = []
        y_train = []

        for b in self.configs.keys():
            configs = np.array(self.configs[b])
            configs = np.concatenate((configs, np.ones([configs.shape[0], 1]) * b), axis=1)
            x_train.extend(configs)
            y_train.extend(self.losses[b])

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        # train BNN here
        bnn = Bohamiann(get_network=get_default_network, use_double_precision=False)

        # train only on good points
        idx = np.argsort(y_train[:, 0])[:100]
        x_train = x_train[idx]
        y_train = y_train[idx]

        bnn.train(x_train, y_train, verbose=True, lr=1e-5, num_burn_in_steps=20000, num_steps=20110)

        # update probs for the categorical parameters for later sampling
        self.logger.debug(
            'done building a new model for budget %f based on %d data points \n\n\n\n\n' % (
                budget, x_train.shape[0]))
