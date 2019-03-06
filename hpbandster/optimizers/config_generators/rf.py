import traceback
from functools import partial

import ConfigSpace
import ConfigSpace.hyperparameters
import numpy as np
from hpbandster.core.base_config_generator import base_config_generator
from hpbandster.optimizers.acquisition_functions.acquisition_functions import expected_improvement
from hpbandster.optimizers.acquisition_functions.local_search import local_search
from robo.models.random_forest import RandomForest


class RF(base_config_generator):
    def __init__(self, configspace, eta=3, min_budget=0.01, max_budget=1,
                 min_points_in_model=None, top_n_percent=15,
                 num_samples=32, random_fraction=1 / 3, **kwargs):
        """
            Fits for each given budget a kernel density estimator on the best N percent of the
            evaluated configurations on this budget.


            Parameters:
            -----------
            configspace: ConfigSpace
                Configuration space object

        """
        super().__init__(**kwargs)
        self.top_n_percent = top_n_percent
        self.configspace = configspace

        self.min_points_in_model = min_points_in_model
        if min_points_in_model is None:
            self.min_points_in_model = len(self.configspace.get_hyperparameters()) + 1

        self.num_samples = num_samples
        self.random_fraction = random_fraction
        self.configspace = configspace

        self.configs = dict()
        self.losses = dict()

        self.rf_models = dict()
        self.is_training = False
        self.n_update = 1

    def largest_budget_with_model(self):
        if len(self.rf_models) == 0:
            return -np.inf
        return max(self.rf_models.keys())

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
        if len(self.rf_models.keys()) == 0:  # or np.random.rand() < self.random_fraction:
            sample = self.configspace.sample_configuration()
            info_dict['model_based_pick'] = False

        if sample is None:
            try:
                # sample from largest budget
                budget = max(self.rf_models.keys())
                # Thompson sampling
                # if args.acquisition == "ts":
                # idx = np.random.randint(len(self.rf_models[budget].sampled_weights))
                # acquisition = partial(thompson_sampling, model=self.rf_models[budget], idx=idx)
                # elif args.acquisition == "ucb":
                # acquisition = partial(lcb, model=self.rf_models[budget])
                # elif args.acquisition == "ei":
                acquisition = partial(expected_improvement, model=self.rf_models[budget],
                                      y_star=np.min(self.rf_models[budget].y))

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
        # else:
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
            self.rf_models[budget] = RandomForest(num_trees=100)
            self.rf_models[budget].train(x_train, y_train)

        # update probs for the categorical parameters for later sampling
        self.logger.debug(
            'done building a new model for budget %f based on %d data points \n\n\n\n\n' % (
                budget, x_train.shape[0]))
