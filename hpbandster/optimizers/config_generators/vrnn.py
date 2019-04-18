import logging
from copy import deepcopy
import traceback


import ConfigSpace
import ConfigSpace.hyperparameters
import ConfigSpace.util
import numpy as np
import scipy.stats as sps
import scipy.optimize as spo
import statsmodels.api as sm

import sklearn.preprocessing as preprocessing

import os
import json

from vdrnn.vrnn_bohb import VRNN

from hpbandster.core.base_config_generator import base_config_generator

import ConfigSpace.util as util
from functools import partial

def local_search(f, x_init, n_steps):
    incumbent = x_init
    incumbent_value = f(x_init)
    for i in range(n_steps):

        f_nbs = []
        nbs = []
        for n in util.get_one_exchange_neighbourhood(x_init, np.random.randint(100000)):
            nbs.append(n)
            f_nbs.append(f(n)[0])

        # check whether we improved
        best = np.argmax(f_nbs)

        if f_nbs[best] > incumbent_value:

            incumbent = nbs[best]
            incumbent_value = f_nbs[best]
            # jump to the next neighbour
            x_init = nbs[best]
        else:
            # in case we converged, stop the local search
            print("Local search performed %d steps" % i)
            break

    return incumbent, incumbent_value


def thompson_sampling(theta_dict, model):

   ##########droout 0        dropout 1          initial lr        shape par         final lr frac     batch size         num layers       avg units per layer
    theta = preprocessing.scale(np.asarray([theta_dict['x6'], theta_dict['x7'], 10**(theta_dict['x0']), theta_dict['x4'],10**(theta_dict['x3']),
								int(2**(theta_dict['x1'])), int(theta_dict['x5']), int(2**(theta_dict['x2']))]))

    # do roll out of theta until T
    val = model.eval(np.expand_dims(theta, axis=0), 50)[-1]#int(budget))[-1]
    return val


class VRNNWrapper(base_config_generator):
	def __init__(self, configspace,
				 num_samples = 64,
				 path= './model_vrnn',
				**kwargs):
		"""
			Fits for each given budget a kernel density estimator on the best N percent of the
			evaluated configurations on this budget.


			Parameters:
			-----------
			configspace: ConfigSpace    super(LCN
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
			bandwidth_factor: float
				widens the bandwidth for contiuous parameters for proposed points to optimize EI
			min_bandwidth: float
				to keep diversity, even when all (good) samples have the same value for one of the parameters,
				a minimum bandwidth (Default: 1e-3) is used instead of zero.

		"""
		super(VRNNWrapper, self).__init__(**kwargs)


		self.configspace = configspace

		self.num_samples = num_samples


		self.configs = dict()
		self.vrnn_models = dict()

		#create an instance of vrnn
		self.vrnn = VRNN()

		#load the vrnn model for learning curve prediciton
		self.path = path

		with open(os.path.join(path,'par.json'), 'r') as f:
			par = json.load(f)
		f.close()

		with open(os.path.join(path,'hyper.json'), 'r') as f:
			hyper_par = json.load(f)
		f.close()
		#TODO: I need the size of the config space!
		#laod the pre-trained model
		print('config space size {}'.format(len(self.configspace.get_hyperparameters())))
		self.vrnn.load_model(hyper_par, par, len(self.configspace.get_hyperparameters()))

	def largest_budget_with_model(self):
		if len(self.vrnn_models) == 0:
			return(-float('inf'))
		return(max(self.vrnn_models.keys()))

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

		self.logger.debug('start sampling a new configuration.')


        
		sample = None
		info_dict = {}
		info_dict['model_based_pick'] = True
		self.vrnn.lstm.mask_generate(1)
		acquisition = partial(thompson_sampling, model=self.vrnn)


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
		# best = -np.inf
		# best_vector = None
        #
		# # sample dropout mask and keep it fixed for all the configuration sample
		#
        #
		# for i in range(self.num_samples):
        #
		# 	theta_dict = self.configspace.sample_configuration()
        #
		# 	#reshape e standardizzazione
        #
         #    ##########droout 0        dropout 1          initial lr        shape par         final lr frac     batch size         num layers       avg units per layer
		# 	theta = preprocessing.scale(np.asarray([theta_dict['x6'], theta_dict['x7'], 10**(theta_dict['x0']), theta_dict['x4'],10**(theta_dict['x3']),
		# 								int(2**(theta_dict['x1'])), int(theta_dict['x5']), int(2**(theta_dict['x2']))]))
        #
		# 	# do roll out of theta until T
		# 	val = self.vrnn.eval(np.expand_dims(theta, axis=0), 50)[-1]#int(budget))[-1]
        #
		# 	if val > best:
		# 		best = val
		# 		best_vector = theta_dict.get_array()#theta
		# sample = ConfigSpace.Configuration(self.configspace, vector=best_vector)

		return sample.get_dictionary(), info_dict


	def impute_conditional_data(self, array):

		return_array = np.empty_like(array)

		for i in range(array.shape[0]):
			datum = np.copy(array[i])
			nan_indices = np.argwhere(np.isnan(datum)).flatten()

			while (np.any(nan_indices)):
				nan_idx = nan_indices[0]
				valid_indices = np.argwhere(np.isfinite(array[:,nan_idx])).flatten()

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
			return_array[i,:] = datum
		return(return_array)

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

		# if job.result is None:
		# 	# One could skip crashed results, but we decided to
		# 	# assign a +inf loss and count them as bad configurations
		# 	loss = np.inf
		# else:
		# 	# same for non numeric losses.
		# 	# Note that this means losses of minus infinity will count as bad!
		# 	loss = job.result["loss"] if np.isfinite(job.result["loss"]) else np.inf
		#
		# budget = job.kwargs["budget"]
		#
		# if budget not in self.configs.keys():
		# 	self.configs[budget] = []
		# 	self.losses[budget] = []
		#
		# # skip model building if we already have a bigger model
		# if max(list(self.kde_models.keys()) + [-np.inf]) > budget:
		# 	return
		#
		# # We want to get a numerical representation of the configuration in the original space
		#
		# conf = ConfigSpace.Configuration(self.configspace, job.kwargs["config"])
		# self.configs[budget].append(conf.get_array())
		# self.losses[budget].append(loss)
		#
		#
		# # skip model building:
		# #		a) if not enough points are available
		# if len(self.configs[budget]) <= self.min_points_in_model-1:
		# 	self.logger.debug("Only %i run(s) for budget %f available, need more than %s -> can't build model!"%(len(self.configs[budget]), budget, self.min_points_in_model+1))
		# 	return
		#
		# #		b) during warnm starting when we feed previous results in and only update once
		# if not update_model:
		# 	return
		#
		# train_configs = np.array(self.configs[budget])
		# train_losses =  np.array(self.losses[budget])
		#
		# n_good= max(self.min_points_in_model, (self.top_n_percent * train_configs.shape[0])//100 )
		# #n_bad = min(max(self.min_points_in_model, ((100-self.top_n_percent)*train_configs.shape[0])//100), 10)
		# n_bad = max(self.min_points_in_model, ((100-self.top_n_percent)*train_configs.shape[0])//100)
		#
		# # Refit KDE for the current budget
		# idx = np.argsort(train_losses)
		#
		# train_data_good = self.impute_conditional_data(train_configs[idx[:n_good]])
		# train_data_bad  = self.impute_conditional_data(train_configs[idx[n_good:n_good+n_bad]])
		#
		# if train_data_good.shape[0] <= train_data_good.shape[1]:
		# 	return
		# if train_data_bad.shape[0] <= train_data_bad.shape[1]:
		# 	return
		#
		# #more expensive crossvalidation method
		# #bw_estimation = 'cv_ls'
		#
		# # quick rule of thumb
		# bw_estimation = 'normal_reference'
		#
		# bad_kde = sm.nonparametric.KDEMultivariate(data=train_data_bad,  var_type=self.kde_vartypes, bw=bw_estimation)
		# good_kde = sm.nonparametric.KDEMultivariate(data=train_data_good, var_type=self.kde_vartypes, bw=bw_estimation)
		#
		# bad_kde.bw = np.clip(bad_kde.bw, self.min_bandwidth,None)
		# good_kde.bw = np.clip(good_kde.bw, self.min_bandwidth,None)
		#
		# self.kde_models[budget] = {
		# 		'good': good_kde,
		# 		'bad' : bad_kde
		# }
		#
		# # update probs for the categorical parameters for later sampling
		# self.logger.debug('done building a new model for budget %f based on %i/%i split\nBest loss for this budget:%f\n\n\n\n\n'%(budget, n_good, n_bad, np.min(train_losses)))
