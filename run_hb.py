import os
import pickle
import sys
import numpy as np
import ConfigSpace
import json
import logging
logging.basicConfig(level=logging.INFO)

from hpbandster.optimizers.hyperband import HyperBand
import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker

#from hpolib.benchmarks.surrogates.paramnet import SurrogateReducedParamNetTime as mlp_surrogate
from hpolib.benchmarks.surrogates.paramnet import SurrogateParamNet as mlp_surrogate

dataset = sys.argv[1]
run_id = int(sys.argv[2])

mlp_budgets = {  # (min, max)-budget for the different data sets
    'adult': (1, 50),
    'higgs': (1, 50),
    'letter': (1, 50),
    'mnist': (1, 50),
    'optdigits': (1, 50),
    'poker': (1, 50),
}


b = mlp_surrogate(dataset=dataset)#, path=surrogate_path)


min_budget = mlp_budgets[dataset][0]
max_budget = mlp_budgets[dataset][1]

output_path = './res'

n_iters = 128

config_space = b.get_configuration_space()

traj = []
wall_clock_time = []

class MyWorker(Worker):

    def compute(self, config, budget, *args, **kwargs):

        c = ConfigSpace.Configuration(config_space, values=config)

        r = b.objective_function(c, step=int(budget))

        y = r["function_value"]

        cost = r["cost"]
        traj.append(y)
        wall_clock_time.append(cost)

        return ({'loss': y,
                 'info': {'cost': cost}
                })


hb_run_id = '0'

NS = hpns.NameServer(run_id=hb_run_id, host='localhost', port=0)
ns_host, ns_port = NS.start()

workers = []
for i in range(1):
    w = MyWorker(nameserver=ns_host, nameserver_port=ns_port,
                 run_id=hb_run_id,  # unique Hyperband run id
                 id=i  # unique ID as all workers belong to the same process
                 )
    w.run(background=True)
    workers.append(w)

HB = HyperBand(configspace=config_space,
           run_id=hb_run_id,
           eta=2, min_budget=min_budget, max_budget=max_budget,  # HB parameters
           nameserver=ns_host,
           nameserver_port=ns_port,
           ping_interval=10)

results = HB.run(n_iters, min_n_workers=1)
HB.shutdown(shutdown_workers=True)
NS.shutdown()

res = dict()

# res['incumbent_trajectory'] = results.get_incumbent_trajectory()['losses']
# res['budgets'] = results.get_incumbent_trajectory()['budgets']
# wall_clock_time = []
# cum_time = 0
# for c in results.get_incumbent_trajectory()["config_ids"]:
#     cum_time += results.get_runs_by_id(c)[-1]["info"]["cost"]
#     wall_clock_time.append(cum_time)
#
# res['wall_clock_time'] = wall_clock_time


inc_traj = []
curr_best = np.inf
for yi in traj:
    if yi < curr_best:
        curr_best = yi
    inc_traj.append(curr_best)
res['incumbent_trajectory'] = inc_traj
res['wall_clock_time'] = np.cumsum(wall_clock_time).tolist()

subdir = "hb"
os.makedirs(os.path.join(output_path, dataset, subdir), exist_ok=True)
fh = open(os.path.join(output_path, dataset, subdir, 'hyperband_%d.json' % run_id), 'w')
json.dump(res, fh)
fh.close()
