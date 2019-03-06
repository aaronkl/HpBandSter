import numpy as np
import ConfigSpace.util as util


def local_search(f, x_init, n_steps):
    incumbent = x_init
    incumbent_value = f(x_init.get_array()[None, :])
    for i in range(n_steps):

        f_nbs = []
        nbs = []
        for n in util.get_one_exchange_neighbourhood(x_init, np.random.randint(100000)):
            nbs.append(n)
            f_nbs.append(f(n.get_array()[None, :])[0])

        # check whether we improved
        best = np.argmax(f_nbs)

        if f_nbs[best] > incumbent_value:

            incumbent = nbs[best]
            incumbent_value = f_nbs[best]
            # jump to the next neighbour
            x_init = nbs[best]
        else:
            # in case we converged, stop the local search
            break

    return incumbent, incumbent_value
