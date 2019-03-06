import numpy as np
from scipy import stats


def lcb(candidates, model):
    mu, var = model.predict(candidates)
    return -(mu[0] - np.sqrt(var[0]))


def expected_improvement(candidates, model, y_star):
    mu, var = model.predict(candidates)

    s = np.sqrt(var)
    diff = (y_star - mu) / s
    f = s * (diff * stats.norm.cdf(diff) + stats.norm.pdf(diff))

    return f
