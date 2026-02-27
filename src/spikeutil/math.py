import numpy as np
import scipy.optimize
import scipy.stats


def wasserstein_centroid(X, bounds=None):

    def _obj(x):
        dist_fun = lambda xx: scipy.stats.wasserstein_distance(x, xx)
        dists = np.apply_along_axis(dist_fun, 1, X)
        return np.linalg.norm(dists)

    x0 = np.mean(X, axis=0)
    res = scipy.optimize.minimize(_obj, x0, bounds=bounds)
    return res.x


def smoothen(x):
    domain = np.arange(0, len(x))
    x = [_local_weighted_regression(x0, domain, x) for x0 in domain]
    x = np.array(x)
    return x


def _local_weighted_regression_weights(x0, X, tau):
    return np.exp(np.sum((X - x0) ** 2, axis=1) / (-2 * (tau**2)))


def _local_weighted_regression(x0, X, Y, tau=1):
    # add bias term
    x0 = np.r_[1, x0]
    X = np.c_[np.ones(len(X)), X]

    # fit model: normal equations with kernel
    xw = X.T * _local_weighted_regression_weights(x0, X, tau)
    theta = np.linalg.pinv(xw @ X) @ xw @ Y
    # predict
    return x0 @ theta
