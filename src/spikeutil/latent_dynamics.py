import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from spikeutil.analysis import firing_rate


def latent_dynamics(X, n_components=3, method="pca"):
    if method == "pca":
        pipe = make_pipeline(
            FunctionTransformer(np.sqrt),
            StandardScaler(),
            PCA(whiten=True, n_components=n_components),
        )
        Xt = pipe.fit_transform(X)
    else:
        raise ValueError("method must e one of pca,")

    return Xt


def avalanches(bst, mode="duration", norm=True):
    avalanches = np.mean(bst, axis=0) > 0
    avalanches = np.split(bst.T, np.where(np.diff(avalanches))[0] + 1)
    avalanches = [a.T for a in avalanches if not np.all(a == 0)]

    if mode == "duration":
        stat = np.array([a.shape[1] for a in avalanches])
        bins = np.arange(1, max(stat) + 1)
        y, _ = np.histogram(stat, bins=bins)
        x = bins[:-1]
    elif mode == "size":
        stat = np.array([np.count_nonzero(np.sum(a, axis=1)) for a in avalanches])
        bins = np.arange(1, max(stat) + 1)
        y, _ = np.histogram(stat, bins=bins)
        x = bins[:-1]
    elif mode == "average":
        stats = [(np.sum(a), a.shape[1]) for a in avalanches]
        counts = dict()
        for count, duration in stats:
            if not duration in counts.keys():
                counts[duration] = []
            counts[duration].append(count)
        counts = {d: np.mean(c) for d, c in counts.items()}
        x = np.arange(1, max(counts.keys()) + 1)
        y = np.zeros(len(x), dtype=float)
        y[np.array(list(counts.keys())) - 1] = list(counts.values())
    elif mode == "avalanches":
        return avalanches
    else:
        raise ValueError

    return x, y
