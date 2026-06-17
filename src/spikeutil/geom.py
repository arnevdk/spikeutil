import numpy as np
from factor_rotation import rotate_factors
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from spikeutil.analysis import firing_rate
from spikeutil.core import inst_firing_rate


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


def factor_analysis(sorting, n_components=8, fr_kwargs=None):
    if fr_kwargs is None:
        fr_kwargs = dict()
    fr_kwargs.setdefault("kernel_sigma", 0.05)
    fr_kwargs.setdefault("sfreq", 100)
    X, sfreq = inst_firing_rate(sorting, **fr_kwargs)

    Xt = StandardScaler(with_mean=False).fit_transform(X)
    Xt = X

    R = np.corrcoef(Xt, rowvar=False)
    shrinkage = 1e-6
    R = (1 - shrinkage) * R + shrinkage * np.eye(len(R))

    eigvals, eigvecs = np.linalg.eigh(R)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    L = eigvecs[:, :n_components] * np.sqrt(eigvals[:n_components])

    L, T = rotate_factors(L, "oblimin", 0, "oblique")
    # L,T = rotate_factors(L, 'varimax')

    idx = np.argsort(-np.sum(L**2, axis=0))
    L = L[:, idx]

    Xt = Xt @ L
    Xt = np.sign(np.mean(X, axis=1) @ Xt) * Xt

    return Xt, L
