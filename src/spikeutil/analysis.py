import numpy as np
import quantities as pq
import scipy.signal
from elephant.conversion import BinnedSpikeTrain
from elephant.kernels import GaussianKernel
from elephant.statistics import instantaneous_rate

from spikeutil.core import sorting_to_neo
from spikeutil.math import smoothen, wasserstein_centroid


def normalized_coactivity(sorting, bin_width=0.05, t_stop=None):
    seg = sorting_to_neo(sorting)
    if t_stop is not None:
        t_stop = t_stop * pq.s
    bst = BinnedSpikeTrain(seg.spiketrains, bin_size=bin_width * pq.s, t_stop=t_stop)
    bst = bst.to_array()
    coactivity = np.mean(bst, axis=0) / len(seg.spiketrains)
    time = np.arange(len(coactivity)) * bin_width + bin_width / 2
    return time, coactivity


def log_isi_hist(st, bin_width_log=0.05, min_log=-3, max_log=2, pdf=True, smooth=True):
    isi = np.diff(st)
    logbins = 10 ** np.arange(min_log, max_log, bin_width_log)
    hist, bin_edges = np.histogram(isi, bins=logbins)
    if pdf:
        hist = hist / np.sum(hist)
    if smooth:
        hist = smoothen(hist)
    return hist, bin_edges


def log_isi_hists(sorting, method="all", **kwargs):
    if method == "wasserstein":
        if kwargs is None:
            kwargs = dict()
        kwargs["pdf"] = True

    if method in ["wasserstein", "mean", "all"]:
        hists = []
        for unit_id in sorting.get_unit_ids():
            st = sorting.get_unit_spike_train_in_seconds(unit_id)
            hist, bin_edges = log_isi_hist(st, **kwargs)
            hists.append(hist)
        hists = np.vstack(hists)
        if method == "mean":
            hist = np.mean(hists, axis=0)
            return hist, bin_edges
        elif method == "wasserstein":
            bounds = [(0, 1)] * hists.shape[1]
            hist = wasserstein_centroid(hists, bounds)
            return hist, bin_edges
        return hists, bin_edges
    elif method == "coactivity":
        co_spike_train = []
        for unit_id in sorting.get_unit_ids():
            unit_spike_train = sorting.get_unit_spike_train_in_seconds(unit_id)
            co_spike_train.append(unit_spike_train)
        co_spike_train = np.concatenate(co_spike_train)
        co_spike_train = np.sort(co_spike_train)
        return log_isi_hist(co_spike_train, **kwargs)
    raise ValueError(
        "method should be one of 'all', 'mean', 'wasserstein', 'coactivity'"
    )


def firing_rate_psd(
    sorting, kernel_sigma=0.02, sfreq=1000, nperseg=2**16, normalize=True
):
    seg = sorting_to_neo(sorting)

    kernel = GaussianKernel(sigma=kernel_sigma * pq.s)
    fr = instantaneous_rate(seg.spiketrains, 1 / sfreq * pq.s, kernel=kernel)
    fr = fr.T
    fr = fr / np.mean(fr, axis=1)[:, np.newaxis]
    fr = np.mean(fr, axis=0)

    f, Pxx_den = scipy.signal.welch(fr, sfreq, nperseg=nperseg)
    return f, Pxx_den
