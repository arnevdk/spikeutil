import numpy as np
import quantities as pq
import scipy.signal
from spikeutil.core import sorting_to_neo
from spikeutil.math import smoothen, wasserstein_centroid


def log_isi_hist(st, bin_width_log=0.05, min_log=-3, max_log=2, pdf=True, smooth=True):
    isi = np.diff(st)
    logbins = 10 ** np.arange(min_log, max_log, bin_width_log)
    hist, bin_edges = np.histogram(isi, bins=logbins)
    if smooth:
        hist = smoothen(hist)
    if pdf:
        hist = hist / np.sum(hist)
        hist = np.clip(hist, a_min=1e-6, a_max=1)
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


def firing_rate(
    sorting, kernel_sigma=0.02, sfreq=500, normalize=True, coactivity=False, t_stop=None
):
    if t_stop is not None:
        t_stop = t_stop * pq.s
    seg = sorting_to_neo(sorting)
    kernel = GaussianKernel(sigma=kernel_sigma * pq.s)
    fr = instantaneous_rate(
        seg.spiketrains,
        1 / sfreq * pq.s,
        kernel=kernel,
        t_stop=t_stop,
        pool_spike_trains=coactivity,
    )
    fr = np.array(fr).squeeze()
    if normalize and coactivity:
        fr = fr / len(seg.spiketrains)
    return fr, sfreq


def firing_rate_psd(fr, sfreq, nperseg=2**14, normalize=True):
    f, Pxx_den = scipy.signal.welch(fr, sfreq, nperseg=nperseg)
    if normalize:
        Pxx_den = Pxx_den / np.var(fr)
    return f, Pxx_den
