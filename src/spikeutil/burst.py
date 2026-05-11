import numpy as np
import scipy.ndimage
import scipy.signal.windows
import spikeinterface.curation as sc
import spikeinterface.metrics as sm

from spikeutil.math import smoothen, wasserstein_centroid


def log_isi_hist(
    st, bin_width_log=0.05, min_log=-3, max_log=2, pdf=True, smooth=True, N=1
):
    isi = st[N:] - st[:-N]
    logbins = 10 ** np.arange(min_log, max_log, bin_width_log)
    hist, bin_edges = np.histogram(isi, bins=logbins)
    if smooth:
        filt_len = 16
        filt = scipy.signal.windows.gaussian(filt_len, 1)
        hist = np.convolve(hist, filt, mode="same")
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


def detect_tonic_units(sorting, censor_period=0.250, quantile=0.75):
    # Remove spikes with low ISI
    sorting_censored = sc.remove_duplicated_spikes(
        sorting, censored_period_ms=censor_period * 10**3
    )
    spike_vec = sorting_censored.to_spike_vector()

    # Calculate firing rates
    _, count = np.unique(spike_vec["unit_index"], return_counts=True)
    duration = spike_vec["sample_index"][-1] / sorting.sampling_frequency
    fr = count / duration
    # Detect tonic units as units with the highest tonic firing rate
    is_tonic = fr > np.quantile(fr, quantile)
    tonic_units = sorting.unit_ids[is_tonic]
    return tonic_units


def network_burst_params(
    sorting,
    exclude_tonic=True,
    Ns=32,
    prominence=0.1,
    tonic_kwargs=None,
    logisi_kwargs=None,
):
    """
    Bakkum DJ, Radivojevic M, Frey U, Franke F, Hierlemann A and Takahashi H
    (2014) Parameters for burst detection. Front. Comput. Neurosci. 7:193. doi:
    10.3389/fncom.2013.00193
    """
    if logisi_kwargs is None:
        logisi_kwargs = dict()
    logisi_kwargs.setdefault("min_log", -2)
    logisi_kwargs.setdefault("max_log", 3)
    if tonic_kwargs is None:
        tonic_kwargs = dict()

    if exclude_tonic:
        tonic_units = detect_tonic_units(sorting, **tonic_kwargs)
        sorting = sorting.remove_units(tonic_units)

    # Merge all units into one single spiketrain
    st = sorting.to_spike_vector()["sample_index"] / sorting.sampling_frequency

    # Calculage logISI_N distributions
    N_range = np.geomspace(1, len(st), Ns, dtype=int)
    hists = []
    for N in N_range:
        hist, edges = log_isi_hist(st, N=N, **logisi_kwargs)
        hists.append(hist)
    x = edges[:-1] + 0.5 * np.diff(edges)
    hists = np.vstack(hists)

    # Find minimal N with clear separation between intra- and inter-burst intervals
    valley_idx = -1
    for Ni, hist in enumerate(hists):
        hist = np.log10(hist)
        valley_idc, props = scipy.signal.find_peaks(-hist, prominence=prominence)
        if len(valley_idc):
            valley_idx = valley_idc[0]
            break

    if valley_idx == -1:
        raise RuntimeError(
            "Unable to determine optimal burst identification parameters, probably no bursting behavior present"
        )
    N = N_range[Ni]
    isi_N_cutoff = x[valley_idx]
    return N, isi_N_cutoff


def detect_network_bursts(sorting, N=10, isi_N_cutoff=0.5):
    bursts = []
    st = sorting.to_spike_vector()["sample_index"] / sorting.sampling_frequency
    isi_N = st[N:] - st[:-N]
    bursting = isi_N < isi_N_cutoff

    bursts = np.flatnonzero(np.diff(np.r_[0, bursting.view(np.int8), 0]))
    bursts = bursts.reshape(-1, 2)

    bursts[:, 1] += N - 1
    bursts = st[bursts]
    return bursts
