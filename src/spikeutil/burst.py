import numpy as np
import ot
import scipy.ndimage
import scipy.signal.windows
import scipy.stats
import spikeinterface.curation as sc
import spikeinterface.metrics as sm
from kneed import KneeLocator

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
            #bounds = [(0, 1)] * hists.shape[1]
            #hist = wasserstein_centroid(hists, bounds)
            n = hists.shape[1]
            M = ot.utils.dist0(n)
            M /= M.max()
            hist = ot.barycenter(hists.T,M,1e-3)
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


def detect_tonic_units(sorting, censor_period=None, min_firing_rate=1):
    import matplotlib.pyplot as plt
    import spikeinterface.widgets as sw

    # Calculate tonic firing rates
    spike_vec = sorting.to_spike_vector()
    _, count = np.unique(spike_vec["unit_index"], return_counts=True)
    duration = spike_vec["sample_index"][-1] / sorting.sampling_frequency
    fr_total = count / duration


    sw.plot_rasters(sorting.select_units(sorting.unit_ids[np.argsort(fr_total)]), time_range=[0,60],figsize=(16,3))
    plt.show()
    
    quiet_units = sorting.unit_ids[fr_total>min_firing_rate]
    sorting = sorting.select_units(quiet_units)

    # Calculate tonic firing rates
    spike_vec = sorting.to_spike_vector()
    _, count = np.unique(spike_vec["unit_index"], return_counts=True)
    duration = spike_vec["sample_index"][-1] / sorting.sampling_frequency
    fr_total = count / duration



    cv = np.empty_like(fr_total)
    for i,u in enumerate(sorting.unit_ids):
       st = sorting.get_unit_spike_train_in_seconds(u)
       isi = np.diff(st)
       cv[i] = np.std(isi)/np.mean(isi)
    

    if censor_period is None:
        censor_period = 0
        avg_isis = []
        for u in sorting.unit_ids:
            st = sorting.get_unit_spike_train_in_seconds(u)
            avg_isi = np.median(np.diff(st))
            avg_isis.append(avg_isi)
        censor_period = np.nanmedian(avg_isis)*3
    print(censor_period)

    # Decimate spikes within bursts
    sorting_censored = sc.remove_duplicated_spikes(
        sorting, censored_period_ms=censor_period * 10**3, method='keep_first_iterative'
    )

    # Calculate tonic firing rates
    spike_vec = sorting_censored.to_spike_vector()
    _, count = np.unique(spike_vec["unit_index"], return_counts=True)
    duration = spike_vec["sample_index"][-1] / sorting_censored.sampling_frequency
    fr_tonic = count / duration
    fr_max = 1/censor_period

    sw.plot_rasters(sorting, time_range=[0,60],figsize=(16,3))
    plt.ylim([None, len(sorting.unit_ids)])
    plt.show()



    sw.plot_rasters(sorting.select_units(sorting.unit_ids[np.argsort(fr_total)]), time_range=[0,60],figsize=(16,3))
    plt.show()

    sw.plot_rasters(sorting.select_units(sorting.unit_ids[np.argsort(fr_tonic)]), time_range=[0,60],figsize=(16,3))
    plt.show()


    sw.plot_rasters(sorting_censored.select_units(sorting_censored.unit_ids[np.argsort(fr_total)]), time_range=[0,60],figsize=(16,3))
    plt.show()

    sw.plot_rasters(sorting_censored.select_units(sorting_censored.unit_ids[np.argsort(fr_tonic)]), time_range=[0,60],figsize=(16,3))
    plt.show()


    score = fr_tonic/fr_max
    score -=np.median(score)
    score /= scipy.stats.median_abs_deviation(score)
    kde = scipy.stats.gaussian_kde(score)
    x = np.linspace(min(score), max(score), 1024)


    mode = x[np.argmax(kde.pdf(x))]
    #thresh = max(mode, np.median(score)) 
    thresh = np.median(score)
    thresh = 2

    #diff2 = np.diff(np.diff(kde.pdf(x)))
    #inflection = x[np.where(np.diff(np.sign(diff2)).astype(bool) & (np.diff(diff2)<0))[0]]
    ##inflection = x[np.where(np.diff(np.sign(diff2)).astype(bool))[0]]
    #inflection = inflection[inflection>thresh]
    #if len(inflection):
    #    thresh = inflection[0]
    #print(thresh)

    #plt.plot(x, kde.pdf(x), color='k')
    #plt.axvline(inflection, color='r')
    #plt.axvline(mode, color='gray')
    #plt.axvline(np.median(score), color='gray', linestyle='--')
    #plt.show()

    #plt.axvline(inflection)
    #plt.axvline(mode)
    #plt.plot(x[:-2],diff2)
    #plt.axhline(0)
    #plt.show()

    is_tonic = score > thresh
    #is_tonic = score > np.median(score)
    #is_tonic = score >= thresh
    #is_tonic =  fr_tonic > np.quantile(fr_tonic,0.75)


    plt.scatter(score[~is_tonic], cv[~is_tonic], color='k', s=5)
    plt.scatter(score[is_tonic], cv[is_tonic], color='r', s=5)
    plt.xlabel('fr_tonic')
    plt.ylabel('cv')
    plt.show()
    
    sorting_censored_burst = sorting_censored.select_units(sorting_censored.unit_ids[~is_tonic])
    sw.plot_rasters(sorting_censored_burst, time_range=[0,60],figsize=(16,3))
    plt.show()

    sorting_censored_tonic = sorting_censored.select_units(sorting_censored.unit_ids[is_tonic])
    sw.plot_rasters(sorting_censored_tonic, time_range=[0,60],figsize=(16,3))
    plt.show()


    sorting_burst = sorting.select_units(sorting.unit_ids[~is_tonic])
    sw.plot_rasters(sorting_burst, time_range=[0,60],figsize=(16,3))
    plt.show()

    sorting_tonic = sorting.select_units(sorting.unit_ids[is_tonic])
    sw.plot_rasters(sorting_tonic, time_range=[0,60],figsize=(16,3))
    plt.show()



    tonic_units = sorting.unit_ids[is_tonic]
    return np.hstack([quiet_units, tonic_units])


def network_burst_params(
    sorting,
    exclude_tonic=True,
    Ns=128,
    logisi_kwargs=None,
    return_hists=False ,
    prominence_tol=1e-4,
    min_prob=1e-4,
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

    # Merge all units into one single spiketrain
    st = sorting.to_spike_vector()["sample_index"] / sorting.sampling_frequency

    # Calculage logISI_N distributions
    N_range = np.geomspace(3, int(len(st)/3), Ns, dtype=int)
    hists = []
    for N in N_range:
        hist, edges = log_isi_hist(st, N=N, **logisi_kwargs)
        hists.append(hist)
    x = edges[:-1] + 0.5 * np.diff(edges)
    hists = np.vstack(hists)

    # Find minimal N with clear separation between intra- and inter-burst intervals
    #max_p = 0
    isi_cutoff = 0
    N = 0
    max_prom = 0
    prev_prom = 0
    proms = np.zeros(len(N_range))
    for Ni, hist in enumerate(hists):
        hist = hist.copy()
        log_hist = np.log10(hist)
        peak_idc, peak_props = scipy.signal.find_peaks(log_hist, prominence=0)

        if len(peak_idc) >= 2:
            peak1_idx = peak_idc[0]
            peak2_idx = peak_idc[1]
            valley_idx =  np.argmin(hist[peak1_idx:peak2_idx])+peak1_idx

            peak1 = hist[peak1_idx]
            peak2 = hist[peak2_idx]
            valley = hist[valley_idx]

            if valley < min_prob:
                continue


            prom = min(peak1,peak2)-valley
            #prom = (min(peak1,peak2)-valley)/(max(peak1,peak2)-valley)

            proms[Ni] = prom
            if prom >= max_prom:
                max_prom = prom
                isi_cutoff = x[valley_idx]
                N = N_range[Ni]
                
            if prom < max_prom-prominence_tol:
                break

    import matplotlib.pyplot as plt
    plt.semilogx(N_range, proms)
    plt.axvline(N)
    plt.show()
    if N == 0:
        raise RuntimeError(
            "Unable to determine optimal burst identification parameters, probably no bursting behavior present"
        )
    if return_hists:
        return N, isi_cutoff, hists, edges, N_range
    return N, isi_cutoff


def detect_network_bursts(sorting, N=10, isi_N_cutoff=0.5, merge=True):
    bursts = []
    st = sorting.to_spike_vector()["sample_index"] / sorting.sampling_frequency
    isi_N = st[N:] - st[:-N]

    bursting = isi_N < isi_N_cutoff

    bursts = np.flatnonzero(np.diff(np.r_[0, bursting.view(np.int8), 0]))
    bursts = bursts.reshape(-1, 2)

    bursts[:,0] += 1
    bursts[:,1] += N-1
    bursts = st[bursts]
    if merge:
        bursts = merge_events(bursts)

    return bursts


def merge_events(events):
    if len(events) == 0:
        return events

    events = events[np.argsort(events[:, 0])]
    starts = events[:, 0]
    ends = events[:, 1]

    cummax_end = np.maximum.accumulate(ends)

    new_group = np.empty(len(events), dtype=bool)
    new_group[0] = True
    new_group[1:] = starts[1:] > cummax_end[:-1]

    idx = np.flatnonzero(new_group)

    merged_starts = starts[idx]
    merged_ends = np.maximum.reduceat(ends, idx)

    return np.column_stack((merged_starts, merged_ends))
