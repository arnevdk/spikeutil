import numpy as np
import scipy.ndimage
import scipy.signal.windows
import scipy.stats
import spikeinterface.curation as sc
import spikeinterface.metrics as sm
import ot
import scipy.ndimage
from spikeutil.math import smoothen, wasserstein_centroid


def detect_bursts(
        sorting,
        method='participation',
        remove_tonic=False,
        merge_interval=0.200,
        min_burst_duration=0.300,
        tonic_kwargs=None,
        method_kwargs=None
    ):
      
    if remove_tonic:
        if tonic_kwargs is None:
            tonic_kwargs=dict()
        tonic_units = detect_tonic_units(sorting, **tonic_kwargs)
        print(f'Tonic units: {tonic_units}')

    if method_kwargs is None:
        method_kwargs=dict()

    if method == 'isi_N':
        N, isi_cutoff = isi_N_params(sorting, **method_kwargs)
        bursts = detect_bursts_isi_N(sorting, N=N, isi_cutoff = isi_cutoff)
    elif method == 'participation':
        bursts = detect_bursts_participation(sorting, **method_kwargs)
    else:
        ValueError(f"Method must be in ['isi_N', 'participation']")
 
    bursts = merge_bursts(bursts, max_interval=merge_interval)
    burst_duration = bursts[:,1]-bursts[:,0]
    bursts = bursts[burst_duration >= min_burst_duration]
    if len(bursts) <= 3:
        raise RuntimeError("Insufficient number of bursts detected")

    return bursts

def merge_bursts(bursts, max_interval=0.200):
    if not len(bursts):
        return bursts
    new_bursts = [bursts[0]]
    for i in range(1,len(bursts)):
        if bursts[i][0] - new_bursts[-1][1] <= max_interval:
            new_bursts[-1][1] = bursts[i][1]
        else:
            new_bursts.append(bursts[i])
    return np.array(new_bursts)
    



def detect_tonic_units(sorting, censor_period=None, min_firing_rate=1, score_thresh=2):

    # Calculate tonic firing rates
    spike_vec = sorting.to_spike_vector()
    _, count = np.unique(spike_vec["unit_index"], return_counts=True)
    duration = spike_vec["sample_index"][-1] / sorting.sampling_frequency
    fr_total = count / duration


    #import matplotlib.pyplot as plt
    #import spikeinterface.widgets as sw
    #sw.plot_rasters(sorting.select_units(sorting.unit_ids[np.argsort(fr_total)]), time_range=[0,60],figsize=(16,3))
    #plt.show()
    
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

    # Decimate spikes within bursts
    sorting_censored = sc.remove_duplicated_spikes(
        sorting,
        censored_period_ms=censor_period * 10**3, 
        method='keep_first_iterative',
    )

    # Calculate tonic firing rates
    spike_vec = sorting_censored.to_spike_vector()
    _, count = np.unique(spike_vec["unit_index"], return_counts=True)
    duration = spike_vec["sample_index"][-1] / sorting_censored.sampling_frequency
    fr_tonic = count / duration
    fr_max = 1/censor_period

    score = fr_tonic/fr_max
    score -=np.median(score)
    score /= scipy.stats.median_abs_deviation(score)
    kde = scipy.stats.gaussian_kde(score)
    x = np.linspace(min(score), max(score), 1024)


    mode = x[np.argmax(kde.pdf(x))]

    is_tonic = score > score_thresh


    tonic_units = sorting.unit_ids[is_tonic]
    return np.hstack([quiet_units, tonic_units])


def detect_bursts_participation(sorting, bin_width=0.100, smooth_sigma=2, thresh_Z=1):
    n_spikes, n_units = participation_bins(sorting, bin_width=bin_width)
    stat = n_spikes*n_units
    stat = scipy.ndimage.gaussian_filter(stat, sigma=smooth_sigma)

    spread = 1.4826*scipy.stats.median_abs_deviation(stat)
    baseline = np.median(stat)
    thresh=baseline+thresh_Z*spread

    bursting = stat > thresh
    bursts = np.flatnonzero(np.diff(np.r_[0, bursting.view(np.int8), 0]))
    bursts = bursts.reshape(-1, 2)
    bursts=bursts*bin_width

    return bursts

def participation_bins(sorting, bin_width=0.100):
    spike_vector = sorting.to_spike_vector()
    samples = spike_vector['sample_index']
    units = spike_vector['unit_index']
    sfreq = sorting.sampling_frequency
    n_samples = len(samples)
    bin_width_samples = int(bin_width*sfreq)
    s_max = spike_vector[-1]['sample_index']
    s=0
    i = 0
    n_spikes = []
    n_units = []
    while s < s_max:
        bin_units = []
        bin_n_spikes = 0
        while i < n_samples and samples[i] < s+bin_width_samples:
            bin_units.append(units[i])
            bin_n_spikes +=1
            i+=1
        
        bin_n_units = len(np.unique(bin_units))
        n_spikes.append(bin_n_spikes)
        n_units.append(bin_n_units)
        s += bin_width_samples
    return np.array(n_spikes), np.array(n_units)

def isi_N_params(
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

    #import matplotlib.pyplot as plt
    #plt.semilogx(N_range, proms)
    #plt.axvline(N)
    #plt.show()
    if N == 0:
        raise RuntimeError(
            "Unable to determine optimal burst identification parameters, probably no bursting behavior present"
        )
    if return_hists:
        return N, isi_cutoff, hists, edges, N_range
    return N, isi_cutoff


def detect_bursts_isi_N(sorting, N=10, isi_cutoff=0.5):
    """
    Pasquale, V., Martinoia, S. & Chiappalone, M. A self-adapting approach for
    the detection of bursts and network bursts in neuronal cultures.
    Journal of Computational Neuroscience 29, 213–229 (2009).
    """
    bursts = []
    st = sorting.to_spike_vector()["sample_index"] / sorting.sampling_frequency
    isi_N = st[N:] - st[:-N]

    bursting = isi_N < isi_cutoff

    bursts = np.flatnonzero(np.diff(np.r_[0, bursting.view(np.int8), 0]))
    bursts = bursts.reshape(-1, 2)

    bursts[:,0] += 1
    bursts[:,1] += N-1
    bursts = st[bursts]
    return bursts

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


