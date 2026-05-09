import numpy as np
import pandas as pd
import scipy.optimize
from elephant.spike_train_synchrony import spike_contrast
from sklearn.metrics import r2_score

from spikeutil.analysis import binned_spike_train
from spikeutil.burst import (detect_network_bursts, detect_tonic_units,
                             network_burst_params)
from spikeutil.core import sorting_to_neo


def compute_qm_features(analyzer):
    qm = analyzer.get_metrics_extension_data()
    qm.columns = ["qm_" + c for c in qm.columns]
    qm = qm.fillna(0)
    return qm

    # Detect on good and mua


def compute_network_burst_features(analyzer):
    sorting = analyzer.sorting
    features = dict()

    # Detect bursts
    analyzer_bursting = analyzer.remove_units(detect_tonic_units(sorting))
    sorting_bursting = analyzer_bursting.sorting
    N, isi_N_cutoff = network_burst_params(sorting_bursting)
    bursts = detect_network_bursts(sorting_bursting, N, isi_N_cutoff)
    print(f"Detected {len(bursts)} bursts")

    # Get burst time features
    features["burst_N"] = N
    features["burst_isi_N_cutoff"] = isi_N_cutoff
    duration = (
        sorting.to_spike_vector()["sample_index"][-1] / sorting.sampling_frequency
    )
    features["burst_rate"] = len(bursts) / duration
    burst_duration = bursts[:, 1] - bursts[:, 0]
    features["burst_duration_mean"] = np.mean(burst_duration)
    features["burst_duration_var"] = np.var(burst_duration)
    features["burst_total_time"] = np.sum(burst_duration) / duration
    features["burst_inter_burst_interval_mean"] = np.mean(
        bursts[1:, 0] - bursts[:-1, 1]
    )
    features["burst_inter_burst_interval_var"] = np.var(bursts[1:, 0] - bursts[:-1, 1])

    # Get burst firing_rate_features
    bursts_fr = []
    bursts_t = []

    t, bst = binned_spike_train(analyzer.sorting, normalize_width=False)
    bst = np.sum(bst, axis=1)
    s = np.arange(len(t))

    burst_features = []
    for t_start, t_stop in bursts:
        s_start = s[t >= t_start][0]
        if len(s[t >= t_stop]) == 0:
            s_stop = s[-1]
        else:
            s_stop = s[t >= t_stop][0]

        burst_fr = bst[s_start:s_stop]
        burst_t = t[s_start:s_stop]
        bursts_fr.append(burst_fr)
        bursts_t.append(burst_t)

        burst_features.append(
            {
                "burst_decay_time": burst_t[-1] - burst_t[np.argmax(burst_fr)],
                "burst_rise_time": burst_t[np.argmax(burst_fr)] - burst_t[0],
                "burst_firing_rate_abs": np.sum(burst_fr) / duration,
                "burst_firing_rate_norm": np.sum(burst_fr) / (t_stop - t_start),
            }
        )
    burst_features = {
        k: np.mean([bf[k] for bf in burst_features])
        for k, v in burst_features[0].items()
    }
    features.update(burst_features)
    return features


def compute_unit_features(analyzer):
    sorting = analyzer.sorting

    features = []
    acgs = np.diagonal(analyzer.get_extension("correlograms").get_data()[0]).T
    t, bst = binned_spike_train(analyzer.sorting, normalize_width=False)
    duration = (
        sorting.to_spike_vector()["sample_index"][-1] / sorting.sampling_frequency
    )

    for i, uid in enumerate(analyzer.unit_ids):
        st = analyzer.sorting.get_unit_spike_train_in_seconds(uid)
        isi = np.diff(st)

        fr = len(st) / duration

        # ACG fit params
        acg = acgs[i]
        bins = analyzer.get_extension("correlograms").get_data()[1]
        lags = bins[:-1] + np.diff(bins) / 2
        lags_pos = lags[lags >= 0]
        acg_pos = acg[lags >= 0]
        params = fit_acg(lags_pos, acg_pos)

        acg_pred = acg_fit(lags_pos, *params)

        features.append(
            {
                "unit_isi_var_coeff": np.std(isi) / np.mean(isi),
                "unit_isi_mean": np.mean(isi),
                "unit_isi_median": np.median(isi),
                "unit_isi_var": np.median(isi),
                "unit_acg_fit_c": params[0],
                "unit_acg_fit_t_refrac": params[1],
                "unit_acg_fit_tau_decay": params[2],
                "unit_acg_fit_d": params[3],
                "unit_acg_fit_tau_rise": params[4],
                "unit_acg_fit_h": params[5],
                "unit_acg_fit_tau_burst": params[6],
                "unit_acg_fit_rate_asymptote": params[7],
                "unit_acg_fit_r2": r2_score(acg_pos, acg_pred),
                "unit_fr_var_coeff": np.var(bst[i]) / fr,
                "unit_fr_gini": gini(bst[i]),
                "unit_fr_instability": np.mean(isi) / fr,
            }
        )

    features = pd.DataFrame(features, index=analyzer.unit_ids)
    return features


def compute_network_features(analyzer):
    features = dict()

    seg = sorting_to_neo(analyzer.sorting)
    features["network_spike_contrast"], sc_props = spike_contrast(
        list(seg.spiketrains), return_trace=True
    )
    features["network_spike_contrast_max_bin"] = float(
        sc_props.bin_size[np.argmax(sc_props.contrast)]
    )

    return features


def acg_fit(x, c, t_refrac, tau_decay, d, tau_rise, h, tau_burst, rate_asymptote):
    xtr = -(x - t_refrac)
    acg_fit = np.clip(
        c * (np.exp(xtr / tau_decay) - d * np.exp(xtr / tau_rise))
        + h * np.exp(xtr / tau_burst)
        + rate_asymptote,
        a_min=0,
        a_max=None,
    )
    return acg_fit


def fit_acg(lags, acg):
    init = [30, 5, 20, 2, 1, 2, 1.5, 0.5]
    lower_bounds = [-np.inf, 0, 0, -np.inf, 0, -np.inf, 0, 0]
    upper_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.max(acg)]
    bounds = (lower_bounds, upper_bounds)
    res = scipy.optimize.curve_fit(
        acg_fit, lags, acg, p0=init, bounds=bounds, maxfev=1e6 * (len(init) + 1)
    )
    params = res[0]
    return params


def gini(x):
    """Compute Gini coefficient of array of values"""
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x) ** 2 * np.mean(x))
