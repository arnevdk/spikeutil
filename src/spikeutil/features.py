import warnings

import numpy as np
import pandas as pd
import scipy.optimize
from elephant.spike_train_synchrony import spike_contrast
from sklearn.metrics import r2_score

from spikeutil.analysis import binned_spike_train
from spikeutil.burst import detect_bursts
from spikeutil.core import inst_firing_rate, sorting_to_neo


def compute_qm_features(analyzer):
    qm = analyzer.get_metrics_extension_data()
    qm.columns = ["qm_" + c for c in qm.columns]
    qm = qm.fillna(0)
    return qm

    # Detect on good and mua


def compute_network_burst_features(analyzer, **burst_params):
    sorting = analyzer.sorting
    n_units = len(sorting.unit_ids)
    features = dict()
    duration = (
        sorting.to_spike_vector()["sample_index"][-1] / sorting.sampling_frequency
    )


    try:
        bursts = detect_bursts(analyzer.sorting, **burst_params)
        print(f"Detected {len(bursts)} bursts")

        # Get burst time features
        features["burst_rate"] = len(bursts) / duration
        burst_duration = bursts[:, 1] - bursts[:, 0]
        features["burst_duration_mean"] = np.mean(burst_duration)
        features["burst_duration_var"] = np.var(burst_duration)
        features["burst_total_time"] = np.sum(burst_duration) / duration
        features["burst_inter_burst_interval_mean"] = np.nanmean(
            bursts[1:, 0] - bursts[:-1, 1]
        )
        features["burst_inter_burst_interval_var"] = np.nanvar(
            bursts[1:, 0] - bursts[:-1, 1]
        )

        intra_burst_spikes = 0
        intra_burst_time = 0
        for t_start, t_stop in bursts:
            intra_burst_sorting = sorting.time_slice(t_start, t_stop)
            n_spikes = intra_burst_sorting.count_total_num_spikes()
            intra_burst_spikes += n_spikes
            intra_burst_time += t_stop - t_start
        features["burst_intra_burst_firing_rate"] = intra_burst_spikes / (
            intra_burst_time * n_units
        )

        inter_burst_spikes = 0
        inter_burst_time = 0
        inter_bursts = np.hstack([bursts[:-1, [1]], bursts[1:, [0]]])
        for t_start, t_stop in inter_bursts:
            inter_burst_sorting = sorting.time_slice(t_start, t_stop)
            n_spikes = inter_burst_sorting.count_total_num_spikes()
            inter_burst_spikes += n_spikes
            inter_burst_time += t_stop - t_start
        features["burst_inter_burst_firing_rate"] = inter_burst_spikes / (
            inter_burst_time * n_units
        )

        rise_times = []
        decay_times = []
        for t_start, t_stop in bursts:
            intra_burst_sorting = sorting.time_slice(t_start, t_stop)
            fr, sfreq = inst_firing_rate(
                intra_burst_sorting, coactivity=True, normalize=True
            )
            time = np.arange(len(fr)) / sfreq + t_start
            t_peak = time[np.argmax(fr)]
            rise_times.append(t_peak - t_start)
            decay_times.append(t_stop - t_peak)
        features["burst_rise_time"] = np.mean(rise_times)
        features["burst_decay_time"] = np.mean(decay_times)
        return features
    except RuntimeError:
        return dict()


def compute_unit_features(analyzer):
    # TODO: population coupling

    sorting = analyzer.sorting

    features = []
    acgs = np.diagonal(analyzer.get_extension("correlograms").get_data()[0]).T
    t, bsts = binned_spike_train(analyzer.sorting, normalize_width=False)
    duration = (
        sorting.to_spike_vector()["sample_index"][-1] / sorting.sampling_frequency
    )

    for i, uid in enumerate(analyzer.unit_ids):
        st = analyzer.sorting.get_unit_spike_train_in_seconds(uid)
        isi = np.diff(st)
        bst = bsts[:, i]

        fr = len(st) / duration

        # ACG fit params
        acg = acgs[i]
        bins = analyzer.get_extension("correlograms").get_data()[1]
        lags = bins[:-1] + np.diff(bins) / 2
        lags_pos = lags[lags >= 0]
        acg_pos = acg[lags >= 0]
        try:
            params = fit_acg(lags_pos, acg_pos)
        except RuntimeError as e:
            warnings.warn(str(e))
            params = init_acg_fit_params(lags_pos, acg_pos)

        acg_baseline = np.mean(acg[(lags >= 40) & (lags <= 50)])
        acg_peak = np.mean(acg[(lags >= 0) & (lags <= 10)])
        burst_index_royer2012 = acg_peak / acg_baseline
        if burst_index_royer2012 > 0:
            burst_index_royer2012 /= acg_peak
        else:
            burst_index_royer2012 /= acg_baseline

        acg_pred = acg_fit(lags_pos, *params)

        features.append(
            {
                "unit_isi_mean": np.mean(isi),
                "unit_isi_median": np.median(isi),
                "unit_isi_var": np.median(isi),
                "unit_isi_var_coeff": np.std(isi) / np.mean(isi),
                "unit_isi_var_coeff2": np.mean(
                    2 * np.abs(isi[1:] - isi[:-1]) / (isi[1:] + isi[:-1])
                ),
                "unit_isi_burst_mizuseki2012": np.count_nonzero(isi < 0.006) / len(isi),
                "unit_acg_fit_c": params[0],
                "unit_acg_fit_t_refrac": params[1],
                "unit_acg_fit_tau_decay": params[2],
                "unit_acg_fit_d": params[3],
                "unit_acg_fit_tau_rise": params[4],
                "unit_acg_fit_h": params[5],
                "unit_acg_fit_tau_burst": params[6],
                "unit_acg_fit_rate_asymptote": params[7],
                "unit_acg_fit_r2": r2_score(acg_pos, acg_pred),
                "unit_acg_bust_index_royer2012": burst_index_royer2012,
                "unit_fr_var_coeff": np.var(bst) / fr,
                # "unit_fr_gini": gini(bst),
                "unit_fr_instability": np.mean(np.abs(np.diff(bst))) / fr,
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


def acg_fit(x, *params):

    def acg_fit_raw(
        x, c, t_refrac, tau_decay, d, tau_rise, h, tau_burst, rate_asymptote
    ):
        xtr = -(x - t_refrac)
        e_decay = np.exp(xtr / tau_decay)
        e_rise = np.exp(xtr / tau_rise)
        e_burst = np.exp(xtr / tau_burst)
        y = c * (e_decay - d * e_rise) + h * e_burst + rate_asymptote
        return y

    y = acg_fit_raw(x, *params)
    y = np.maximum(y, 0)
    return y


def acg_jac(x, *params):

    def acg_fit_raw(
        x, c, t_refrac, tau_decay, d, tau_rise, h, tau_burst, rate_asymptote
    ):
        xtr = -(x - t_refrac)
        e_decay = np.exp(xtr / tau_decay)
        e_rise = np.exp(xtr / tau_rise)
        e_burst = np.exp(xtr / tau_burst)
        y = c * (e_decay - d * e_rise) + h * e_burst + rate_asymptote
        return y

    def acg_jac_raw(
        x, c, t_refrac, tau_decay, d, tau_rise, h, tau_burst, rate_asymptote
    ):
        xtr = -(x - t_refrac)
        e_decay = np.exp(xtr / tau_decay)
        e_rise = np.exp(xtr / tau_rise)
        e_burst = np.exp(xtr / tau_burst)

        dc = e_decay - d * e_rise
        dt_refrac = (
            c * (e_decay / tau_decay - d * e_rise / tau_rise) + h * e_burst / tau_burst
        )
        dtau_decay = c * e_decay * xtr * (-1 / tau_decay**2)
        dd = -c * e_rise
        dtau_rise = c * d * e_rise * xtr * (-1 / tau_decay**2)
        dh = e_burst
        dtau_burst = h * e_burst * xtr * (-1 / tau_burst**2)
        drate_asymptote = np.ones_like(x)

        return np.vstack(
            [dc, dt_refrac, dtau_decay, dd, dtau_rise, dh, dtau_burst, drate_asymptote]
        ).T

    y = acg_fit_raw(x, *params)
    J = acg_jac_raw(x, *params)
    J[y < 0] = 0
    return J


def init_acg_fit_params(lags, acg):
    if len(acg[acg > 0]):
        init_r_asymp = acg[acg > 0][-1]
    else:
        init_r_asymp = 1
    init_t_refrac = max(5, lags[acg >= np.mean(acg)][0])
    init_c = max(np.max(acg), 1)
    init = [init_c, init_t_refrac, 20, 2.0, 1, 2.0, 1.5, init_r_asymp]
    return init


def fit_acg(lags, acg):
    init = init_acg_fit_params(lags, acg)
    bounds = (
        [0, 0, 1e-12, -np.inf, 1e-12, -np.inf, 1e-12, 0],
        [np.inf, np.max(lags), 1e3, np.inf, 1e3, np.inf, 1e3, max(1, np.max(acg))],
    )
    res = scipy.optimize.curve_fit(
        acg_fit,
        lags,
        acg,
        p0=init,
        bounds=bounds,
        jac=acg_jac,
        maxfev=int(1e3 * (len(acg) + 1)),
    )
    params = res[0]
    return params


def gini(x):
    """Compute Gini coefficient of array of values"""
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x) ** 2 * np.mean(x))
