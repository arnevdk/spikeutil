import neo
import numpy as np
import pandas as pd
import quantities as pq
import scipy.ndimage
import spikeinterface.extractors as se
from elephant.conversion import BinnedSpikeTrain
from elephant.kernels import GaussianKernel
from elephant.statistics import instantaneous_rate


def sorting_to_neo(sorting, t_stop=None, merge=False):
    # TODO deduplicate code by defining list of spiketrains first
    if t_stop is None:
        t_stop = sorting.to_spike_vector()[-1]["sample_index"]
        t_stop = sorting.sample_index_to_time(t_stop)
    else:
        sorting = sorting.time_slice(0,t_stop)

    sfreq = sorting.get_sampling_frequency()

    seg = neo.Segment()
    if not merge:
        for unit_id in sorting.get_unit_ids():
            spike_train = sorting.get_unit_spike_train_in_seconds(unit_id)
            spike_train_neo = neo.SpikeTrain(
                spike_train,
                t_stop=t_stop,
                units=pq.s,
                sampling_rate=sfreq * pq.Hz,
                name=unit_id,
            )
            seg.spiketrains.append(spike_train_neo)
    else:
        spike_trains = []
        for unit_id in sorting.get_unit_ids():
            spike_trains.append(sorting.get_unit_spike_train_in_seconds(unit_id))
        spike_train = np.concatenate(spike_trains)
        spike_train = np.sort(spike_train)
        spike_train_neo = neo.SpikeTrain(
            spike_train,
            t_stop=t_stop,
            units=pq.s,
            sampling_rate=sfreq * pq.Hz,
            name=unit_id,
        )
        seg.spiketrains.append(spike_train_neo)
 

    return seg


def spikes_as_df(sorting):
    # Unit index refers to relative index of the unit given dropped units, not to the original unit id
    spikes = sorting.to_spike_vector()
    df = pd.DataFrame(spikes)
    df["time"] = df["sample_index"] / sorting.get_sampling_frequency()
    return df


def binned_firing_rate(sorting, bin_width=0.005, t_stop=None, normalize_width=True):
    seg = sorting_to_neo(sorting)
    if t_stop is not None:
        t_stop = t_stop * pq.s
    bst = BinnedSpikeTrain(seg.spiketrains, bin_size=bin_width * pq.s, t_stop=t_stop)
    bst = bst.to_array().T
    if normalize_width:
        bst = bst / bin_width
    time = np.arange(len(bst)) * bin_width + bin_width / 2
    return time, bst


def inst_firing_rate(
    sorting, kernel_sigma=0.1, sfreq=500, coactivity=True, t_max=None, normalize=True
):
    n_units = len(sorting.unit_ids)
    t_end = sorting.get_times()[-1]
    bins = np.linspace(0, t_end, int(round(t_end*sfreq)))
    fr = np.empty((n_units,len(bins)-1))
    sigma = kernel_sigma * sfreq
    for i, unit_id in enumerate(sorting.unit_ids):
        st = sorting.get_unit_spike_train_in_seconds(unit_id)
        fr[i,:], bins = np.histogram(st, bins=bins)

    fr = fr.astype(np.float64)
    fr = scipy.ndimage.gaussian_filter1d(fr, sigma)

    if coactivity:
        fr =  np.mean(fr, axis=0)

    return fr, sfreq

# def inst_firing_rate(
#    sorting, kernel_sigma=0.1, sfreq=500, coactivity=False, t_max=None
# ):
#    if t_max is not None:
#        t_max = t_max * pq.s
#    seg = sorting_to_neo(sorting)
#    kernel = GaussianKernel(sigma=kernel_sigma * pq.s)
#
#    # Convert to single merged spiketrain to speed up
#    # elephant instantaneous_rate computation
#    if coactivity:
#        st = sorting.to_spike_vector()
#        st['unit_index']=0
#        sfreq = sorting.get_sampling_frequency()
#        sorting = se.NumpySorting.from_samples_and_labels(
#                st['sample_index'],
#                st['unit_index'],
#                sfreq,
#        )
#
#    fr = instantaneous_rate(
#        seg.spiketrains,
#        1 / sfreq * pq.s,
#        kernel=kernel,
#        t_stop=t_max,
#        pool_spike_trains=coactivity,
#    )
#    fr = np.array(fr).squeeze()
#    return fr, sfreq
