import neo
import numpy as np
import pandas as pd
import quantities as pq
from elephant.conversion import BinnedSpikeTrain
from elephant.kernels import GaussianKernel
from elephant.statistics import instantaneous_rate
import spikeinterface.extractors as se
import scipy.ndimage


def sorting_to_neo(sorting):
    t_stop = sorting.to_spike_vector()[-1]['sample_index'] 
    t_stop = sorting.sample_index_to_time(t_stop)

    sfreq = sorting.get_sampling_frequency()

    seg = neo.Segment()
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
    sorting, kernel_sigma=0.1, sfreq=500, coactivity=True, t_max=None
):
    if not coactivity:
        raise NotImplemented

    st = sorting.to_spike_vector()
    st = sorting.sample_index_to_time(st['sample_index'])
    if t_max is None:
        t_max = st[-1]
    n_bins = int(t_max*sfreq)
    bins = np.linspace(0,t_max,n_bins)
    binned_fr,bins = np.histogram(st,bins=bins)
    sigma = kernel_sigma*sfreq
    binned_fr = binned_fr.astype(np.float64)
    inst_rate = scipy.ndimage.gaussian_filter1d(binned_fr,sigma)

    return inst_rate,sfreq



#def inst_firing_rate(
#    sorting, kernel_sigma=0.1, sfreq=500, coactivity=False, t_max=None
#):
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
