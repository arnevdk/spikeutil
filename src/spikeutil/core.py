import neo
import pandas as pd
import quantities as pq
from elephant.conversion import BinnedSpikeTrain
from elephant.kernels import GaussianKernel
from elephant.statistics import instantaneous_rate



def sorting_to_neo(sorting):

    t_stop = 0
    for unit_id in sorting.get_unit_ids():
        spike_train = sorting.get_unit_spike_train_in_seconds(unit_id)
        t_stop = max(t_stop, max(spike_train))

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


