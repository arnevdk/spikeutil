import neo
import pandas as pd
import quantities as pq


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
