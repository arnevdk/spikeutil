import numpy as np
from elephant.spike_train_correlation import spike_time_tiling_coefficient
from joblib import Parallel, delayed

from spikeutil.core import sorting_to_neo


def fc_sttc(sorting):
    seg = sorting_to_neo(sorting)
    n_spiketrains = len(seg.spiketrains)
    idc = np.array(np.triu_indices(n_spiketrains, k=1)).T
    sttc_vals = Parallel(n_jobs=-1)(
        delayed(spike_time_tiling_coefficient)(seg.spiketrains[i], seg.spiketrains[j])
        for i, j in idc
    )

    sttc = np.eye(n_spiketrains)
    sttc[idc[:, 0], idc[:, 1]] = sttc_vals
    sttc[idc[:, 1], idc[:, 0]] = sttc_vals
    return sttc
