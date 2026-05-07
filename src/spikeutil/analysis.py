import numpy as np
import quantities as pq
import scipy.signal
from elephant.conversion import BinnedSpikeTrain
from elephant.kernels import GaussianKernel
from elephant.statistics import instantaneous_rate

from spikeutil.core import sorting_to_neo


def binned_spike_train(sorting, bin_width=0.005, t_stop=None, normalize_width=True):
    seg = sorting_to_neo(sorting)
    if t_stop is not None:
        t_stop = t_stop * pq.s
    bst = BinnedSpikeTrain(seg.spiketrains, bin_size=bin_width * pq.s, t_stop=t_stop)
    bst = bst.to_array().T
    if normalize_width:
        bst = bst / bin_width
    time = np.arange(len(bst)) * bin_width + bin_width / 2
    return time, bst


def firing_rate(
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


def firing_rate_psd(fr, sfreq, nperseg=2**14, normalize=True):
    f, Pxx_den = scipy.signal.welch(fr, sfreq, nperseg=nperseg)
    if normalize:
        Pxx_den = Pxx_den / np.var(fr)
    return f, Pxx_den
