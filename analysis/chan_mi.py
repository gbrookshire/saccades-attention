"""
Compute MI between data at each channel and some stimulus/response vector
Use this for channel selection
"""

import numpy as np
import gcmi
from tqdm import tqdm


def mi_by_chan(meg_data, labels):
    """ Compute MI between each MEG channel and some categorical variable.
    """
    n_chans = meg_data.shape[1]
    n_timepoints = meg_data.shape[2]
    mi = np.full([n_chans, n_timepoints], np.nan)
    for i_chan in tqdm(range(n_chans)):
        for i_time in range(n_timepoints):
            x = meg_data[:,i_chan,i_time]
            x_gc = gcmi.copnorm(x)
            m = gcmi.mi_model_gd(x_gc, labels, len(np.unique(labels)))
            mi[i_chan, i_time] = m
    return mi


def find_peak(d, mi):
    """ Given a timecourse of MI, find the channels and timepoints of peak MI.
    """
    # Find the timepoint of maximum activation
    rms = lambda x, axis: np.sqrt(np.mean(x ** 2, axis))
    rms_mi = rms(mi, axis=0)
    t_peak_inx = np.argmax(rms_mi)
    t_peak = d['times'][t_peak_inx]

    # Find the channels with maximum activation
    peak_width = 3 # in samples (downsampled)
    mi_peak = mi[:, (t_peak_inx-peak_width):(t_peak_inx+peak_width)]
    mi_peak = np.mean(mi_peak, axis=-1)
    chan_rank = np.argsort(mi_peak)
    chan_rank = chan_rank[::-1] # Reverse list - best chans are first
    meg_chan_names = [e for e in d['raw'].info['ch_names']
                        if e.startswith('MEG')]
    meg_chan_names = np.array(meg_chan_names)
    chan_order = meg_chan_names[chan_rank].tolist()

    return t_peak, chan_order, chan_rank


def plot(d, mi, t_peak_inx, chan_rank):
    # Plot the topography of channels
    mi_info = mne.pick_info(d['raw'].info,
                            sel=mne.pick_types(d['raw'].info, meg=True)) 
    mi_evoked = mne.EvokedArray(mi, mi_info, tmin=d['times'][0])
    for ch_type in ('mag', 'grad'): # Grads combined with RMS
        mi_evoked.plot_topomap(times=d['times'][t_peak_inx],
                            average=0.05, # Average in time in a window this size
                            ch_type=ch_type,
                            vmin=0,
                            vmax=np.max,
                            contours=0,
                            cmap='viridis', # 'viridis'
                            units='Bits',
                            scalings=dict(grad=1, mag=1),
                            cbar_fmt='%1.3f',
                            mask=(chan_rank < 10).transpose())
                            #sensors=True,
                            #mask=FILL_IN,
                            #mask_params=dict(marker='o',
                            #                 markerfacecolor='w',
                            #                 markeredgecolor='w',
                            #                 linewidth=0,
                            #                 markersize=4)


if __name__ == '__main__':
    import sys
    import json
    import mne
    import reconstruct_item
    expt_info = json.load(open('expt_info.json'))

    try:
        n = int(sys.argv[1])
    except IndexError:
        n = int(input('Subject number: '))

    d = reconstruct_item.preproc(n)
    mi = mi_by_chan(d['meg_data'], d['y'])
    t_peak, chan_order, chan_rank = find_peak(d, mi)

    data_dir = expt_info['data_dir']
    fname = f"{data_dir}mi_peak/{n}_item.h5"
    mne.externals.h5io.write_hdf5(fname,
                                  [mi, t_peak, chan_order, chan_rank],
                                  overwrite=True)

