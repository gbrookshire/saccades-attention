"""
Compute MI between data at each channel and some stimulus/response vector
Use this for channel selection
"""

import numpy as np
import gcmi
from tqdm import tqdm
from mne.externals.h5io import write_hdf5, read_hdf5


def sig_deriv(x):
    """ Make a combined signal of the raw signal and its derivative
    """
    x_der = np.diff(x, axis=0, prepend=x[0])
    x_comb = np.vstack([x, x_der])
    return x_comb


def mi_item(d):
    """ Compute MI between each MEG channel and some stimulus identity
    """
    meg_data = d['meg_data']
    categ = d['y']
    n_chans = meg_data.shape[1]
    n_timepoints = meg_data.shape[2]
    mi = np.full([n_chans, n_timepoints], np.nan)
    for i_chan in range(n_chans): #tqdm(range(n_chans)):
        for i_time in range(n_timepoints):
            x = sig_deriv(meg_data[:,i_chan,i_time])
            x_gc = gcmi.copnorm(x)
            m = gcmi.mi_model_gd(x_gc, categ, len(np.unique(categ)))
            mi[i_chan, i_time] = m
    return mi


def mi_saccade_dir(d):
    """ Compute MI between each MEG channel and saccade direction
    """
    meg_data = d['meg_data']
    # Put together the saccade data
    x_change, y_change = (d['fix_info'][f"{dim}_change"].copy().to_numpy()
                            for dim in ('x', 'y'))
    sacc = np.vstack([x_change, y_change])
    sacc_gc = gcmi.copnorm(sacc)
    # Compute MI foe each channel and timepoint
    n_chans = meg_data.shape[1]
    n_timepoints = meg_data.shape[2]
    mi = np.full([n_chans, n_timepoints], np.nan)
    for i_chan in range(n_chans): # tqdm(range(n_chans)):
        for i_time in range(n_timepoints):
            x = sig_deriv(meg_data[:, i_chan, i_time])
            x_gc = gcmi.copnorm(x)
            m = gcmi.mi_gg(x_gc, sacc_gc, demeaned=True)
            mi[i_chan, i_time] = m
    return mi


def permuted_mi(d, mi_fnc, k):
    """ Compute MI after randomly permuting the trials

    ***** This changes the data structure `d` in-place!
    """
    def shuff_mi(d):
        """ Convenience func to compute MI on shuffled data. """
        n_trials = d['meg_data'].shape[0]
        shuffle_inx = list(range(n_trials))
        np.random.shuffle(shuffle_inx)
        d['meg_data'] = d['meg_data'][shuffle_inx, :, :]
        return mi_fnc(d)

    results = [shuff_mi(d) for _ in range(k)]
    results = np.array(results)
    return results


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

    peak = {'times': d['times'],
            't_inx': t_peak_inx,
            'chan_order': chan_order,
            'chan_rank': chan_rank}
    return peak


def plot(n, analysis_type):
    """ Plot the topography of channels
    n: subject number
    analysis_type: item or sacc
    """
    # Read in a raw data file to get the 'info' object
    import pandas as pd
    from load_data import meg_filename
    subject_info = pd.read_csv(expt_info['data_dir'] + 'subject_info.csv',
                           engine='python', sep=',')
    fname = meg_filename(subject_info['meg'][n])
    raw = mne.io.read_raw_fif(fname)
    # Read the MI
    fname = f"{data_dir}mi_peak/{n}_{analysis_type}.h5"
    mi, peak = read_hdf5(fname)
    # Plot the topography
    mi_info = mne.pick_info(raw.info, sel=mne.pick_types(raw.info, meg=True)) 
    mi_info['sfreq'] = 100. # After downsampling
    mi_evoked = mne.EvokedArray(mi, mi_info, tmin=peak['times'][0])
    for ch_type in ('mag', 'grad', True): # Grads combined with RMS
        times = np.linspace(-0.2, 0.2, 9)
        mi_evoked.plot_topomap(times=times, #peak['times'][peak['t_inx']],
                            average=0.05, # Avg in time in a window this size
                            ch_type=ch_type,
                            vmin=0,
                            vmax=np.max,
                            contours=0,
                            cmap='viridis', # 'viridis'
                            units='Bits',
                            scalings=dict(grad=1, mag=1),
                            cbar_fmt='%1.3f')
                            #mask=(peak['chan_rank'] < 10).transpose())
                            #sensors=True
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
    import reconstruct_saccade
    expt_info = json.load(open('expt_info.json'))
    data_dir = expt_info['data_dir']

    try:
        n = int(sys.argv[1])
        analysis_type = sys.argv[2]
    except IndexError:
        n = int(input('Subject number: '))
        analysis_type = input('Analysis type (item/sacc): ')
    assert analysis_type in ('item', 'sacc')

    if analysis_type == 'item': # Get MI with item identity
        d = reconstruct_item.preproc(n)
        mi_fnc = mi_item
    elif analysis_type == 'sacc': # Get MI with item identity
        d = reconstruct_saccade.preproc(n)
        mi_fnc = mi_saccade_dir

    # Compute MI
    mi = mi_fnc(d)
    peak = find_peak(d, mi)
    fname = f"{data_dir}mi_peak/{n}_{analysis_type}.h5"
    write_hdf5(fname, [mi, peak], overwrite=True)

    # Compute permuted MI to see about chance levels
    k = 500
    perm_mi = permuted_mi(d, mi_fnc, k)
    fname = f"{data_dir}mi_peak/{n}_{analysis_type}_perm.h5"
    write_hdf5(fname, perm_mi, overwrite=True)

