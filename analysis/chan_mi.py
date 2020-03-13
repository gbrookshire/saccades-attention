"""
Compute MI between data at each channel and some stimulus/response vector
Use this for channel selection
"""

import json
import numpy as np
import gcmi
from tqdm import tqdm
import matplotlib.pyplot as plt
import mne
from mne.externals.h5io import write_hdf5, read_hdf5

expt_info = json.load(open('expt_info.json'))
data_dir = expt_info['data_dir']

plt.ion()


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


def mi_continuous(d, y):
    """ Compute MI between each MEG channel and some continuous variable
    """
    meg_data = d['meg_data']
    y_gc = gcmi.copnorm(y)
    # Compute MI foe each channel and timepoint
    n_chans = meg_data.shape[1]
    n_timepoints = meg_data.shape[2]
    mi = np.full([n_chans, n_timepoints], np.nan)
    for i_chan in range(n_chans): # tqdm(range(n_chans)):
        for i_time in range(n_timepoints):
            x = sig_deriv(meg_data[:, i_chan, i_time])
            x_gc = gcmi.copnorm(x)
            m = gcmi.mi_gg(x_gc, y_gc, demeaned=True)
            mi[i_chan, i_time] = m
    return mi


def mi_saccade_dir(d):
    """ Compute MI between each MEG channel and saccade direction
    """
    # FIXME --- NOT WORKING YET
    # Put together the saccade data
    x_change, y_change = (d['fix_info'][f"{dim}_change"].copy().to_numpy()
                                for dim in ('x', 'y'))
    sacc = np.vstack([x_change, y_change])
    mi = mi_continuous(d, sacc)
    return mi


def mi_saccade_side(d):
    """ Compute MI between each MEG channel and saccade leftward/rightward
    """
    x_change = d['fix_info']["x_change"].copy().to_numpy()
    side = ['left' if x < 0 else 'right' for x in x_change]
    d['y'] = side
    mi = mi_item(d)
    return mi


def mi_abs_loc(d):
    """ Compute MI between each MEG channel and absolute allocentric location
    """
    # Put together the saccade data
    loc = np.vstack([d['fix_info']['x_avg'],
                     d['fix_info']['y_avg']])
    mi = mi_continuous(d, loc)
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

    """
    For saccades, this is not what we want to do. We want to select channels
    with the most information in the pre-stimulus times, because post-stimulus
    times will be dominated by activity related to movement of the eyes.
    """


def load_raw(n):
    """ Load the raw meg data for one subject
    """
    import pandas as pd
    from load_data import meg_filename
    subject_info = pd.read_csv(expt_info['data_dir'] + 'subject_info.csv',
                           engine='python', sep=',')
    fname = meg_filename(subject_info['meg'][n])
    raw = mne.io.read_raw_fif(fname)
    return raw


def plot_timecourse(n, analysis_type):
    """ Plot the timecourse of MI at each channel, along with a high upper
    quantile to see how the MI differs from what you would expect by chance.
    """
    # Read the MI
    mi, peak = read_hdf5(f"{data_dir}mi_peak/{n}_{analysis_type}.h5")
    perm_mi = read_hdf5(f"{data_dir}mi_peak/{n}_{analysis_type}_perm.h5")

    # Show an upper percentile of the permuted data
    quantile = 0.99
    q = 1 - ((1 - quantile)/ 2)
    upper_quant = np.squeeze(np.quantile(perm_mi, q, axis=0))

    # Read in a raw data file to get the 'info' object
    raw = load_raw(n)
    mi_info = mne.pick_info(raw.info, sel=mne.pick_types(raw.info, meg=True)) 
    mi_info['sfreq'] = 100. # After downsampling
    mi_evoked = mne.EvokedArray(mi, mi_info, tmin=peak['times'][0])
    fig = mi_evoked.plot(spatial_colors=True, unit=False)
    fig.axes[0].set_ylabel('Bits')
    fig.axes[1].set_ylabel('Bits')
    # Plot the upper percentile of permuted data
    ### Still need to adjust this to select mags/grads
    #fig.axes[0].plot(peak['times'], upper_quant.T, 'k', zorder=1000)

    # # Plot it on a normal axis
    # plt.plot(peak['times'], mi.T)
    # plt.plot(peak['times'], upper_quant.T, '-k')
    # #plt.plot(perm_mi[99,:,:].T, '-k') # One example permutation
    # plt.ylabel('MI (bits)')
    # plt.xlabel('Time (s)')


def plot_topo(n, analysis_type):
    """ Plot the topography of channels
    n: subject number
    analysis_type: item or sacc
    """
    # Read in a raw data file to get the 'info' object
    raw = load_raw(n)
    # Read the MI
    fname = f"{data_dir}mi_peak/{n}_{analysis_type}.h5"
    mi, peak = read_hdf5(fname)
    # Plot the topography
    mi_info = mne.pick_info(raw.info, sel=mne.pick_types(raw.info, meg=True)) 
    mi_info['sfreq'] = 100. # After downsampling
    mi_evoked = mne.EvokedArray(mi, mi_info, tmin=peak['times'][0])
    for ch_type in ('mag', 'grad'): # Grads combined with RMS
        times = np.linspace(-0.2, 0.2, 9)
        mi_evoked.plot_topomap(times=times, #peak['times'][peak['t_inx']],
                            average=0.05, # Avg in time in a window this size
                            ch_type=ch_type,
                            vmin=0,
                            vmax=np.max,
                            contours=0,
                            cmap='viridis',
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


def plot_topo_avg(analysis_type):
    """ Plot average topography over subjects
    """
    import everyone
    def load_mi(row):
        fname = f"{data_dir}mi_peak/{row['n']}_{analysis_type}.h5"
        mi, peak = read_hdf5(fname)
        return mi, peak

    mi_all, peak_all = list(zip(*everyone.apply_fnc(load_mi)))
    mi_avg = np.mean(mi_all, axis=0)

    # Rectify and log-transform
    mi_avg[mi_avg < 0] = 1e-6
    #mi_avg = np.log10(mi_avg)
    
    # Read in a raw data file to get the 'info' object
    raw = load_raw(2)

    # Plot the topography
    mi_info = mne.pick_info(raw.info, sel=mne.pick_types(raw.info, meg=True)) 
    mi_info['sfreq'] = 100. # After downsampling
    mi_evoked = mne.EvokedArray(mi_avg, mi_info, tmin=peak['times'][0])
    for ch_type in ('mag', 'grad'): # Grads combined with RMS
        times = np.linspace(-0.2, 0.2, 9)
        mi_evoked.plot_topomap(times=times, #peak['times'][peak['t_inx']],
                            average=0.05, # Avg in time in a window this size
                            ch_type=ch_type,
                            vmin=0,
                            vmax=np.max,
                            contours=0,
                            cmap='viridis',
                            units='Bits',
                            scalings=dict(grad=1, mag=1),
                            cbar_fmt='%1.3f')

    # Plot MI in each sensor
    lay = mne.channels.layout.find_layout(mi_info)
    times = peak_all[0]['times']
    t_sel = (-0.4 < times) & (times < -0.0)
    m = np.mean(mi_avg[:, t_sel], axis=1) # Avg MI over the time range
    m -= np.min(m) # Set minimum value to 0
    m /= np.max(m) # Set maximum value to 1
    cmap = plt.cm.Purples
    colors = cmap(m)
    plt.figure(figsize=(6, 6))
    for i_chan in range(lay.pos.shape[0]):
        plt.plot(lay.pos[i_chan,0], lay.pos[i_chan,1], 'o',
                color=colors[i_chan,:],
                markersize=9)
    plt.box(False)
    plt.xticks([])
    plt.yticks([])


if __name__ == '__main__':
    import sys
    import mne
    import reconstruct_item
    import reconstruct_saccade

    try:
        n = int(sys.argv[1])
        analysis_type = sys.argv[2]
    except IndexError:
        n = int(input('Subject number: '))
        analysis_type = input('Analysis type (item/sacc/loc): ')
    assert analysis_type in ('item', 'sacc', 'sacc_side', 'loc')

    if analysis_type == 'item': # Get MI with item identity
        d = reconstruct_item.preproc(n)
        mi_fnc = mi_item
    elif analysis_type == 'sacc': # Get MI with saccade direction
        d = reconstruct_saccade.preproc(n)
        mi_fnc = mi_saccade_dir
    elif analysis_type == 'sacc_side': # MI with saccade side (left vs right)
        d = reconstruct_saccade.preproc(n)
        mi_fnc = mi_saccade_side
    elif analysis_type == 'loc': # Get MI with allocentric location
        d = reconstruct_saccade.preproc(n)
        mi_fnc = mi_abs_loc

    # Compute MI
    mi = mi_fnc(d)
    peak = find_peak(d, mi)
    fname = f"{data_dir}mi_peak/{analysis_type}/{n}.h5"
    write_hdf5(fname, [mi, peak], overwrite=True)

    # Compute permuted MI to see about chance levels
    k = 500
    perm_mi = permuted_mi(d, mi_fnc, k)
    fname = f"{data_dir}mi_peak/{analysis_type}/{n}_perm.h5"
    write_hdf5(fname, perm_mi, overwrite=True)

