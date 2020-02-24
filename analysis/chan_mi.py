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

    # Plot it
    plt.plot(peak['times'], mi.T)
    plt.plot(peak['times'], upper_quant.T, '-k')
    #plt.plot(perm_mi[99,:,:].T, '-k') # One example permutation
    plt.ylabel('MI (bits)')
    plt.xlabel('Time (s)')


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
    import mne
    import reconstruct_item
    import reconstruct_saccade

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


#### Sketchpad



def aggregate(analysis_type = 'sacc'):
    for n in [2,3,4,5,6]:
        plt.subplot(3, 2, n)
        plot_timecourse(n, analysis_type)
        plt.tight_layout()




# Cluster-based permutation test
# https://mne.tools/stable/auto_tutorials/stats-source-space/
   # plot_stats_cluster_spatio_temporal_repeated_measures_anova.html


# Get connectivity
ch_type = 'grad'
connectivity, ch_names = mne.channels.find_ch_connectivity(raw.info,
                                                           ch_type=ch_type)
plt.imshow(connectivity.toarray(), cmap='gray')

# Take only mags or grads
raw = load_raw(n) # Read in a raw data file to get the 'info' object
def ch_type_inx(ch_type):
    """ Get indexes of mags or grads in the MI array
    ch_type: 'mag'|'grad'
    """
    meg_info = mne.pick_info(raw.info, sel=mne.pick_types(raw.info, meg=True))
    picks = mne.pick_types(meg_info, meg=ch_type)
    return picks
ch_inx = ch_type_inx(ch_type)
x_emp = mi[:,ch_inx,:]
x_perm = perm_mi[:,ch_inx,:]

stat, clusters, pval, H0 = mne.stats.permutation_cluster_test(
        X=[x_emp, x_perm],
        n_permutations=100,
        tail=0,
        stat_fun=mne.stats.f_oneway,
        #out_type='indices',
        connectivity=connectivity,
        )

"""
stat_fun: Should take a list of arrays as arguments
    Could do a within-subjects regression by building a stat_fun that takes the
    design matrix into account.
"""

n_subjects = 30
n_permutations = 500

# Make the design matrix
conds_per_subj = [1] + ([0] * n_permutations) # 1=emp, 0=perm
x = pd.DataFrame({'subj': np.repeat(range(n_subjects), len(conds_per_subj)),
                  'cond': np.tile(conds_per_subj, n_subjects)})

# Simulate some data
y = np.random.normal(size=x.shape[0])
y += np.repeat(np.random.normal(size=n_subjects), len(conds_per_subj)) # Subject offsets
y += x['cond'] * 4 # Difference between conditions
x['y'] = pd.Series(y, index=x.index)

import statsmodels.api as sm
import statsmodels.formula.api as smf

md = smf.mixedlm("y ~ cond", x,
                 groups=x['subj'], # Random intercepts for each subject
                 re_formula="~cond") # Random slope for each subject
mdf = md.fit()
print(mdf.summary())

# Then return mdf.tvalues, and set some sensible threshold in mne

def lmer_statfun(X, design, param_name, **kwargs):
    """ X: array (observation, channel, time)
        design: pd.DataFrame
    """
    meg_data = np.reshape(X, [X.shape[0], -1) # Collapse later dims (other than obs)
    stat = []
    for i in range(x.shape[-1]):
        data = design
        data['y'] = pd.Series(meg_data[:,i], index=data.index)
        md = smf.mixedlm(data=data, **kwargs)
        mdf = md.fit()
        s = mdf.tvalues[param_name]
        stat.append(s)
    stat = np.reshape(stat, X.shape[1:]) # Get back to original dimensions
    return stat

def statfun(X):
    """ This would be passed to mne.stats.permutation_cluster_test
    """
    design = FILL_IN
    param_name = FILL_IN
    lmer_kwargs = FILL_IN
    fnc = lmer_statfun(X, design, param_name, **lmer_kwargs)
    return fnc
