"""
RSA-type analysis

Ole says: I wonder if you could do a RSA type of analysis? I.e. correlate the
trials (correlation over sensors after a 1-100 Hz BP filter)  for when moving
to the same object (ie correlate all possible 'same' trial pairs and then
average). Also do the the correlation for trials when moving to different
objects; 'different trial pairs'. This is quite similar Lin's analysis and
might be a more sensitive.
"""


import sys
import json
import pickle
import itertools
import numpy as np
from itertools import combinations, permutations
import matplotlib.pyplot as plt
from tqdm import tqdm
import mne

import load_data
import artifacts

expt_info = json.load(open('expt_info.json')) 

def preprocess(n, lock_event='saccade', chan_sel='all', filt=[1, 30]):
    """
    lock_event: saccade or fixation
    chan_sel: all or grad or mag
    filt: 2-item sequence specifying a BP filter
    """
    
    d = load_data.load_data(n)
    
    if chan_sel == 'all':
        chan_sel = True

    if lock_event == 'fixation':
        event_key = 'fix_on'
    elif lock_event == 'saccade':
        event_key = 'fix_off'
    
    # Select events for segmentation
    row_sel = d['fix_events'][:,2] == expt_info['event_dict'][event_key] 
    events = d['fix_events'][row_sel, :] 
    
    # When locking to saccade onsets, we have to adjust for the fact that item
    # identity is tagged to saccade onset. This means shifting all the events
    # by 1.
    if lock_event == 'saccade':
        events = events[:-1,:]
        events = np.vstack([[0, 0, 200], events])

    # Only keep trials that didn't have another eye movement too recently
    prior_saccade_thresh = 250 # In samples (i.e. ms)
    prior_saccade_time = events[1:,0] - events[:-1,0]
    too_close = prior_saccade_time < 250
    too_close = np.hstack([[False], too_close])

    # Select fixations to a new object
    new_obj = np.diff(d['fix_info']['closest_stim']) != 0
    new_obj = np.hstack((True, new_obj)) # First fixation is to a new object

    # Apply the selections
    trial_sel = new_obj & ~too_close
    d['fix_info'] = d['fix_info'].loc[trial_sel]
    events = events[trial_sel,:]
    
    # Preprocess the data
    d['raw'].load_data()
    # Reject ICA artifacts
    d['ica'].apply(d['raw']) 
    # Filter the data
    d['raw'].filter(l_freq=filt[0], h_freq=filt[1], # band-pass filter 
                    method='fir', phase='minimum', # causal filter
                    n_jobs=5)
    # Epoch the data
    picks = mne.pick_types(d['raw'].info,
                           meg=chan_sel,
                           eeg=False, eog=False,
                           stim=False, exclude='bads')
    epochs = mne.Epochs(d['raw'],
                        events,
                        tmin=-1.0, tmax=0.5,
                        reject_by_annotation=True,
                        preload=True,
                        baseline=None,
                        picks=picks,
                        proj=True) 
    # Resample (do this after epoching to make sure trigger times are correct)
    epochs.resample(100, n_jobs=5) 
    # Reject trials that wre manually marked as bad
    meg_data = epochs.get_data()
    d['fix_info'] = d['fix_info'].iloc[epochs.selection] 
    # Reject trials with high global field power (GFP)
    bad_trials = artifacts.identify_gfp(meg_data, sd=4)
    meg_data = meg_data[~bad_trials,:,:]
    d['fix_info'] = d['fix_info'].iloc[~bad_trials] 
    # Add important fields to the data
    d['meg_data'] = meg_data
    d['times'] = epochs.times
    
    return d


def corr_analysis(d):
    """
    For each timepoint, check whether the spatial patterns are more similar
    between saccades toward the same item, compared with saccades toward
    different items.
    """
    x = d['meg_data']
    presaccade_item = d['fix_info']['prev_stim']
    postsaccade_item = d['fix_info']['closest_stim']

    # Exclude trials that don't haev a previous stim
    nans = np.isnan(presaccade_item)
    x = x[~nans,:,:]
    presaccade_item = presaccade_item[~nans].astype(np.int)
    postsaccade_item = postsaccade_item[~nans].astype(np.int)

    # Get the transition label of each trial. E.g. if one saccade goes from
    # item 1 to item 4, the label for that trial will be '1-4'
    trans_label = np.char.array(presaccade_item) + \
                    np.full(x.shape[0], b'-') + \
                    np.char.array(postsaccade_item)
    trans_label = trans_label.astype(str)

    ## # Check how many of each transition we have
    ## hist_labels, hist_counts = np.unique(trans_label, return_counts=True)
    ## plt.bar(range(len(hist_labels)), hist_counts) 
    ## plt.xticks(range(len(hist_labels)), hist_labels)
    ## plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=90)

    # Get all the unique 'same/diff' comparisons between transition types.
    # E.g. the saccades '1-5' and '2-5' are in the 'same' condition, whereas
    # '0-1' and '2-3' are in the 'different' condition. The conditions are
    # labeled in the function rsa_matrix().
    rsa_mat, transition_labels = rsa_matrix()
    def find_unique_comp_inx(val):
        comp_inx = np.nonzero(rsa_mat == val)
        comps = [(transition_labels[pre], transition_labels[post])
                    for pre,post in zip(*comp_inx)]
        return comps

    same_comps = find_unique_comp_inx(1)
    diff_comps = find_unique_comp_inx(-1)

    # Find all combinations of trials that are in one of these lists. This is a
    # list of all pairs of trials (by index).
    trial_combinations = ((n1, n2)
                            for n1 in range(x.shape[0])
                            for n2 in range(x.shape[0]))
    same_trial_inx = [] # Keep track of inx in the corr mat
    diff_trial_inx = []
    for trial_combo in trial_combinations: # For each combination of 2 trials
        if trial_combo[0] == trial_combo[1]:
            # Ignore fixations that have the same item 1 and item 2
            continue
        # Get the labels for this combination of trials
        label_combo = (trans_label[trial_combo[0]], trans_label[trial_combo[1]])
        # Check if that combination of labels is in the 'same' or 'diff' lists
        if label_combo in same_comps:
            same_trial_inx.append(trial_combo)
        elif label_combo in diff_comps:
            diff_trial_inx.append(trial_combo)

    ## # Do we get the expected ratio of 'different' to 'same' comparisons?
    ## # It should be about 3:1 to mirror the RSA matrix
    ## n_diff = len(diff_trial_inx)
    ## n_same = len(same_trial_inx)
    ## print(n_diff)
    ## print(n_same)
    ## print(n_diff / n_same)

    # For each timepoint, get the difference between same- and diff- trials
    same_corr_timecourse = []
    diff_corr_timecourse = []
    for i_time in tqdm(range(x.shape[2])):
        # Get the correlations of all spatial patterns at this timepoint
        c = np.corrcoef(x[:,:,i_time])

        # Pull out the correlations between pairs of saccades in the 'same' and
        # 'different' conditions
        same_corr = c[tuple(zip(*same_trial_inx))]
        diff_corr = c[tuple(zip(*diff_trial_inx))]

        # Average across all these correlations
        same_corr = same_corr.mean()
        diff_corr = diff_corr.mean()

        # Keep track of this averaged value in the timecourse
        same_corr_timecourse.append(same_corr)
        diff_corr_timecourse.append(diff_corr)

    return same_corr_timecourse, diff_corr_timecourse


def test_corr_analysis():
    """
    Test the analysis on simulated data
    """

    n_trials = 200
    n_labels = 6
    n_channels = 100
    n_timepoints = 150

    # Make spatio-temporal patterns of activity for each response
    # Make each pattern only appear at the middle of the timepoints
    # This simulation only makes response-induced patterns -- no predictions
    from scipy import stats
    pattern_env = np.zeros(n_timepoints)
    env_width = int(n_timepoints / 5)
    env_midpoint = int(n_timepoints / 2)
    env = stats.norm.pdf(range(env_width*2), env_width, env_width/4)
    pattern_env[env_midpoint-env_width:env_midpoint+env_width] = env
    patterns = []
    for _ in range(n_labels):
        p = np.random.normal(size=(n_channels, n_timepoints)) * pattern_env
        patterns.append(p)

    # Simulate the data from the spatio-temporal patterns
    x = np.zeros((n_trials, n_channels, n_timepoints))
    trial_labels = np.random.choice(n_labels, size=n_trials)
    for i_trial, lab in enumerate(trial_labels):
        x[i_trial,:,:] = patterns[lab]
    # Add some noise to the patterns
    noise_strength = 0.1
    x = x + np.random.normal(size=x.shape, scale=noise_strength)

    # Build the data structure
    d = {}
    d['meg_data'] = x
    d['fix_info'] = {}
    d['fix_info']['closest_stim'] = trial_labels
    d['fix_info']['prev_stim'] = np.hstack([[np.nan], trial_labels[:-1]])

    same_coef, diff_coef = corr_analysis(d)
    plt.plot(same_coef)
    plt.plot(diff_coef)
    plt.show()


def aggregate():
    import everyone
    chan_sel = 'grad' # grad or mag or all
    lock_event = 'saccade' # fixation or saccade
    filt = (1, 30) 
    filt = f"{filt[0]}-{filt[1]}"
    data_dir = expt_info['data_dir']
    def load_rsa(row):
        n = row['n']
        dd = 'no_close_saccades' # Which version to use
        fname = f"{data_dir}rsa/{dd}/{n}_{chan_sel}_{lock_event}_{filt}.h5"
        res = mne.externals.h5io.write_hdf5(fname)
        return res
    results = everyone.apply_fnc(load_rsa)
    same_coef, diff_coef, times = zip(*results)
    same_coef = np.array(same_coef)
    diff_coef = np.array(diff_coef)

    # Plot the averages
    plt.subplot(2, 1, 1)
    same_mean = np.mean(same_coef, axis=0)
    diff_mean = np.mean(diff_coef, axis=0)
    plt.plot(times[0], same_mean, '-r')
    plt.plot(times[0], diff_mean, '-k')
    plt.text(-0.8, same_mean.max() * 0.9, 'Same', color='r')
    plt.text(-0.8, same_mean.max() * 0.8, 'Diff', color='k')
    plt.xlabel('Time (s)')
    plt.ylabel('$R^2$')

    # Plot the difference between same and diff trials
    plt.subplot(2, 1, 2)
    same_minus_diff = same_coef - diff_coef
    plt.plot(times[0], same_minus_diff.transpose(), '-k', alpha=0.3)
    plt.axhline(y=0, linestyle='--', color='k')
    plt.axvline(x=0, linestyle='--', color='k')
    plt.plot(times[0], np.mean(same_minus_diff, axis=0), '-b')
    plt.xlabel('Time (s)')
    plt.ylabel('Same - Diff')

    plt.tight_layout()
    fname = f"{data_dir}plots/rsa/rsa_{chan_sel}_{lock_event}_{filt}.png"
    #plt.savefig(fname)
    plt.show()


def rsa_matrix(plot=False):
    """ Construct the matrix of which transitions are considered 'same' or
    'different' for the RSA analysis.

    Different: Saccades in which neither the pre-saccade or post-saccade items
    are the same. Don't compare A-->B to B-->C. Instead, only make comparisons
    like A-->B and C-->D.

    Same: Saccades in which the pre-saccade item is different but the
    post-saccade item is the same.
    """
    items = range(6)
    transitions = list(permutations(items, 2))
    # Sort by post-saccade item
    transitions = sorted(transitions, key=lambda x: (x[1], x[0])) 
    transition_labels = [f'{e[0]}-{e[1]}' for e in transitions]
    rsa_mat = np.zeros([len(transitions)] * 2)
    for i_x, t_x in enumerate(transitions):
        for i_y, t_y in enumerate(transitions):
            if len(set(t_x + t_y)) == 4: # All 4 items are different
                rsa_mat[i_y, i_x] = -1
            elif (t_x[0] != t_y[0]) and (t_x[1] == t_y[1]): # Diff pre same post
                rsa_mat[i_y, i_x] = 1
            else:
                pass # Neither a "same" nor a "different" saccade

    if plot:
        plt.imshow(rsa_mat, cmap='bwr')
        plt.xticks(range(len(transitions)), transition_labels)
        plt.gca().xaxis.tick_top()
        plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=90)
        plt.yticks(range(len(transitions)), transition_labels)

        fname = f"{expt_info['data_dir']}plots/rsa/rsa_matrix.png"
        plt.savefig(fname)

        print(f"{np.sum(rsa_mat == 1)} types of 'same' saccades")
        print(f"{np.sum(rsa_mat == -1)} types of 'different' saccades")
        print(f"{np.sum(rsa_mat == 0)} types of ignored saccades")

    return rsa_mat, transition_labels


if __name__ == '__main__':
    n = int(sys.argv[1])
    
    chan_sel = 'all'
    filt = [1, 30]
    lock_event = 'saccade'
    print(n, filt, chan_sel, lock_event)
    d = preprocess(n, lock_event, chan_sel, filt)
    same_coef, diff_coef = corr_analysis(d)

    data_dir = expt_info['data_dir']
    fname = f"{data_dir}rsa/{n}_{chan_sel}_{lock_event}_{filt[0]}-{filt[1]}.h5"
    mne.externals.h5io.write_hdf5(fname, [same_coef, diff_coef, d['times']])


