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
import matplotlib.pyplot as plt
from tqdm import tqdm
import mne

import load_data
import artifacts

expt_info = json.load(open('expt_info.json')) 

def preprocess(n):
    
    d = load_data.load_data(n)
    
    locking_event = 'saccade' # fixation or saccade
    if locking_event == 'fixation':
        event_key = 'fix_on'
    elif locking_event == 'saccade':
        event_key = 'fix_off'
    
    # Select events for segmentation
    row_sel = d['fix_events'][:,2] == expt_info['event_dict'][event_key] 
    events = d['fix_events'][row_sel, :] 
    
    # When locking to saccade onsets, we have to adjust for the fact that item
    # identity is tagged to saccade onset. This means shifting all the events
    # by 1.
    if locking_event == 'saccade':
        events = events[:-1,:]
        events = np.vstack([[0, 0, 200], events])
    
    # Select fixations to a new object
    new_obj = np.diff(d['fix_info']['closest_stim']) != 0
    new_obj = np.hstack((True, new_obj)) # First fixation is to a new object
    d['fix_info'] = d['fix_info'].loc[new_obj]
    events = events[new_obj,:]
    
    # Preprocess the data
    d['raw'].load_data()
    # Reject ICA artifacts
    d['ica'].apply(d['raw']) 
    # Filter the data
    d['raw'].filter(l_freq=1, h_freq=100, # band-pass filter 
                    method='fir', phase='minimum', # causal filter
                    n_jobs=5)
    # Epoch the data
    picks = mne.pick_types(d['raw'].info,
                        meg=True, eeg=False, eog=False,
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
    
    # Get the feature to reconstruct
    # In this case, it's the stimulus label
    d['y'] = d['fix_info']['closest_stim']
    d['y'] = d['y'].astype(int).to_numpy() 
    
    return d


def corr_analysis(x, labels):
    """
    For each timepoint, check whether the spatial
    x: Brain data (trial x channel x time)
    labels: Label for each trial
    """
    unique_labels = np.unique(labels)

    same_coef = []
    diff_coef = []
    for i_time in tqdm(range(x.shape[-1])):
        # Get correlations between each trial
        # How similar are the scalp topographies across trials?
        x_t = x[:,:,i_time]
        c = np.corrcoef(x_t) # Correlation matrix
        # Then go through each label and compare corr coefs of
        # same-label vs different-label trials

        # Get all trials with the same label
        same_label_coef = []
        for lab in unique_labels:
            # Get all trials with this label
            lab_inx = labels == lab
            lab_inx = np.nonzero(lab_inx)[0]
            # Get all combinations of these trials
            trial_combos = itertools.combinations(lab_inx, 2)
            trial_combos = tuple(zip(*trial_combos)) # Arrange for indexing
            # Extract all these corr coefs
            same_label_coef.extend(c[trial_combos])
        same_coef.append(np.mean(same_label_coef))

        # Get all trials with different labels
        diff_combos = [] 
        for lab_combo in itertools.combinations(unique_labels, 2):
            inx = [np.nonzero(labels == lab)[0].tolist() for lab in lab_combo]
            label_combos = list(zip(*inx)) # Make into pairs
            diff_combos.extend(label_combos)
        diff_combos = tuple(zip(*diff_combos)) # Arrange for indexing
        diff_label_coef = c[diff_combos]
        diff_coef.append(np.mean(diff_label_coef))
            
    return same_coef, diff_coef


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

    same_coef, diff_coef = corr_analysis(x, trial_labels)
    plt.plot(same_coef)
    plt.plot(diff_coef)
    plt.show()


def aggregate():
    import everyone
    locking_event = 'fixation' # fixation or saccade
    def load_rsa(row):
        n = row['n']
        fname = f"{expt_info['data_dir']}rsa/{locking_event}/{n}.pkl"
        res = pickle.load(open(fname, 'rb'))
        return res
    results = everyone.apply(load_rsa)
    same_coef, diff_coef, times = zip(*results)
    #plt.figure()
    #for i_subj in range(len(same_coef)):
    #    plt.subplot(2, 3, i_subj + 1)
    #    plt.plot(times[i_subj], same_coef[i_subj], '-r')
    #    plt.plot(times[i_subj], diff_coef[i_subj], '-k')
    #    plt.xlabel('Time (s)')
    #    plt.ylabel('$R^2$')
    #plt.tight_layout()

    # Plot the averages
    #plt.figure()
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
    same_minus_diff = np.array(same_coef) - np.array(diff_mean)
    plt.plot(times[0], same_minus_diff.transpose(), '-k', alpha=0.3)
    plt.axhline(y=0, linestyle='--', color='k')
    plt.axvline(x=0, linestyle='--', color='k')
    plt.plot(times[0], np.mean(same_minus_diff, axis=0), '-b')
    plt.xlabel('Time (s)')
    plt.ylabel('Same - Diff')

    plt.tight_layout()



if __name__ == '__main__':
    try:
        n = sys.argv[1]
    except IndexError:
        n = input('Subject number: ')
    n = int(n)
    d = preprocess(n)
    same_coef, diff_coef = corr_analysis(d['meg_data'], d['y'])

    fname = f"{expt_info['data_dir']}rsa/{n}.pkl"
    pickle.dump([same_coef, diff_coef, d['times']], open(fname, 'wb'))


