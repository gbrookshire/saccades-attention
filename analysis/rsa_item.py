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
import itertools
import socket
import numpy as np
import pandas as pd
from itertools import combinations, permutations
import matplotlib.pyplot as plt
from tqdm import tqdm
import mne

import load_data
import artifacts
import dist_convert as dc

expt_info = json.load(open('expt_info.json')) 

if socket.gethostname() == 'colles-d164179':
    data_dir = expt_info['data_dir']['external']
else:
    data_dir = expt_info['data_dir']['standard']
    

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

    # # Only keep trials that didn't have another eye movement too recently
    # prior_saccade_thresh = 250 # In samples (i.e. ms)
    # prior_saccade_time = events[1:,0] - events[:-1,0]
    # too_close = prior_saccade_time < 250
    # too_close = np.hstack([[False], too_close])

    # Select fixations to a new object
    new_obj = np.diff(d['fix_info']['closest_stim']) != 0
    new_obj = np.hstack((True, new_obj)) # First fixation is to a new object

    # # Select fixations to nearby objects
    # distance_threshold = 4.0 # visual degrees
    # x_avg = d['fix_info']['x_avg'].to_numpy()
    # y_avg = d['fix_info']['y_avg'].to_numpy()
    # x_change = x_avg[1:] - x_avg[:-1]
    # y_change = y_avg[1:] - y_avg[:-1]
    # sacc_dist = (x_change ** 2 + y_change ** 2) ** (1/2)
    # sacc_dist = dc.pix2deg(sacc_dist)
    # #plt.hist(sacc_dist, 50)
    # close_saccades = sacc_dist < distance_threshold
    # close_saccades = np.hstack([False, close_saccades])

    # Apply the selections
    trial_sel = new_obj # & ~too_close & ~close_saccades
    d['fix_info'] = d['fix_info'].loc[trial_sel]
    events = events[trial_sel,:]
    print(f"Trials kept in the analysis: {trial_sel.sum()}")

    # Get the time of the next saccade onset, and whether it's to a new item
    t_this_fix_onset = d['fix_info']['start_meg']
    t_next_sac_onset = np.hstack([d['fix_info']['start_meg'][1:], [np.nan]])
    fix_dur = t_next_sac_onset - t_this_fix_onset
    d['fix_info']['fix_dur'] = pd.Series(fix_dur.to_numpy(),
                                         index=d['fix_info'].index)

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
                        tmin=-1.0, tmax=1.0,
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
    # # Load info about which channels carry the most info about stim identity
    # fname = f"{data_dir}mi_peak/{n}_item.h5"
    # mi, t_peak, chan_order, chan_rank = mne.externals.h5io.read_hdf5(fname)
    # # Select grads/mags in the MI ordering
    # n_top_chans = 20 # How many channels to keep in the analysis
    # keep_chans = np.isin(epochs.info['ch_names'], chan_order[:n_top_chans])
    # d['mi_keep_chans'] = keep_chans
    
    return d


def corr_analysis(d):
    """
    For each timepoint, check whether the spatial patterns are more similar
    between saccades toward the same item, compared with saccades toward
    different items.
    """
    x = d['meg_data']
    ###x = x[:, d['mi_keep_chans'], :]
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

    # # Another way to look at predictive processing.
    # # If people saccade from A -> B -> C, can we decode at fixation onset at B
    # # that they are about to look to C?
    # trans_label = np.hstack([trans_label[1:], np.nan])

    # # "Retrospective" processing
    # # When people saccade from A -> B, does some activity from A linger (or get
    # # reactivated) when they fixate on B?
    # trans_label = np.hstack([np.nan, trans_label[:-1]])

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

    # Standard version
    # For each timepoint, get the difference between same- and diff- trials
    same_corr_timecourse = []
    diff_corr_timecourse = []
    for i_time in tqdm(range(x.shape[2])):

        # # ~~~ Normal version ~~~
        # # Get the correlations of all spatial patterns at this timepoint
        # c = np.corrcoef(x[:,:,i_time])

        # ~~~ Compare spatial patterns in perception vs reactivation
        nan_trial = np.full(x.shape[1], np.nan)
        percep_data = x[:,:,i_time]
        reactiv_data = np.vstack([x[1:,:,i_time], nan_trial])
        c = _corrcoef_by_column(percep_data.T, reactiv_data.T)

        # Pull out the correlations between pairs of saccades in the 'same' and
        # 'different' conditions
        same_corr = c[tuple(zip(*same_trial_inx))]
        diff_corr = c[tuple(zip(*diff_trial_inx))]

        # Average across all these correlations
        same_corr = np.nanmean(same_corr)
        diff_corr = np.nanmean(diff_corr)

        # Keep track of this averaged value in the timecourse
        same_corr_timecourse.append(same_corr)
        diff_corr_timecourse.append(diff_corr)

    return same_corr_timecourse, diff_corr_timecourse


def simple_reactivation(d):
    """ Check whether activity at item N+1 is similar to activity at item N
    """

    x = d['meg_data'].copy()

    # Get the transition labels to ensure that the randomly permuted
    # correlations don't come from or go to the same item as the compared item.
    # This helps make the randomly permuted data comparable to the experimental
    # data.
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

    # Find the correlations
    corr_timecourse_real = []
    corr_timecourse_rand = []
    for i_time in tqdm(range(x.shape[2])):
        corr_by_trial_real = []
        corr_by_trial_rand = []
        for i_trial in range(1, x.shape[0]):
            # Get the real correlation
            c_real = np.corrcoef(x[i_trial, :, i_time],
                                 x[i_trial-1, :, i_time])[0,1]
            corr_by_trial_real.append(c_real)
            # Get the random correlation
            curr_trans = trans_label[i_trial] # Which stims did this fix go b/w
            diff_trans = np.nonzero(trans_label != curr_trans)[0] # Diff stims
            rand_trial = np.random.choice(diff_trans)
            c_rand = np.corrcoef(x[i_trial, :, i_time],
                                 x[rand_trial, :, i_time])[0,1]
            corr_by_trial_rand.append(c_rand)
        corr_timecourse_real.append(np.mean(corr_by_trial_real))
        corr_timecourse_rand.append(np.mean(corr_by_trial_rand))

    plt.plot(d['times'], corr_timecourse_real, label='corr(n, n+1)')
    plt.plot(d['times'], corr_timecourse_rand, label='corr(n, rand)')
    plt.axhline(y=0, linestyle='--', color='k')
    plt.ylabel('R')
    plt.xlabel('Time (s)')
    plt.legend()


def _corrcoef_by_column(O, P):
    """ Find coorrelations between each pair of columns b/w O and P
    Adapted from https://github.com/ikizhvatov/efficient-columnwise-correlation
    """
    (n, t) = O.shape      # n traces of t samples
    (n_bis, m) = P.shape  # n predictions for each of m candidates
    # Compute O - mean(0) and P - mean(P)
    DO = O - (np.einsum("nt->t", O, optimize='optimal') / np.double(n))
    DP = P - (np.einsum("nm->m", P, optimize='optimal') / np.double(n))
    cov = np.einsum("nm,nt->mt", DP, DO, optimize='optimal')
    varP = np.einsum("nm,nm->m", DP, DP, optimize='optimal')
    varO = np.einsum("nt,nt->t", DO, DO, optimize='optimal')
    tmp = np.einsum("m,t->mt", varP, varO, optimize='optimal')
    return cov / np.sqrt(tmp)


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
    chan_sel = 'all' # grad or mag or all
    lock_event = 'fixation' # fixation or saccade
    filt = (1, 30) 
    filt = f"{filt[0]}-{filt[1]}"
    ver = 'normal_vs_retro' # 'normal' etc
    def load_rsa(row):
        n = row['n']
        fname = f"{data_dir}rsa/{ver}/{n}_{chan_sel}_{lock_event}_{filt}.h5"
        res = mne.externals.h5io.read_hdf5(fname)
        return res
    results = everyone.apply_fnc(load_rsa)
    same_coef, diff_coef, times = zip(*results)
    same_coef = np.array(same_coef)
    diff_coef = np.array(diff_coef)

    def rsa_plot(x_same, x_diff):
        # Plot the averages
        plt.subplot(2, 1, 1)
        plt.plot(times[0], x_same, '-r')
        plt.plot(times[0], x_diff, '-k')
        plt.text(-0.8, x_same.max() * 0.9, 'Same', color='r')
        plt.text(-0.8, x_same.max() * 0.8, 'Diff', color='k')
        plt.xlabel('Time (s)')
        plt.ylabel('$R^2$')
        plt.title(lock_event)
        # Plot the difference between same and diff trials
        plt.subplot(2, 1, 2)
        plt.axhline(y=0, linestyle='--', color='k')
        plt.axvline(x=0, linestyle='--', color='k')
        plt.plot(times[0], x_same - x_diff, '-b')
        plt.xlabel('Time (s)')
        plt.ylabel('Same - Diff')
        plt.tight_layout()

    # Plot the averages
    same_mean = np.mean(same_coef, axis=0)
    diff_mean = np.mean(diff_coef, axis=0)
    plt.close('all')
    rsa_plot(same_mean, diff_mean)
    plt.plot(times[0], (same_coef - diff_coef).transpose(),
             '-k', alpha=0.3)
    fname = f"{data_dir}plots/rsa/{ver}/{chan_sel}_{lock_event}_{filt}.png"
    plt.savefig(fname)
    #plt.show()

    # # Plot the individual traces
    # for n in range(same_coef.shape[0]):
    #     plt.close('all')
    #     rsa_plot(same_coef[n,:], diff_coef[n,:])
    #     save_dir =f"{data_dir}plots/rsa/{ver}/indiv/" 
    #     fname = f"{save_dir}/{chan_sel}_{lock_event}_{filt}_{n}.png"
    #     plt.savefig(fname)
    # plt.close("all")


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

        fname = f"{data_dir}plots/rsa/rsa_matrix.png"
        plt.savefig(fname)

        print(f"{np.sum(rsa_mat == 1)} types of 'same' saccades")
        print(f"{np.sum(rsa_mat == -1)} types of 'different' saccades")
        print(f"{np.sum(rsa_mat == 0)} types of ignored saccades")

    return rsa_mat, transition_labels


def rsa_matrix_retro():
    """ Same as above but for 'retrospective' processing
    """
    items = range(6)
    transitions = list(permutations(items, 2))
    # Sort by post-saccade item
    transitions = sorted(transitions, key=lambda x: (x[1], x[0])) 
    transition_labels = [f'{e[0]}-{e[1]}' for e in transitions]
    rsa_mat = np.zeros([len(transitions)] * 2)
    for i_x, t_x in enumerate(transitions):
        for i_y, t_y in enumerate(transitions):
            # All 4 items are different
            if len(set(t_x + t_y)) == 4: 
                rsa_mat[i_y, i_x] = -1
            # Same item before fixation, different item after
            elif (t_x[0] == t_y[0]) and (t_x[1] != t_y[1]):
                rsa_mat[i_y, i_x] = 1 
            else:
                pass # Neither a "same" nor a "different" saccade
    return rsa_mat, transition_labels


def plot_fix_duration_hist():
    for n in [2,3,4,5,6]:
        chan_sel = 'all'
        filt = [1, 30]
        lock_event = 'fixation'
        d = preprocess(n, lock_event, chan_sel, filt)
        fix_dur = d['fix_info']['fix_dur']
        fix_dur = fix_dur[fix_dur < 2000] # Only keep durations < 2 s
        plt.hist(fix_dur, 60)
        plt.xlabel('Time (ms)')
        plt.ylabel('Count')
        plt.savefig(f"../data/plots/fixation-dur-hist/{n}.png")
        plt.close('all')


if __name__ == '__main__':
    n = int(sys.argv[1])
    
    chan_sel = 'all'
    filt = [1, 30]
    lock_event = 'fixation'
    print(n, filt, chan_sel, lock_event)
    d = preprocess(n, lock_event, chan_sel, filt)
    same_coef, diff_coef = corr_analysis(d)

    fname = f"{data_dir}rsa/{n}_{chan_sel}_{lock_event}_{filt[0]}-{filt[1]}.h5"
    mne.externals.h5io.write_hdf5(fname,
                                  [same_coef, diff_coef, d['times']],
                                  overwrite=True)


