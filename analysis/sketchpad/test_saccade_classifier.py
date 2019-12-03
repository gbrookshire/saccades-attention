"""
Classify the direction of the upcoming saccade

This is actually a Forward Encoding Model, because we're predicting
saccade location based on brain activity. (Is that right?)

"""

# TODO
# Cross-correlation in saccade direction?
# - This will make sure that successful classification isn't due to 
#   post-saccade effects + multiple saccades in the same direction.
# When did the previous saccade happen?
# Make saccade histogram into P(saccade) for each time point

import os 
import json
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate

import eyelink_parser 
import stim_positions
import fixation_events
import load_data
import artifacts

expt_info = json.load(open('expt_info.json'))
scaler = StandardScaler()

# Model parameters
# Main classifier
clf_params = {'selection': 'random', # Speeds up model fitting
              'max_iter': 1e4}

# Cross-validataion
cv_params= {'cv': 5, 
            'n_jobs': 3}

# CV of regularization parameters
cv_reg_params = {'selection': 'random', 
                 'max_iter': 1e5}

def preprocess(n):
    """ Preprocess the MEG dat for classification analyses
    """ 
    # Load the data
    d = load_data.load_data(n)
    
    # Select fixation offsets -- i.e. saccade onsets
    row_sel = d['fix_events'][:,2] == expt_info['event_dict']['fix_off']
    d['fix_events'] = d['fix_events'][row_sel, :]
    
    # Epoch the data
    tmin = -1.0
    tmax = 0.5
    picks = mne.pick_types(d['raw'].info,
                           meg=True, eeg=False, eog=False,
                           stim=False, exclude='bads')
    reject = dict(grad=4000e-13, # T / m (gradiometers)
                  mag=4e-12, # T (magnetometers)
                  #eeg=40e-6, # V (EEG channels)
                  #eog=250e-6 # V (EOG channels)
                  ) 
    epochs = mne.Epochs(d['raw'], d['fix_events'],
                        tmin=tmin, tmax=tmax, 
                        #reject=reject,
                        reject_by_annotation=True,
                        preload=True,
                        baseline=None,
                        picks=picks)
    
    # Reject ICA artifacts
    d['ica'].apply(epochs)
    
    # Resample after epoching to make sure trigger times are correct
    epochs.resample(200, n_jobs=3)
    
    # Get the direction of the next saccade
    # Check the timing between fixations to make sure
    # that the next item on the list is actually the next fixation.
    fix = d['fix_info'].copy()
    onsets = fix['start'][1:].to_numpy()
    offsets = fix['end'][:-1].to_numpy()
    saccade_dur = onsets - offsets
    saccade_dur = saccade_dur / 1000
    saccade_dur = np.hstack((saccade_dur, np.inf))
    plausible_saccade = saccade_dur < 0.15
    x_change = fix['x_avg'][1:].to_numpy() - fix['x_avg'][:-1].to_numpy()
    x_change = np.hstack((x_change, np.nan))
    y_change = fix['y_avg'][1:].to_numpy() - fix['y_avg'][:-1].to_numpy()
    y_change = np.hstack((y_change, np.nan))
    fix['x_change'] = pd.Series(x_change, index=fix.index)
    fix['saccade_dur'] = pd.Series(saccade_dur, index=fix.index)
    fix['saccade'] = pd.Series(plausible_saccade, index=fix.index)
    fix['y_change'] = pd.Series(y_change, index=fix.index)
    
    # Get data structures for running regressions
    meg_data = epochs.get_data() # Trial x Channel x Time 
    fix = fix.iloc[epochs.selection] # Toss manually-marked bad trials
    
    # Only keep trials that are a real saccade
    meg_data = meg_data[fix['saccade'] == True, :]
    fix = fix.loc[fix['saccade'] == True]
    
    # Toss trials with high GFP
    bad_trials = artifacts.identify_gfp(meg_data, sd=4) 
    meg_data = meg_data[~bad_trials,:,:]
    fix = fix.loc[~bad_trials]
    
    ## Separately predict movement in the x and y directions 
    ## Or look at sin and cos of the angle of movement?
    ## - Plus the distance of the saccade
    #x_change = fix['x_change'].to_numpy()
    #y_change = fix['y_change'].to_numpy()

    return meg_data, fix, epochs.times
    

def cv_reg_param(meg_data, ground_truth, times, t_cv=-0.05, plot=True):
    """
    Use cross-validation to find the regularization parameter (also called C,
    lambda, or alpha) for LASSO regressions.

    Don't run this separately for every subject/timepoint. Only run this in a
    few subjects, to get a sense of reasonable values.

    t_cv: Time-point at which we're cross-validating (in s)
    """
    i_time = np.nonzero(times >= t_cv)[0][0] # Find index of the timepoint
    x = meg_data[:,:,i_time]
    x = scaler.fit_transform(x)
    clf = LassoCV(**cv_reg_params, **cv_params)
    clf.fit(x, ground_truth)
    print('R^2: ', clf.score(x, ground_truth))
    print('Regularization parameters: ', clf.alpha_)
    print('Avg number of nonzero coefs: ',
          np.mean(np.sum(clf.coef_ != 0, axis=1)))
    if plot:
        plt.plot(ground_truth, clf.predict(x), 'o', alpha=0.5) 
    return clf


def reconstruct_saccade(meg_data, ground_truth, times, alpha=3.57):
    """ Reconstruct saccade direction from the MEG data.
    """ 
    # TODO make this work for y_change in addition to x_change
    # TODO refactor to eliminate duplicated code
    # Set up the main classifier
    clf = Lasso(alpha=alpha, **clf_params)

    # Run the classifier for each time-point
    results = []
    for i_time in tqdm(range(times.size)):
        # Select data at this time-point
        x = meg_data[:,:,i_time] 
        # Standardize the data within each MEG channel
        x = scaler.fit_transform(x) 
        # Cross-validated classifiers
        res = cross_validate(clf, x, x_change,
                             return_estimator=True,
                             **cv_params)
        # Store the results
        results.append(res)

    return results


def plot_results(times, results): 
    accuracy = [r['test_score'].mean() for r in results] 
    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot classifier accuracy over time
    a0.plot(times, accuracy)
    a0.set_ylabel('$R^2$')
    
    # Plot saccade durations
    bins = np.arange(times[0], times[-1], step=0.01) 
    
    # Plot proportion of trials that fall during a saccade
    hist, bin_edges = np.histogram(fix['saccade_dur'],
                                   bins=bins)
    hist = np.cumsum(hist)
    hist = (hist.max() - hist) / hist.max()
    hist[bins[:-1] < 0] = 0
    plt.step(bins[:-1], hist, where='post')
    a1.set_xlabel('Time (s)')
    a1.set_ylabel('P(saccade)')
    
    ## Plot histogram of saccade durations
    #a1.hist(fix['saccade_dur'], bins=bins)
    #a1.set_ylabel('Count')
    
    plt.tight_layout()
    plt.show()


