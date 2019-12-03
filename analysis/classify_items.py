"""
Try out MEG classifier analyses
"""

import os 
import json
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate

import eyelink_parser 
import stim_positions
import fixation_events
import load_data

expt_info = json.load(open('expt_info.json'))
scaler = StandardScaler()

# Classifier parameters
# Main classifier
clf_params = {'penalty': 'l1', 
              'solver': 'liblinear',
              'multi_class': 'ovr',
              'max_iter': 1e4}

# Cross-validataion
cv_params= {'cv': 5, 
            'n_jobs': 5,
            'scoring': 'accuracy'}

# CV of regularization parameter
cv_reg_params = {'penalty': 'l1', 
                 'solver': 'saga',
                 'multi_class': 'multinomial',
                 'max_iter': 1e4}

def preprocess(n):
    """ Preprocess the MEG data for classifying the stimuli
    """ 
    # Load the data
    d = load_data.load_data(n)
    
    # Select fixation onsets
    row_sel = d['fix_events'][:,2] == expt_info['event_dict']['fix_on']
    d['fix_events'] = d['fix_events'][row_sel, :]
    
    # Select fixations to a new target
    new_obj = np.diff(d['fix_info']['closest_stim']) != 0
    new_obj = np.hstack((True, new_obj)) # First fixation is to a new object
    d['fix_info'] = d['fix_info'].loc[new_obj]
    d['fix_events'] = d['fix_events'][new_obj,:]
    
    # Epoch the data
    tmin = -0.4
    tmax = 0.4
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
    epochs.resample(200)
    
    # Prep data structures for running classifiers
    meg_data = epochs.get_data() # Trial x Channel x Time
    labels = d['fix_info']['closest_stim'] # Stimulus to decode
    labels = labels.astype(int).to_numpy()
    labels = labels[epochs.selection] # Only keep retained trials
    
    # Toss weird trials (Should have been done above)
    gfp = np.std(meg_data, axis=1) # Global field power
    max_gfp = np.max(gfp, axis=1) # Max per trial
    zscore = lambda x: (x - x.mean()) / (x.std()) # z-score a vector
    bad_trials = zscore(max_gfp) > 4
    meg_data = meg_data[~bad_trials,:,:]
    labels = labels[~bad_trials]

    return meg_data, labels, epochs.times


def cv_reg_param(meg_data, labels, times, t_cv=0.1):
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
    Cs_to_test = np.linspace(0.001, 1, 20)
    clf = LogisticRegressionCV(Cs=Cs_to_test,
                               **cv_reg_params,
                               **cv_params)
    clf.fit(x, labels)
    print('Accuracy: ', clf.score(x, labels))
    print('Regularization parameters: ', clf.C_)
    print('Avg number of nonzero coefs: ',
          np.mean(np.sum(clf.coef_ != 0, axis=1)))
    return clf


def decode_stimulus(meg_data, labels, times, C=0.05):
    """ Decode the stimulus from the MEG data.
    """ 
    # Set up the classifier
    clf = LogisticRegression(C=C, **clf_params)
    
    # Run the classifier for each time-point
    results = []
    accuracy = []
    for i_time in tqdm(range(times.size)):
        # Select data at this time-point
        x = meg_data[:,:,i_time] 
        # Standardize the data within each MEG channel
        x = scaler.fit_transform(x) 
        # Cross-validated classifiers
        res = cross_validate(clf, x, labels,
                             return_estimator=True,
                             **cv_params)
        # Store the results
        results.append(res)

    return results


def plot_results(labels, times, results):
    accuracy = [r['test_score'].mean() for r in results] 
    plt.plot(times, accuracy)
    plt.plot([times.min(), times.max()], # Mark chance level
            np.array([1, 1]) * (1 / len(np.unique(labels))),
            '--k')
    plt.ylabel('Accuracy')
    plt.xlabel('Time (s)')
    plt.show()


def _timing_test_cv(n_jobs=3):
    """
    Test the timing of multiple jobs in sklearn.cross_validate
    With downsampling to 200 Hz on Yali's brief dataset
    n_jobs
    1: 355 ms/loop
    5: 178 ms/loop  ******
    10: 226 ms/loop

    This uses the IPython magic function %timeit, so it won't work
    in a standard python interpreter.
    """ 
    cv_params= {'cv': 5,
                'n_jobs': n_jobs,
                'scoring': 'accuracy'}
    i_time = 100
    x = meg_data[:,:,i_time]
    x = scaler.fit_transform(x)
    clf = LogisticRegression(C=0.05, **clf_params)
    #%timeit cross_validate(clf, x, labels, return_estimator=True, **cv_params)


if __name__ == '__main__':
    n = int(input('Subject number: '))
    meg_data, labels, times = preprocess(n)
    results = decode_stimulus(meg_data, labels, times, C=0.05)
    plot_results(labels, times, results)
