"""
Functions to simplify reconstructing stimuli & behavior from MEG data.
"""

import json
import numpy as np
from tqdm import tqdm
import mne
import load_data
import artifacts
from sklearn.model_selection import cross_validate 
from sklearn.preprocessing import StandardScaler

expt_info = json.load(open('expt_info.json'))
scaler = StandardScaler()


def preprocess(d, events, **epochs_kwargs):
    """ Add a field to `d`: a (Trial x Channel x Time) array of MEG data.
    Modifies `d['fix_info']` in-place to remove rejected trials.
    """
    # Epoch the data
    picks = mne.pick_types(d['raw'].info,
                        meg=True, eeg=False, eog=False,
                        stim=False, exclude='bads')
    reject = dict(grad=4000e-13, # T / m (gradiometers)
                mag=4e-12, # T (magnetometers)
                #eeg=40e-6, # V (EEG channels)
                #eog=250e-6 # V (EOG channels)
                ) 
    epochs = mne.Epochs(d['raw'],
                        events,
                        #reject=reject,
                        reject_by_annotation=True,
                        preload=True,
                        baseline=None,
                        picks=picks,
                        **epochs_kwargs) 
    # Reject ICA artifacts
    d['ica'].apply(epochs) 
    # Resample after epoching to make sure trigger times are correct
    epochs.resample(200, n_jobs=3) 
    # Reject trials that wre manually marked as bad
    meg_data = epochs.get_data()
    d['fix_info'] = d['fix_info'].iloc[epochs.selection] 
    # Reject trials with high GFP
    bad_trials = artifacts.identify_gfp(meg_data, sd=4)
    meg_data = meg_data[~bad_trials,:,:]
    d['fix_info'] = d['fix_info'].iloc[~bad_trials] 
    # Add important fields to the data
    d['meg_data'] = meg_data
    d['times'] = epochs.times


def cv_reg_param(mdl, d, t_cv):
    """
    Use cross-validation to find the regularization parameter (also called C,
    lambda, or alpha) for LASSO regressions.

    Don't run this separately for every subject/timepoint. Only run this in a
    few subjects, to get a sense of reasonable values.

    mdl: Instance of a sklearn CV model, e.g. LassoCV or LogisticRegressionCV
    t_cv: Time-point at which we're cross-validating (in s)
    """
    i_time = np.nonzero(d['times'] >= t_cv)[0][0] # Find index of the timepoint
    x = d['meg_data'][:,:,i_time]
    x = scaler.fit_transform(x) 
    mdl.fit(x, d['y'])
    print('Score: ', mdl.score(x, d['y']))
    print('Avg number of nonzero coefs: ',
          np.mean(np.sum(mdl.coef_ != 0, axis=1)))
    #print('Regularization parameters: ', mdl.C_)

    return mdl


def reconstruct(mdl, d, **cv_params):
    """ Reconstruct the feature in `d['y']` based on each timepoint of
    `d['meg_data']`. The model is constructed as a sklearn model, e.g. Lasso or
    LogisticRegression.
    """ 
    # Run the classifier for each time-point
    results = []
    for i_time in tqdm(range(d['times'].size)):
        # Select data at this time-point
        x = d['meg_data'][:,:,i_time] 
        # Standardize the data within each MEG channel
        x = scaler.fit_transform(x) 
        # Cross-validated classifiers
        res = cross_validate(mdl, x, d['y'],
                             return_estimator=True,
                             **cv_params)
        # Store the results
        results.append(res)

    return results


if __name__ == '__main__':
    # This gives slightly different results from the prior version
    # It's very similar, but the actual numbers are a little different.
    # Why is that?
    #
    # This example runs classifies which image the subject is looking at
    n = int(input('Subject number: ')) 
    d = load_data.load_data(n)
    # Select fixation onsets
    row_sel = d['fix_events'][:,2] == expt_info['event_dict']['fix_on'] 
    events = d['fix_events'][row_sel, :] 
    # Select fixations to a new object
    new_obj = np.diff(d['fix_info']['closest_stim']) != 0
    new_obj = np.hstack((True, new_obj)) # First fixation is to a new object
    d['fix_info'] = d['fix_info'].loc[new_obj]
    events = events[new_obj,:]
    # Get preprocessed data
    preprocess(d, events, tmin=-1.0, tmax=0.5) 
    # Get the stimuli to reconstruct
    d['y'] = d['fix_info']['closest_stim']
    d['y'] = d['y'].astype(int).to_numpy() 
    # Reconstruct stimuli at each timepoint
    from sklearn.linear_model import LogisticRegression
    cv_params = {'cv': 5, 
                 'n_jobs': 5,
                 'scoring': 'accuracy'}
    mdl_params = {'penalty': 'l1', 
                  'solver': 'liblinear',
                  'multi_class': 'ovr',
                  'max_iter': 1e4} 
    mdl = LogisticRegression(C=0.05, **mdl_params)
    results = reconstruct(mdl, d, **cv_params)
    accuracy = [r['test_score'].mean() for r in results] 
    # Plot the results
    import matplotlib.pyplot as plt
    plt.plot(d['times'], accuracy)
    plt.plot([d['times'].min(), d['times'].max()], [1/6, 1/6], '--k')
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy')

