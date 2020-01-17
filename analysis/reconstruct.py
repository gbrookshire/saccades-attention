"""
Functions to simplify reconstructing stimuli & behavior from MEG data.
For an example of how to use this, see reconstruct_item.py
"""

# TODO
"""
- Carefully go through pipeline
- Convert to a class def?
    - That way all these side effects make more sense...
"""

import json
import numpy as np
from tqdm import tqdm
import mne
import artifacts
from sklearn.model_selection import cross_validate 
from sklearn.preprocessing import StandardScaler

expt_info = json.load(open('expt_info.json'))
scaler = StandardScaler()


def preprocess(d, events, **epochs_kwargs):
    """ Add a field to `d`: a (Trial x Channel x Time) array of MEG data.
    Modifies `d['fix_info']` *in-place* to remove rejected trials.
    """
    # Epoch the data
    picks = mne.pick_types(d['raw'].info,
                        meg=True, eeg=False, eog=False,
                        stim=False, exclude='bads')
    epochs = mne.Epochs(d['raw'],
                        events,
                        reject_by_annotation=True,
                        preload=True,
                        baseline=None,
                        picks=picks,
                        proj=True,
                        **epochs_kwargs) 
    # Reject ICA artifacts
    d['ica'].apply(epochs) 
    # # Resample after epoching to make sure trigger times are correct
    # epochs.resample(100, n_jobs=3) 
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


def cv_reg_param(mdl, d, t_cv):
    """
    Use cross-validation to find the regularization parameter (also called C,
    lambda, or alpha) for LASSO regressions.

    Don't run this separately for every subject/timepoint. Only run this in a
    few subjects, to get a sense of reasonable values.

    mdl: Instance of a sklearn CV model, e.g. LassoCV or LogisticRegressionCV
    d: output of
    t_cv: Time-point at which we're cross-validating (in s)
    """
    i_time = np.nonzero(d['times'] >= t_cv)[0][0] # Find index of the timepoint
    x = d['meg_data'][:,:,i_time]
    x = scaler.fit_transform(x) 
    mdl.fit(x, d['y'])
    print('Score: ', mdl.score(x, d['y']))
    print('Avg number of nonzero coefs: ',
          np.mean(np.sum(mdl.coef_ != 0, axis=1)))
    for param_name in ('C_', 'alpha_'):
        try:
            reg_param = getattr(mdl, param_name)
        except AttributeError:
            pass
        print('Regularization parameters: ', reg_param) 

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

