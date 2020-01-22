"""
Reconstruct the direction of the saccade, time-locked to the saccade onset.
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
import matplotlib.pyplot as plt

import load_data
import reconstruct

expt_info = json.load(open('expt_info.json')) 


def run(n): 
    # Load the data
    d = load_data.load_data(n)
    # Select fixation offsets -- i.e. saccade onsets
    row_sel = d['fix_events'][:,2] == expt_info['event_dict']['fix_off']
    events = d['fix_events'][row_sel, :] 
    # Get the feature to reconstruct: the direction of the next saccade
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
    events = events[fix['saccade']] 
    fix = fix.loc[fix['saccade']]
    d['fix_info'] = fix
    
    # Filtering or other preprocessing specific to this analysis
    def preproc_erp_filt(raw):
        """ Filter the data as is standard for an ERP analysis
        """
        # Filter the data
        # These parameters make sure the filter is *causal*, so all effects
        # can't bleed backward in time from post- to pre-saccade time-points
        raw.filter(l_freq=1, h_freq=40, # band-pass filter 
                method='fir', phase='minimum', # causal filter
                n_jobs=5)
        return raw
    
    def preproc_alpha_pow(raw):
        """ Get alpha power using the Hilbert transform
        """
        raw.filter(l_freq=8, h_freq=12, # band-pass filter 
                method='fir', phase='minimum', # causal filter
                n_jobs=5)
        hilb_picks = mne.pick_types(raw.info, meg=True)
        raw.apply_hilbert(hilb_picks)
        raw.apply_function(lambda x: np.real(np.abs(x)), hilb_picks)
        return raw

    # Preprocess the data
    reconstruct.preprocess(d, events,
                           #preproc_erp_filt,
                           #preproc_alpha_pow,
                           tmin=-1.0, tmax=0.5)

    # Get alpha power


    # Model parameters

    ## LASSO regressions
    #cv_params = {'cv': 5, 
    #             'n_jobs': 5}
    #mdl_params = {'selection': 'random', # Speeds up model fitting
    #              'max_iter': 1e4}
    ## CV to get model parameters
    #mdl = LassoCV(alphas=np.logspace(0.1, 1, 20), **mdl_params)
    #mdl = reconstruct.cv_reg_param(mdl, d.copy(), -0.1)
    ## The final LASSO model
    #mdl = Lasso(alpha=3.57, **mdl_params)
    ## Separately reconstruct x- and y-direction of movement
    #results = {}
    #for dim in ('x', 'y'):
    #    # Set the variable to reconstruct
    #    d['y'] = d['fix_info'][f"{dim}_change"].copy().to_numpy()
    #    # Reconstruct stimuli at each timepoint
    #    results[dim] = reconstruct.reconstruct(mdl, d.copy(), **cv_params)

    ## Ridge regressions
    #cv_params = {'cv': None, # Efficient leave-one-out CV, aka Generalized CV
    #             'store_cv_values': True,
    #             'n_jobs': 5}
    #mdl_params = {}
    ## CV to find an appropriate value for alpha (regularization param)
    #mdl = RidgeCV(alphas=np.logspace(-1, 2, 20), **mdl_params)
    #d['y'] = d['fix_info']['x_change'].copy().to_numpy()
    #mdl = reconstruct.cv_reg_param(mdl, d.copy(), -0.1)
    #
    ## Separately reconstruct x- and y-direction of movement
    #mdl = Ridge(alpha=10.0, **mdl_params)
    #results = {}
    #for dim in ('x', 'y'):
    #    # Set the variable to reconstruct
    #    d['y'] = d['fix_info'][f"{dim}_change"].copy().to_numpy()
    #    # Reconstruct stimuli at each timepoint
    #    results[dim] = reconstruct.reconstruct(mdl, d.copy(), **cv_params)

    # Elastic-net regressions
    cv_params = {'cv': 5,
                 'n_jobs': 5}
    mdl_params = {'selection': 'random' # Speed up fitting
                 }
    ## CV to get elastic-net model parameters
    #mdl = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
    #                   n_alphas=10,
    #                   max_iter=1e4,
    #                   **mdl_params, **cv_params)
    #d['y'] = d['fix_info']['x_change'].copy().to_numpy()
    #mdl = reconstruct.cv_reg_param(mdl, d.copy(), -0.1)
    # With n=2 and t=-0.1, this gives: l1_ratio=0.9, alpha=5.56
    # Reconstruct eye movements, separately for x- and y-direction
    mdl = ElasticNet(alpha=5.56, l1_ratio=0.9, **mdl_params)
    results = {}
    for dim in ('x', 'y'):
        # Set the variable to reconstruct
        d['y'] = d['fix_info'][f"{dim}_change"].copy().to_numpy()
        # Reconstruct stimuli at each timepoint
        results[dim] = reconstruct.reconstruct(mdl, d.copy(), **cv_params)

    return (results, d['times'])


def plot(n):
    fname = f'../data/reconstruct/saccade/{n}.pkl'
    results, times = pickle.load(open(fname, 'rb'))
    for dim in ('x', 'y'):
        accuracy = [r['test_score'].mean() for r in results[dim]]
        plt.plot(times, accuracy)
    plt.xlabel('Time (s)')
    plt.ylabel('$R^2$')


def aggregate():
    subjects = (2, 3, 4, 5)
    for i_plot,n in enumerate(subjects):
        plt.subplot(2, 2, i_plot + 1)
        plot(n)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        n = sys.argv[1]
    except IndexError:
        n = input('Subject number: ')
    n = int(n)
    results, times = run(n)
    fname = f"{expt_info['data_dir']}reconstruct/saccade/{n}.pkl"
    pickle.dump([results, times], open(fname, 'wb'))




