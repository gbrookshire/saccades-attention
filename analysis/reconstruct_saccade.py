"""
Reconstruct the direction of the saccade, time-locked to the saccade onset.
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV

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
    #########d['fix_info'] = fix
    
    # Filtering or other preprocessing specific to this analysis
    raw = d['raw']
    raw.load_data()
    # Filter the data
    # These parameters make sure the filter is *causal*, so all effects
    # can't bleed backward in time from post- to pre-saccade time-points
    raw.filter(l_freq=1, h_freq=40, # band-pass filter 
               method='fir', phase='minimum') # causal filter

    # Preprocess the data
    reconstruct.preprocess(d, events, tmin=-1.0, tmax=0.5) 
    # Model parameters
    cv_params = {'cv': 5, 
                 'n_jobs': 5}
    mdl_params = {'selection': 'random', # Speeds up model fitting
                  'max_iter': 1e4}
    # Separately reconstruct x- and y-direction
    mdl = Lasso(alpha=3.57, **mdl_params)
    results = {}
    for dim in ('x', 'y'):
        # Set the variable to reconstruct
        field_name = f"{dim}_change"
        d['y'] = fix[field_name].to_numpy()
        # Reconstruct stimuli at each timepoint
        results[dim] = reconstruct.reconstruct(mdl, d, **cv_params)

    return (results, d['times'])



def plot(n):
    fname = f"{n}.pkl"
    fname = '../data/reconstruct/saccade/' + fname
    results, times = pickle.load(open(fname, 'rb'))
    accuracy = [r['test_score'].mean() for r in results]

    pass


if __name__ == "__main__":
    expt_info = json.load(open('expt_info.json')) 
    try:
        n = sys.argv[1]
    except IndexError:
        n = input('Subject number: ')
    n = int(n)
    results = run(n)
    fname = f"{expt_info['data_dir']}reconstruct/saccade/{n}.pkl"
    pickle.dump(results, open(fname, 'wb'))


