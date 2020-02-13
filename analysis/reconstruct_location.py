"""
Reconstruct the quadrant of the next fixation in allocentric screen-coordinates
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import matplotlib.pyplot as plt

import load_data
import reconstruct

expt_info = json.load(open('expt_info.json')) 


def run(n): 
    # Load the data
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

    # Only keep trials that didn't have another eye movement too recently
    prior_saccade_thresh = 250 # In samples (i.e. ms)
    prior_saccade_time = events[1:,0] - events[:-1,0]
    too_close = prior_saccade_time < 250
    too_close = np.hstack([[False], too_close])

    # Get the location of the next fixation
    fix = d['fix_info'].copy()
    x_loc = fix['x_avg'].to_numpy()
    y_loc = fix['y_avg'].to_numpy()
    center_thresh = 50 # Only keep fixations this far from the center
    horiz_side = np.zeros(x_loc.shape)
    horiz_side[x_loc < (x_loc.mean() - center_thresh)] = -1
    horiz_side[x_loc > (x_loc.mean() + center_thresh)] = 1
    vert_side = np.zeros(y_loc.shape)
    vert_side[y_loc < (y_loc.mean() - center_thresh)] = -1
    vert_side[y_loc > (y_loc.mean() + center_thresh)] = +1
    quadrant = np.full(x_loc.shape, np.nan)
    i_quad = 0
    for x_side in [-1, 1]:
        for y_side in [-1, 1]:
            sel = (horiz_side == x_side) & (vert_side == y_side)
            quadrant[sel] = i_quad
            #plt.plot(x_loc[sel], y_loc[sel], 'o')
            i_quad += 1
    fix['quadrant'] = pd.Series(quadrant, index=fix.index)
    d['fix_info'] = fix

    # Apply the selections
    not_in_the_middle = ~np.isnan(fix['quadrant'])
    trial_sel = (~too_close) & not_in_the_middle
    d['fix_info'] = d['fix_info'].loc[trial_sel]
    events = events[trial_sel,:]
    
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

    # Preprocess the data
    reconstruct.preprocess(d, events, preproc_erp_filt, tmin=-1.0, tmax=0.5) 

    # Get the feature to reconstruct
    # In this case, it's the stimulus label
    d['y'] = d['fix_info']['quadrant']
    d['y'] = d['y'].astype(int).to_numpy() 

    # CV regularization doesn't work b/c there's no above-chance classif.
    ## cv_reg_mdl_params = {'Cs': np.logspace(-10, -2, 10),
    ##                      'cv': 5,
    ##                      'n_jobs': 5,
    ##                      'scoring': 'accuracy',
    ##                      'penalty': 'l1',
    ##                      'solver': 'saga',
    ##                      'multi_class': 'ovr'}
    ## mdl = LogisticRegressionCV(**cv_reg_mdl_params)
    ## mdl = reconstruct.cv_reg_param(mdl, d.copy(), 0.1)

    # Reconstruct stimuli at each timepoint
    cv_params = {'cv': 5, 
                 'n_jobs': 5,
                 'scoring': 'accuracy'}
    mdl_params = {'penalty': 'l1', 
                  'solver': 'liblinear',
                  'multi_class': 'ovr',
                  'max_iter': 1e4} 
    mdl = LogisticRegression(C=0.05, **mdl_params)
    results = reconstruct.reconstruct(mdl, d, **cv_params)

    return (results, d['times'])



def aggregate():
    # Get and plot average accuracy
    acc = []
    t = []
    for n in (2,3,4,5,6):
        fname = f"../data/reconstruct/location/{n}.pkl"
        results, times = pickle.load(open(fname, 'rb'))
        accuracy = [r['test_score'].mean() for r in results]
        acc.append(accuracy)
        t.append(times)
    acc = np.array(acc)
    acc_mean = np.mean(acc, 0)
    plt.plot(times, acc.T, '-k', alpha=0.3) # Individual subjects
    # sem = lambda x,axis: np.std(x, axis=axis) / np.sqrt(x.shape[axis]) 
    # acc_sem = sem(acc, 0) * 1.96 # 95% CI
    # plt.fill_between(times, acc_mean + acc_sem, acc_mean - acc_sem,
    #                  facecolor='blue', edgecolor='none', alpha=0.5)
    plt.plot(times, acc_mean, '-b')
    plt.axvline(x=0, color='black', linestyle='-') # Fixation onset
    #plt.axhline(y=1/4, color='black', linestyle='--') # Chance level
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy')
    plt.xlim(times.min(), times.max())
    plt.show()


if __name__ == "__main__":
    try:
        n = sys.argv[1]
    except IndexError:
        n = input('Subject number: ')
    n = int(n)
    results, times = run(n)
    fname = f"{expt_info['data_dir']}reconstruct/location/{n}.pkl"
    pickle.dump([results, times], open(fname, 'wb'))


