"""
Reconstruct which image the participant was looking at, time-locked to the
fixation onset. For positive time-points, this reconstructs perception at the
fovea for newly-fixated images. For negative time-points, this is either
prediction or sensory pre-processing.
"""

import sys
import json
import socket
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from mne.externals.h5io import write_hdf5, read_hdf5

import load_data
import reconstruct

expt_info = json.load(open('expt_info.json')) 

if socket.gethostname() == 'colles-d164179':
    data_dir = expt_info['data_dir']['external']
else:
    data_dir = expt_info['data_dir']['standard']


def run(n): 
    d = preproc(n)
    results = fit(d)
    return (results, d['times'])


def preproc(n):
    # Load the data
    d = load_data.load_data(n)

    locking_event = 'fixation' # fixation or saccade
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

    def preproc_erp_filt(raw):
        """ Filter the data as is standard for an ERP analysis
        """
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
    d['y'] = d['fix_info']['closest_stim']
    d['y'] = d['y'].astype(int).to_numpy() 

    # Model "retrospective" coding
    d['y'] = np.hstack([0, d['y'][:-1]])

    return d


def fit(d):
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
    return results


def model_adjacent_items(d):
    """
    Look at how brain activity corresponds to items at different lags.
        item n-1
        item n
        item n+1
    """
    y_real = d['y'].copy()
    
    # Normal analysis - does brain activity correspond to this fixation
    d['y'] = y_real 
    results = fit(d)
    accuracy = [r['test_score'].mean() for r in results] 
    plt.plot(d['times'], accuracy)

    # Predictive analysis - does brain activity correspond to item n+1
    d['y'] = np.hstack([y_real[1:], 0])
    results = fit(d)
    accuracy = [r['test_score'].mean() for r in results] 
    plt.plot(d['times'], accuracy)

    # Retrospective analysis - does brain activity correspond to item n-1
    d['y'] = np.hstack([0, y_real[:-1]])
    results = fit(d)
    accuracy = [r['test_score'].mean() for r in results] 
    plt.plot(d['times'], accuracy)

    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy')


def plot(results, times):
    import mne
    # Get the accuracy
    accuracy = [r['test_score'].mean() for r in results] 
    # Plot the results
    plt.figure()
    plt.plot(times, accuracy)
    plt.plot([times.min(), times.max()], [1/6, 1/6], '--k')
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy')
    # Get the channels used in the LASSO regression
    # This is the number of non-zero parameters 
    meg_chan_inx = mne.pick_types(d['raw'].info, meg=True)
    n_nonzero = np.zeros([0, meg_chan_inx.size])
    for r_t in results:
        coefs = [m.coef_ for m in r_t['estimator']]
        coefs = np.array(coefs)
        avg_param_count = np.mean(coefs != 0, axis=(0, 1))
        n_nonzero = np.vstack((n_nonzero, avg_param_count))
    # Plot the number of nonzero channels over time
    plt.figure()
    plt.plot(times, np.sum(n_nonzero, axis=1))
    plt.xlabel('Time (s)')
    plt.ylabel('Number of nonzero parameters')

    # Plot a topography of how often each channel appears
    x = np.reshape(np.mean(n_nonzero.transpose(), axis=1), [-1, 1])
    nonzero_params = mne.EvokedArray(x,
                                     mne.pick_info(d['raw'].info,
                                                   meg_chan_inx))

    # # This doesn't really work
    # lay = mne.channels.find_layout(nonzero_params.info)
    # x = lay.pos[:,0]
    # y = lay.pos[:,1]
    # x -= x.mean()
    # y -= y.mean()
    # plt.plot(x, y, '*r') 

    # # Not sure how to get the sensor positions on this plot
    # nonzero_params.plot_sensors()
    # pos = mne.channels.layout._auto_topomap_coords(
    #                     nonzero_params.info,
    #                     picks='mag',
    #                     ignore_overlap=True,
    #                     to_sphere=True) 
    # pos = pos.transpose()
    # x = pos[0,:]
    # y = pos[1,:]
    # x -= x.mean()
    # y -= y.mean()
    # x *= 0.425 / x.max()
    # y *= 0.425 / y.max()
    # plt.plot(x, y, '*r')


def aggregate():
    # Get and plot average accuracy
    acc = []
    t = []
    for n in (2,3,4,5,6):
        analysis_type = 'fix_onset' # saccade_onset, fix_onset
        fname = f"../data/reconstruct/item/{analysis_type}/{n}.pkl"
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
    plt.axhline(y=1/6, color='black', linestyle='--') # Chance level
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
    fname = f"{data_dir}reconstruct/item/{n}.pkl"
    write_hdf5(fname, [results, times], overwrite=True)


