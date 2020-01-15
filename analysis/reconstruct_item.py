"""
Reconstruct which image the participant was looking at, time-locked to the
fixation onset. For positive time-points, this reconstructs perception at the
fovea for newly-fixated images. For negative time-points, this is either
prediction or sensory pre-processing.
"""

import sys
import json
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

import load_data
import reconstruct

expt_info = json.load(open('expt_info.json')) 

def run(n): 
    # Load the data
    d = load_data.load_data(n)
    # Select fixation onsets
    row_sel = d['fix_events'][:,2] == expt_info['event_dict']['fix_on'] 
    events = d['fix_events'][row_sel, :] 
    # Select fixations to a new object
    new_obj = np.diff(d['fix_info']['closest_stim']) != 0
    new_obj = np.hstack((True, new_obj)) # First fixation is to a new object
    d['fix_info'] = d['fix_info'].loc[new_obj]
    events = events[new_obj,:]
    # Preprocesse the data
    reconstruct.preprocess(d, events, tmin=-1.0, tmax=0.5) 
    # Get the feature to reconstruct
    # In this case, it's the stimulus label
    d['y'] = d['fix_info']['closest_stim']
    d['y'] = d['y'].astype(int).to_numpy() 
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


def plot(results, times):
    import matplotlib.pyplot as plt
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
    plt.plot(d['times'], np.sum(n_nonzero, axis=1))
    plt.xlabel('Time (s)')
    plt.ylabel('Number of nonzero parameters')

    # # Plot a topography of how often each channel appears
    # x = np.reshape(np.mean(n_nonzero.transpose(), axis=1), [-1, 1])
    # nonzero_params = mne.EvokedArray(x,
    #                                  mne.pick_info(d['raw'].info,
    #                                                meg_chan_inx))

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
    for n in (2,3,4,5):
        fname = f"{n}.pkl"
        fname = '../data/reconstruct/item/' + fname
        results = pickle.load(open(fname, 'rb'))
        accuracy = [r['test_score'].mean() for r in results]
        acc.append(accuracy)
    acc = np.array(acc)
    t = np.arange(acc.shape[1]) # FIXME
    sem = lambda x,axis: np.std(x, axis=axis) / np.sqrt(x.shape[axis]) 
    acc_sem = sem(acc, 0)
    acc_mean = np.mean(acc, 0)
    plt.plot(t, acc.T, '-k', alpha=0.3)
    #plt.fill_between(t, acc_mean + acc_sem, acc_mean - acc_sem,
    #                 facecolor='blue', edgecolor='none', alpha=0.5)
    plt.plot(t, acc_mean, '-b')
    plt.plot([t.min(), t.max()], [1/6, 1/6], '--k')
    plt.show()


if __name__ == "__main__":
    try:
        n = sys.argv[1]
    except IndexError:
        n = input('Subject number: ')
    n = int(n)
    results, times = run(n)
    fname = f"{expt_info['data_dir']}reconstruct/item/{n}.pkl"
    pickle.dump([results, times], open(fname, 'wb'))

