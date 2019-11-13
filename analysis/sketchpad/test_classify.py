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

import eyelink_parser 
import stim_positions
import fixation_events

expt_info = json.load(open('expt_info.json')) 

fnames = {'meg': '191104/yalitest.fif',
            'eye': '19110415.asc',
            'behav': '2019-11-04-1527.csv'}

# Load the MEG data
fname = expt_info['data_dir'] + 'raw/' + fnames['meg'] 
raw = mne.io.read_raw_fif(fname)
events = mne.find_events(raw, # Segment out the MEG events
                            stim_channel='STI101',
                            mask=0b00111111, # Ignore Nata button triggers
                            shortest_event=1)

# Read in the EyeTracker data
fname = expt_info['data_dir'] + 'eyelink/ascii/' + fnames['eye']
eye_data = eyelink_parser.EyelinkData(fname)

# Load behavioral data
fname = expt_info['data_dir'] + 'logfiles/' + fnames['behav']
behav = pd.read_csv(fname) 

# Get the fixation events
fix_info, events = fixation_events.get_fixation_events(events, eye_data, behav)
fix_events = events[events[:,2] == expt_info['event_dict']['fix_on'], :]

# Epoch the data
tmin = -0.2
tmax = 0.2
picks = mne.pick_types(raw.info,
                       meg=True, eeg=False, eog=False,
                       stim=False, exclude='bads')
reject = dict(grad=4000e-13, # T / m (gradiometers)
              mag=4e-12, # T (magnetometers)
              #eeg=40e-6, # V (EEG channels)
              #eog=250e-6 # V (EOG channels)
              ) 
epochs = mne.Epochs(raw, fix_events,
                    tmin=tmin, tmax=tmax, 
                    #reject=reject,
                    picks=picks)

# Resample after epoching to make sure trigger times are correct
epochs.load_data()
epochs.resample(200)#, n_jobs=3)

# Plot activity evoked by an eye movement
evoked = epochs.average()
evoked.plot()
times = np.arange(-0.05, 0.2, 0.05)
evoked.plot_topomap(times=times, ch_type='grad')
evoked.plot_topomap(times=times, ch_type='mag')

# Classifiers
d = epochs.get_data() # Trial x Channel x Time
labels = fix_info['closest_stim'].astype(int) # Stimulus to decode
labels = labels.to_numpy()

from sklearn.linear_model import Lasso, LinearRegression
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate, cross_val_score                                            

scaler = StandardScaler()

# Toss weird trials (Should have been done above)
gfp = np.std(d, axis=1) # Global field power
max_gfp = np.max(gfp, axis=1) # Max per trial
bad_trials = max_gfp > (np.std(max_gfp) * 6)
bad_trials = np.nonzero(bad_trials)[0]
d = np.delete(d, bad_trials, axis=0)
labels = np.delete(labels, bad_trials, axis=0)

# Details of the classifier calls
clf_params = {'penalty': 'l1', # Main classifier
              'solver': 'liblinear',
              'multi_class': 'ovr',
              'max_iter': 1e4}
cv_params= {'cv': 5, # Cross-validataion
            'n_jobs': 3,
            'scoring': 'accuracy'}
cv_reg_params = {'penalty': 'l1', # CV of regularization param
                 'solver': 'saga',
                 'multi_class': 'multinomial',
                 'max_iter': 1e4}

# Set up the classifier
clf = LogisticRegression(C=0.112, **clf_params)

results = []
accuracy = []
for i_time in tqdm(range(epochs.times.size)):
    # Select data at this time-point
    x = d[:,:,i_time] 
    # Standardize the data within each MEG channel
    x = scaler.fit_transform(x) 
    # Cross-validated classifiers
    res = cross_validate(clf, x, labels,
                         return_estimator=True,
                         **cv_params)
    # Store the results
    results.append(res)
    accuracy.append(res['test_score'].mean())

# Plot the results
plt.plot(epochs.times, accuracy)
plt.plot([epochs.times.min(), epochs.times.max()], # Mark chance level
         np.array([1, 1]) * (1 / len(np.unique(labels))),
         '--k')
plt.ylabel('Accuracy')
plt.xlabel('Time (s)')

# Try out cross-validation to find the best values of C/lambda/alpha
y = np.array(labels == 0) # Try a classifier for one label
i_time = 60
x = d[:,:,i_time]
x = scaler.fit_transform(x)
clf = LogisticRegressionCV(Cs=np.linspace(0.001, 1, 20),
                           **cv_reg_params,
                           **cv_params)
clf.fit(x, labels)
print(clf.C_)
print(clf.scores_)
print(clf.coef_)

