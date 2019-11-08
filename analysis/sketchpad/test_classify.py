"""
Try out MEG classifier analyses
"""

import os 
import json
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

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
              eeg=40e-6, # V (EEG channels)
              eog=250e-6 # V (EOG channels)
              ) 
epochs = mne.Epochs(raw, fix_events,
                    tmin=tmin, tmax=tmax, 
                    picks=picks)

# Resample after epoching to make sure trigger times are correct
epochs.load_data()
epochs.resample(200)

# Plot activity evoked by an eye movement
evoked = epochs.average()
evoked.plot()
times = np.arange(-0.05, 0.2, 0.05)
evoked.plot_topomap(times=times, ch_type='grad')
evoked.plot_topomap(times=times, ch_type='mag')

# Classifiers
d = epochs.get_data() # Trial x Channel x Time
labels = fix_info['closest_stim'].astype(int) # Stimulus to decode

from sklearn.linear_model import Lasso, LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

acc = []
for i_time in range(epochs.times.size):
    x = d[:,:,i_time]
    y = (labels == stim).astype(int)

    # Standardize the data within each MEG channel
    x = scaler.fit_transform(x)

    #linreg = LinearRegression(normalize=True) # Check about normalize
    #linreg.fit(x, y)
    #y_pred = linreg.predict(x)
    #mse = np.mean((y - y_pred) ** 2)

    logreg = LogisticRegression(penalty='l1', solver='liblinear', 
                                # How does this set the alpha (lambda?) parameter?
                                # Fit a separate binary classifier for each label
                                multi_class='ovr' 
                                )
    #logreg = LogisticRegression(penalty='none', solver='saga')
    # It's definitely overfitting to test on the data used for fitting
    logreg.fit(x, labels)
    y_pred_prob = logreg.predict_proba(x)
    y_pred_class = logreg.predict(x)
    a = np.mean(y_pred_class == labels)
    acc.append(a)
