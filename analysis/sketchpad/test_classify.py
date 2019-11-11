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

scaler = StandardScaler()

# Toss weird trials (Should have been done above)
gfp = np.std(d, axis=1) # Global field power
max_gfp = np.max(gfp, axis=1) # Max per trial
bad_trials = max_gfp > (np.std(max_gfp) * 6)
bad_trials = np.nonzero(bad_trials)[0]
d = np.delete(d, bad_trials, axis=0)
labels = np.delete(labels, bad_trials, axis=0)

acc = []
n_nonzero_coefs = []
#for i_time in range(epochs.times.size):
for i_time in tqdm(range(epochs.times.size)):
    x = d[:,:,i_time]
    #y = (labels == stim).astype(int) # For looking at individual labels

    # Standardize the data within each MEG channel
    x = scaler.fit_transform(x)

    #linreg = LinearRegression(normalize=True) # Check about normalize
    #linreg.fit(x, y)
    #y_pred = linreg.predict(x)
    #mse = np.mean((y - y_pred) ** 2)

    logreg = LogisticRegression(penalty='l1',
                                solver='liblinear', 
                                # Here's a value of C that works. How do 
                                # I look for this number programmatically?
                                C=0.05,
                                # How does this set the alpha (lambda?) parameter?
                                # Fit a separate binary classifier for each label
                                multi_class='ovr',
                                #n_jobs=3,
                                max_iter=1e4,
                                )
    #logreg = LogisticRegression(penalty='none', solver='saga')
    # It's definitely overfitting to test on the data used for fitting
    logreg.fit(x, labels)
    #plt.imshow(logreg.coef_, aspect='auto')
    n_coef = np.mean(np.sum(logreg.coef_ != 0, axis=1)) # How many nonzero coefficients?
    n_nonzero_coefs.append(n_coef)
    # y_pred_prob = logreg.predict_proba(x)
    # y_pred_class = logreg.predict(x)
    # a = np.mean(y_pred_class == labels)
    a = logreg.score(x, labels)
    acc.append(a)

# Try out cross-validation
y = np.array(labels == 0) # Try a classifier for one label
i_time = 60
x = d[:,:,i_time]
x = scaler.fit_transform(x)
logreg_cv = LogisticRegressionCV(
                Cs=np.logspace(-10, 5, 15),
                cv=5, # N folds
                penalty='l1', # LASSO
                solver='liblinear', 
                n_jobs=3,
                max_iter=1e4,
                #scoring='roc_auc',
                #multi_class='ovr' # Fit a binary problem for each label
                )
logreg_cv.fit(x, y)
print(logreg_cv.C_)
print(logreg_cv.scores_)
