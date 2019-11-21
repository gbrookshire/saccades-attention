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

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate

import eyelink_parser 
import stim_positions
import fixation_events
import load_data

# Which subject to analyze
n = 1

# Load the data
d = load_data.load_data(n)

# Epoch the data
tmin = -0.4
tmax = 0.4
picks = mne.pick_types(d['raw'].info,
                       meg=True, eeg=False, eog=False,
                       stim=False, exclude='bads')
reject = dict(grad=4000e-13, # T / m (gradiometers)
              mag=4e-12, # T (magnetometers)
              #eeg=40e-6, # V (EEG channels)
              #eog=250e-6 # V (EOG channels)
              ) 
epochs = mne.Epochs(d['raw'], d['fix_events'],
                    tmin=tmin, tmax=tmax, 
                    #reject=reject,
                    reject_by_annotation=True,
                    preload=True,
                    baseline=None,
                    picks=picks)

# Reject ICA artifacts
d['ica'].apply(epochs)

# Resample after epoching to make sure trigger times are correct
epochs.resample(200, n_jobs=3)

# # Plot activity evoked by an eye movement
# evoked = epochs.copy().apply_baseline((None, 0)).average()
# evoked.plot(spatial_colors=True)
# # evoked.plot(gfp='only')
# # times = np.arange(-0.05, 0.2, 0.05)
# # evoked.plot_topomap(times=times, ch_type='grad')
# # evoked.plot_topomap(times=times, ch_type='mag')

# Classifiers
meg_data = epochs.get_data() # Trial x Channel x Time
labels = d['fix_info']['closest_stim'] # Stimulus to decode
labels = labels.astype(int).to_numpy()
labels = labels[epochs.selection] # Only keep retained trials

scaler = StandardScaler()

# Toss weird trials (Should have been done above)
gfp = np.std(meg_data, axis=1) # Global field power
max_gfp = np.max(gfp, axis=1) # Max per trial
zscore = lambda x: (x - x.mean()) / (x.std()) # Func to z-score a vector
bad_trials = zscore(max_gfp) > 4
meg_data = meg_data[~bad_trials,:,:]
labels = labels[~bad_trials]

# Details of the classifier calls
clf_params = {'penalty': 'l1', # Main classifier
              'solver': 'liblinear',
              'multi_class': 'ovr',
              'max_iter': 1e4}
cv_params= {'cv': 5, # Cross-validataion
            'n_jobs': 3,
            'scoring': 'accuracy'}
cv_reg_params = {'penalty': 'l1', # CV of regularization parameter
                 'solver': 'saga',
                 'multi_class': 'multinomial',
                 'max_iter': 1e4}

# Find the best values of C/lambda/alpha
# Only run this once -- not separately for every subject/timepoint
t_cv = 0.1 # Time-point at which we're cross-validating
i_time = np.nonzero(epochs.times >= t_cv)[0][0]
x = meg_data[:,:,i_time]
x = scaler.fit_transform(x)
clf = LogisticRegressionCV(Cs=np.linspace(0.001, 1, 20),
                           verbose=1,
                           **cv_reg_params,
                           **cv_params)
clf.fit(x, labels)
print(clf.C_) # Show the regularization parameter
print(np.mean(np.sum(clf.coef_ != 0, axis=1))) # Avg number of nonzero coefs
print(clf.score(x, labels))

# Set up the classifier
clf = LogisticRegression(C=0.05, **clf_params)

# Run the classifier for each time-point
results = []
accuracy = []
for i_time in tqdm(range(epochs.times.size)):
    # Select data at this time-point
    x = meg_data[:,:,i_time] 
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
plt.show()


