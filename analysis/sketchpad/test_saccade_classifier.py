"""
Classify the direction of the upcoming saccade
"""

import os 
import json
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.linear_model import LinearRegression, LinearRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate

import eyelink_parser 
import stim_positions
import fixation_events
import load_data

# Which subject to analyze
n = 1

expt_info = json.load(open('expt_info.json'))

# Load the data
d = load_data.load_data(n)

# Select fixation offsets -- i.e. saccade onsets
row_sel = d['fix_events'][:,2] == expt_info['event_dict']['fix_off']
fix_offset_events = d['fix_events'][row_sel, :]

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
epochs = mne.Epochs(d['raw'], fix_offset_events,
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

# # Plot saccade activity
# evoked = epochs.copy().apply_baseline((None, 0)).average()
# evoked.plot(spatial_colors=True)
# evoked.plot(gfp='only')

# Get the direction of the next saccade
# Check the timing between fixations to make sure
# that the next item on the list is actually the next fixation.
fix = d['fix_info'].copy()
onsets = fix['start'][1:].to_numpy()
offsets = fix['end'][:-1].to_numpy()
isi = onsets - offsets
isi = isi / 1000
plausible_saccade = isi < 0.15
plausible_saccade = np.hstack((plausible_saccade, [None]))
fix['saccade'] = pd.Series(plausible_saccade, index=fix.index)
x_change = fix['x_avg'][1:].to_numpy() - fix['x_avg'][:-1].to_numpy()
x_change = np.hstack((x_change, np.nan))
fix['x_change'] = pd.Series(x_change, index=fix.index)
y_change = fix['y_avg'][1:].to_numpy() - fix['y_avg'][:-1].to_numpy()
y_change = np.hstack((y_change, np.nan))
fix['y_change'] = pd.Series(y_change, index=fix.index)

# Get data structures for running regressions
meg_data = epochs.get_data() # Trial x Channel x Time 
fix = fix.iloc[epochs.selection] # Toss trials that have been marked as "bad"

# Only keep trials that are a real saccade
meg_data = meg_data[fix['saccade'] == True, :]
fix = fix.loc[fix['saccade'] == True]


scaler = StandardScaler()

# Toss weird trials (Should have been done above)
gfp = np.std(meg_data, axis=1) # Global field power
max_gfp = np.max(gfp, axis=1) # Max per trial
zscore = lambda x: (x - x.mean()) / (x.std()) # Func to z-score a vector
bad_trials = zscore(max_gfp) > 4
meg_data = meg_data[~bad_trials,:,:]
fix = fix.loc[~bad_trials]


# Separately predict movement in the x and y directions 
# Or look at sin and cos of the angle of movement?
# - Plus the distance of the saccade
x_change = fix['x_change']
y_change = fix['y_change']

# Details of the classifier calls
cv_params= {'cv': 5, # Cross-validataion
            'n_jobs': 3}
            #'scoring': 'accuracy'}
cv_reg_params = {#'penalty': 'l1', # CV of regularization parameter
                 #'solver': 'saga',
                 #'multi_class': 'multinomial',
                 'max_iter': 1e4}


# Find the best values of C/lambda/alpha
# Only run this once -- not separately for every subject/timepoint
t_cv = 0.1 # Time-point at which we're cross-validating
i_time = np.nonzero(epochs.times >= t_cv)[0][0]
x = meg_data[:,:,i_time]
x = scaler.fit_transform(x)
clf = LassoCV( verbose=1, **cv_reg_params, **cv_params)
clf.fit(x, x_change)
print(clf.alpha_) # Show the regularization parameter
print(np.sum(clf.coef_ != 0)) # Avg number of nonzero coefs
print(clf.score(x, x_change))


clf_params = {#'penalty': 'l1', # Main classifier
              #'solver': 'liblinear',
              #'multi_class': 'ovr',
              'max_iter': 1e4}

# Set up the classifier
clf = Lasso(alpha=clf.alpha_, **clf_params)

results = []
accuracy = []
for i_time in tqdm(range(epochs.times.size)):
    # Select data at this time-point
    x = meg_data[:,:,i_time] 
    # Standardize the data within each MEG channel
    x = scaler.fit_transform(x) 
    # Cross-validated classifiers
    res = cross_validate(clf, x, x_change,
                         return_estimator=True,
                         **cv_params)
    # Store the results
    results.append(res)
    accuracy.append(res['test_score'].mean())

# Plot the results
plt.plot(epochs.times, accuracy)
# plt.plot([epochs.times.min(), epochs.times.max()], # Mark chance level
#          np.array([1, 1]) * (1 / len(np.unique(labels))),
#          '--k')
plt.ylabel('Accuracy')
plt.xlabel('Time (s)')
plt.show()





