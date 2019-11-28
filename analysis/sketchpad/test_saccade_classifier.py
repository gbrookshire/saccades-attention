"""
Classify the direction of the upcoming saccade
"""

# TODO
# Cross-correlation in saccade direction?
# - This will make sure that successful classification isn't due to 
#   post-saccade effects + multiple saccades in the same direction.

import os 
import json
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
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
tmin = -1.0
tmax = 0.5
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
# Time with n_jobs=1: 54.1 s 
# Time with n_jobs=3: 46.9 s
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

# Get data structures for running regressions
meg_data = epochs.get_data() # Trial x Channel x Time 
fix = fix.iloc[epochs.selection] # Toss trials that have been marked as "bad"
scaler = StandardScaler()

# Only keep trials that are a real saccade
meg_data = meg_data[fix['saccade'] == True, :]
fix = fix.loc[fix['saccade'] == True]

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
x_change = fix['x_change'].to_numpy()
y_change = fix['y_change'].to_numpy()

# Details of the classifier calls
# Cross-validataion params
cv_params= {'cv': 5, 
            'n_jobs': 3}
# CV of regularization params
cv_reg_params = {'selection': 'random', 
                 'max_iter': 1e5}
# Main classifier
clf_params = {'selection': 'random', # Speeds up model fitting
              'max_iter': 1e4}


"""
# Find the best values of C/lambda/alpha
# Only run this once -- not separately for every subject/timepoint
t_cv = -0.05 # Time-point at which we're cross-validating
i_time = np.nonzero(epochs.times >= t_cv)[0][0]
x = meg_data[:,:,i_time]
x = scaler.fit_transform(x)
clf = LassoCV( verbose=1, **cv_reg_params, **cv_params)
clf.fit(x, x_change)
print(clf.alpha_) # Show the regularization parameter
print(np.sum(clf.coef_ != 0)) # Avg number of nonzero coefs
print(clf.score(x, x_change)) # R^2
plt.plot(x_change, clf.predict(x), 'o', alpha=0.5)
"""

# Set up the main classifier
alpha = 3.57 # Identified using CV above
clf = Lasso(alpha=alpha, **clf_params)

results = []
accuracy = []
for i_time in tqdm(range(epochs.times.size)):
    #######np.random.shuffle(x_change)
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
f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})

# Plot classifier accuracy over time
a0.plot(epochs.times, accuracy)
a0.set_ylabel('$R^2$')

# Plot saccade durations
bins = np.arange(epochs.times[0], epochs.times[-1], step=0.01) 
a1.hist(fix['saccade_dur'], bins=bins)
a1.set_xlabel('Time (s)')
a1.set_ylabel('Count')

plt.tight_layout()
plt.show()


