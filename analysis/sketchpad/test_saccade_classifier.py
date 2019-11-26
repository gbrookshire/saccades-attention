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

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
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
d['fix_events'] = d['fix_events'][row_sel, :]

# Get the direction of the next fixation
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


# Separately predict movement in the x and y directions 
# Or look at sin and cos of the angle of movement?
# - Plus the distance of the saccade









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

