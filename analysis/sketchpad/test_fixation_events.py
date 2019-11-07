""" Make a new event for each fixation
"""


import os 
import numpy as np
import mne
import matplotlib.pyplot as plt
import eyelink_parser 

fsample_eyelink = 1000
trial_window = 4.5 # length of the trial to analyze in seconds

event_dict = {'response': 1,
              'fixation': 2,
              'explore': 4,
              'mem_test': 8,
              'drift_correct_start': 16,
              'drift_correct_end': 32} 

# Load the MEG data
fname = '191104/yalitest.fif' 
data_folder = '../data/'
raw_file = os.path.join(data_folder, 'raw', fname) 
raw = mne.io.read_raw_fif(raw_file) 

# Segment out the events
events = mne.find_events(raw,
                         stim_channel='STI101',
                         mask=0b00111111, # Ignore Nata button triggers
                         shortest_event=1)

# Read in the EyeTracker data
fname = '../data/eyelink/ascii/19110415.asc'
edata = eyelink_parser.EyelinkData(fname)

# Get events for the exploration segment of the trial
row_inx = events[:,2] == event_dict['explore']
meg_events = events[row_inx,:]
trial_onsets_meg = meg_events[:,0]

trigs = edata.triggers
row_inx = trigs['value'] == event_dict['explore']
eye_events = trigs.loc[row_inx,:]
trial_onsets_eye = np.array(eye_events['time_stamp'])
trial_offsets_eye = trial_onsets_eye + int(trial_window * fsample_eyelink)

# Make sure MEG and Eyelink data show the same number of trials
assert trial_onsets_eye.size == trial_onsets_meg.size

# How much difference is there in the speed of the clocks?
# Compute drift as a proportion of the total change
drift = np.diff(trial_onsets_meg - trial_onsets_eye) / np.diff(trial_onsets_meg)
# plt.hist(drift)
# plt.xlabel('Drift ratio')
# plt.ylabel('Count')
print("Drift ratio = {:.5f} +/- {:.5f}".format(np.mean(drift), np.std(drift)))

# Find fixations within 4.5 sec of the beginning of each exploration phase
fix = edata.fixations
n_trials = len(trial_onsets_meg)
for i_trial in range(n_trials):
    t_start_meg = trial_onsets_meg[i_trial]
    t_start_eye = trial_onsets_eye[i_trial]
    t_end_eye = trial_offsets_eye[i_trial]
    # Find fixations that start after the stimuli appear
    # and end before the stimuli disappear
    fix_sel = (fix['start'] > t_start_eye) & (fix['end'] < t_end_eye)
    
# Store new data for each fixation
# - Trial number (starting with 0)
# - Time of fixation start in MEG samples
# - TIme of fixation end in MEG samples
fix['trial_number'] = np.nan # Initialize new columns
fix['start_meg'] = np.nan
fix['end_meg'] = np.nan
for i_fix in range(fix.shape[0]):
    t_start_fix = fix['start'][i_fix]
    t_end_fix = fix['end'][i_fix]
    # Which trial is this fixation in?
    # First, find trials with onsets before this fixation
    onset_before_fix = np.nonzero(trial_onsets_eye < t_start_fix)[0]
    # Then get the last trial that began before this fixation
    try:
        trial_inx = onset_before_fix.max()
    except ValueError:
        trial_inx = np.nan
    # Is this fixation after the end of the trial?
    # If so, store it as an NaN
    if not np.isnan(trial_inx):
        if (t_end_fix > trial_offsets_eye[trial_inx]):
            trial_inx = np.nan
    # Store the trial number
    fix.loc[i_fix, 'trial_number'] = trial_inx
    # Store the time in MEG samples
    if not np.isnan(trial_inx):
        trial_start_meg = trial_onsets_meg[trial_inx]
        trial_start_eye = trial_onsets_eye[trial_inx]
        t_diff = trial_start_eye - trial_start_meg
        fix.loc[i_fix, 'start_meg'] = t_start_fix - t_diff
        fix.loc[i_fix, 'end_meg'] = t_end_fix - t_diff

