"""
Preprocess the data for the 'saccades-attention' project
"""

import os 
import numpy as np
import mne
import matplotlib.pyplot as plt

fname = '191104/yalitest.fif'

data_folder = '../data/'
raw_file = os.path.join(data_folder, 'raw', fname)

raw = mne.io.read_raw_fif(raw_file)

# Inspect the data
fig = raw.plot(butterfly=True)
fig.canvas.key_press_event('a') # Press 'a' to start entering annotations 
# Doesn't work on my lab desktop?
interactive_annot = raw.annotations
#raw.annotations.save('saved-annotations.csv')
#annot_from_file = mne.read_annotations('saved-annotations.csv')

# Plot the spectrum of the activity
raw.plot_psd(fmax=50)

# Segment out the events of interest
# Find the events
events = mne.find_events(raw,
                         #stim_channel=['STI001', 'STI002'],
                         stim_channel='STI101',
                         mask=0b00111111, # Get rid of Nata button triggers
                         shortest_event=1)
u,c = np.unique(events[:,2], return_counts=True)
print(u)
print(c)

# Details about the triggers
event_dict = {'response': 1,
              'fixation': 2,
              'explore': 4,
              'mem_test': 8,
              'drift_correct_start': 16,
              'drift_correct_end': 32} 

# Check out the events
fig = mne.viz.plot_events(events, event_id=event_dict, sfreq=raw.info['sfreq']) 
fig.subplots_adjust(right=0.7)

# Only look at MEG channels
raw.pick_types(meg=True)

# Automatic artifact rejection

# ICA
raw_downsamp = mne.io.read_raw_fif(raw_file, preload=True)
raw_downsamp.resample(sfreq=100)
ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(raw_downsamp)

# Plot ICA results
ica.plot_components(inst=raw_downsamp) # Scalp topographies
ica.plot_sources(raw_downsamp) # Time-courses
ica.plot_properties(raw_downsamp, picks=[0, 2])

# Exclude the components that are artifacts
ica.exclude = [2]
ica.apply(raw) # Changes the `raw` object in place

# Check how the data changes when components are excluded
ica.plot_overlay(raw_downsamp, exclude=[2], picks='mag')
ica.plot_overlay(raw_downsamp, exclude=[2], picks='grad')

ica.plot_properties(raw_downsamp, picks=ica.exclude)

# Check whether ICA worked as expected
orig_raw = raw.copy()
raw.load_data()
ica.apply(raw) # This 
orig_raw.plot()
raw.plot()


## Automatically find components that match the EOG recordings
ica.exclude = [] # Empty out the excluded comps (for testing the pipeline)
# find which ICs match the EOG pattern
eog_indices, eog_scores = ica.find_bads_eog(raw)
ica.exclude = eog_indices 
# barplot of ICA component "EOG match" scores
ica.plot_scores(eog_scores) 
# plot diagnostics
ica.plot_properties(raw, picks=eog_indices) 
# plot ICs applied to raw data, with EOG matches highlighted
ica.plot_sources(raw) 
# plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
ica.plot_sources(eog_evoked)
## A similar thing for heartbeat: ica.find_bads_ecg (method='correlation')


# What's the delay between the different parts of the button-box triggers?
def get_trig_times(trig):
    trig_inx = events[:,2] == trig
    trig_times = events[trig_inx, 0]
    return trig_times

trig_times = {trig: get_trig_times(trig) for trig in u}

def compare_trig_times(trig1, trig2):
    t1 = trig_times[trig1]
    t2 = trig_times[trig2]
    a1 = np.tile(t1, [t2.size, 1])
    a2 = np.tile(t2, [t1.size, 1])
    d = a1 - a2.transpose()
    return d

plt.plot(np.diag(compare_trig_times(5888, 5889)))



# Epoch the data
raw.load_data()
raw.filter(l_freq=0.5, h_freq=30)
epochs = mne.Epochs(raw, events,
                    event_id={'explore':4},
                    tmin=-0.5, tmax=0.5,
                    #reject=reject_criteria,
                    preload=True)

# Plot ERF at one channel
epochs.plot_image(picks=['MEG2423'])

# Calculate and plot averaged activity
# Channels combined using GFP
evoked = epochs.average()
mne.viz.plot_compare_evokeds({'explore': evoked},
                             legend='upper left',
                             show_sensors='lower right')

evoked.plot_joint()


###############
# Eye-tracker #
###############

# Read in the EyeTracker data
import eyelink_parser
fname = '../data/eyelink/ascii/19110415.asc'
edata = eyelink_parser.EyelinkData(fname)

# Get events for the main part of the trial
row_inx = events[:,2] == event_dict['explore']
meg_events = events[row_inx,:]

trigs = edata.triggers
row_inx = trigs['value'] == event_dict['explore']
eye_events = trigs.loc[row_inx,:]

# Make sure MEG and Eyelink data show the same number of trials
assert meg_events.shape[0] == eye_events.shape[0]




