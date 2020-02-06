""" Get an MEG-trigger--style event structure for each fixation
"""

import os 
import json
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

import eyelink_parser 
import stim_positions


expt_info = json.load(open('expt_info.json')) 


def _closest_stim(x, y):
    """ Find the closest stimulus to the given position
    Return the index of the closest stim, and the distance to it
    """
    stim_locs = stim_positions.stim_positions() # Positions of the stimuli
    pos = np.array([x, y])
    d = [np.linalg.norm(pos - np.array(s)) for s in stim_locs]
    loc = np.argmin(d)
    min_d = np.min(d)
    return loc, min_d


def get_fixation_events(meg_events, eye_data, behav_data):
    """ Get an mne-compatible array of events (in units of MEG samples)
    """ 
    trial_window_sec = 4.5 # length of the trial to analyze
    trial_window_samp = int(trial_window_sec * expt_info['fsample_eyelink']) 

    # Get the onset of each trial in MEG samples
    row_inx = meg_events[:,2] == expt_info['event_dict']['explore']
    meg_events = meg_events[row_inx,:]
    trial_onsets_meg = meg_events[:,0]
    
    # Get the onset of each trial in Eyelink samples
    trigs = eye_data.triggers
    row_inx = trigs['value'] == expt_info['event_dict']['explore']
    trial_onsets_eye = np.array(trigs.loc[row_inx, 'time_stamp'])
    trial_offsets_eye = trial_onsets_eye + trial_window_samp
    
    # Make sure MEG and Eyelink data have the same number of trials
    assert trial_onsets_eye.size == trial_onsets_meg.size
    
    # How much difference is there in the speed of the Eyelink and MEG clocks?
    drift = np.diff(trial_onsets_meg) / np.diff(trial_onsets_eye) 
    msg = "Timing drift ratio = {:.5f} +/- {:.5f}"
    print(msg.format(np.mean(drift), np.std(drift)))
    
    # Store timing data for each fixation
    fix = eye_data.fixations
    fix['trial_number'] = np.nan # Psychopy trial number
    fix['start_meg'] = np.nan # Time of fixation start in MEG samples
    fix['end_meg'] = np.nan # Time of fix end in MEG samples
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
    
    # Check whether the subject looks at the objects that are on the screen 
    fix['closest_loc'] = np.nan # New column for the closest stim location
    fix['closest_stim'] = np.nan # Stimulus at the fixated location
    fix['prev_stim'] = np.nan # Stimulus at the last fixation
    fix['dist_to_stim'] = np.nan # Distance to the center of closest stim
    fix['on_target'] = None
    for i_fix in range(fix.shape[0]):
        f = fix.loc[i_fix]
        trial_number = f['trial_number']
        if np.isnan(trial_number):
            continue
        closest_loc, dist = _closest_stim(f['x_avg'], f['y_avg'])
        trial_info = behav_data.loc[trial_number]
        disp_stims = trial_info['display_stims']
        disp_stims = disp_stims.strip('[]').split(' ') 
        disp_stims = [int(e) for e in disp_stims]
        disp_locs = [trial_info['loc_{}'.format(s)] for s in disp_stims]
        if closest_loc in disp_locs:
            fix.loc[i_fix, 'closest_loc'] = closest_loc
            fix.loc[i_fix, 'closest_stim'] = disp_stims[disp_locs.index(closest_loc)]
            fix.loc[i_fix, 'dist_to_stim'] = dist
            fix.loc[i_fix, 'on_target'] = True
        else:
            fix.loc[i_fix, 'on_target'] = False 

        # Check which stimulus was in the previous fixation
        back_counter = 1
        while True:
            prev_row = fix.loc[i_fix - back_counter]
            if not prev_row['on_target']:
                fix.loc[i_fix, 'prev_stim'] = np.nan
                break
            elif prev_row['closest_stim'] != fix.loc[i_fix, 'closest_stim']:
                fix.loc[i_fix, 'prev_stim'] = prev_row['closest_stim']
                break
            elif back_counter > 10: # More than X fixations on one target?
                fix.loc[i_fix, 'prev_stim'] = np.nan
                break
            else:
                back_counter += 1

    # What proportion of the fixations were on a target?
    on_target = np.sum(fix['on_target'] == True)
    off_target = np.sum(fix['on_target'] == False)
    print('{} fixations'.format(on_target + off_target))
    print('{} on target, {} off target'.format( on_target, off_target))
    print('{:.1f}% on target'.format(100 * (on_target / (on_target + off_target))))
    
    # # How far away were the fixations from their closest target?
    # distances_deg = [dc.pix2deg(d) for d in distances]
    # plt.hist(distances_deg, bins=100,
    #          histtype='stepfilled', density=True)
    # plt.ylabel('Density')
    # plt.xlabel('Degrees') 
    # plt.show()
    # plt.savefig(os.path.expanduser('~/Downloads/fix_dist.pdf'))

    # Which item was in the previous fixation?
    
    # Only keep fixations on the targets
    fix = fix.loc[fix['on_target'] == True]
    
    # Make a new object of MEG-timed events for each fixation
    events_fix = np.zeros([0, 3], dtype=int)
    for event_type in ['start_meg', 'end_meg']:
        if event_type == 'start_meg':
            trig = 'fix_on'
        elif event_type == 'end_meg':
            trig = 'fix_off'
        else:
            trig = 'ERROR'
        evt_samp = np.array(fix[event_type])
        evt_samp = np.reshape(evt_samp, [-1, 1])
        evt_samp = np.int64(evt_samp)
        evt_dur = np.zeros(evt_samp.shape, dtype=int) 
        trig_val = expt_info['event_dict'][trig]
        evt_trig = np.ones(evt_samp.shape, dtype=int) * trig_val
        evt = np.hstack((evt_samp, evt_dur, evt_trig))
        events_fix = np.vstack((events_fix, evt))
    
    return fix, events_fix


def demo():
    """ Demo the script
    """ 
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
    fix_info, events = get_fixation_events(events, eye_data, behav)
    print(fix_info)
    print(events)


if __name__ == '__main__':
    demo()

