"""
Load the data for one participant
"""

import json
import numpy as np
import pandas as pd
import mne
#import matplotlib.pyplot as plt
#from tqdm import tqdm

import eyelink_parser 
import stim_positions
import fixation_events

expt_info = json.load(open('expt_info.json'))
subject_info = pd.read_csv(expt_info['data_dir'] + 'subject_info.csv')
data_dir = expt_info['data_dir']
    

def load_data(n):
    # Read in the MEG data
    subj_fname = subject_info['meg'][n]
    raw_fname = f'{data_dir}raw/{subj_fname}.fif'
    raw = mne.io.read_raw_fif(raw_fname)
    meg_events = mne.find_events(raw, # Segment out the MEG events
                                 stim_channel='STI101',
                                 mask=0b00111111, # Ignore Nata button triggers
                                 shortest_event=1)
    
    # Read in artifact definitions
    subj_fname = subj_fname.replace('/', '_')
    annot_fname = f'{data_dir}annotations/{subj_fname}.csv'
    annotations = mne.read_annotations(annot_fname)
    raw.set_annotations(annotations)
    ica_fname = f'{data_dir}ica/{subj_fname}-ica.fif'
    ica = mne.preprocessing.read_ica(ica_fname)
    
    # Read in the EyeTracker data
    eye_fname = f'{data_dir}eyelink/ascii/{subject_info["eyelink"][n]}.asc'
    eye_data = eyelink_parser.EyelinkData(eye_fname)
    
    # Load behavioral data
    behav_fname = f'{data_dir}logfiles/{subject_info["behav"][n]}.csv'
    behav = pd.read_csv(behav_fname) 
    
    # Get the fixation events
    fix_info, events = fixation_events.get_fixation_events(meg_events,
                                                           eye_data,
                                                           behav)
    # Look at the beginning of the fixation
    row_sel = events[:,2] == expt_info['event_dict']['fix_on']
    fix_events = events[row_sel, :]
    # Don't look at multiple fixations to the same object
    new_obj = np.diff(fix_info['closest_stim']) != 0
    new_obj = np.hstack((True, new_obj)) # First fixation is to a new object
    fix_info = fix_info.loc[new_obj]
    fix_events = fix_events[new_obj,:]

    # Put all the data into a dictionary
    data = {}
    data['n'] = n
    data['raw'] = raw
    data['ica'] = ica
    data['eye'] = eye_data
    data['behav'] = behav
    data['fix_info'] = fix_info
    data['fix_events'] = fix_events
    data['meg_events'] = meg_events
    
    return data
