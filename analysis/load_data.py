"""
Load the data for one participant
"""

import os
import json
import socket
import pandas as pd
import mne

import eyelink_parser 
import fixation_events

expt_info = json.load(open('expt_info.json'))

if socket.gethostname() == 'colles-d164179':
    data_dir = expt_info['data_dir']['external']
else:
    data_dir = expt_info['data_dir']['standard']

subject_info = pd.read_csv(data_dir + 'subject_info.csv',
                           engine='python', sep=',')
    

def load_data(n):
    subj_fname = subject_info['meg'][n]
    # Read in the MEG data
    raw = mne.io.read_raw_fif(meg_filename(subj_fname))
    print('Finding MEG events')
    meg_events = mne.find_events(raw, # Segment out the MEG events
                                 stim_channel='STI101',
                                 mask=0b00111111, # Ignore Nata button triggers
                                 shortest_event=1)
    
    # Read in artifact definitions
    print('Loading artifact definitions')
    subj_fname = subj_fname.replace('/', '_')
    annot_fname = f'{data_dir}annotations/{subj_fname}.csv'
    annotations = mne.read_annotations(annot_fname)
    raw.set_annotations(annotations)
    ica_fname = f'{data_dir}ica/{subj_fname}-ica.fif'
    ica = mne.preprocessing.read_ica(ica_fname)
    
    # Read in the EyeTracker data
    print('Loading eye-tracker data')
    eye_fname = f'{data_dir}eyelink/ascii/{subject_info["eyelink"][n]}.asc'
    eye_data = eyelink_parser.EyelinkData(eye_fname)
    
    # Load behavioral data
    print('Loading behavioral data')
    behav_fname = f'{data_dir}logfiles/{subject_info["behav"][n]}.csv'
    behav = pd.read_csv(behav_fname) 
    
    # Get the fixation events
    print('Loading fixation events')
    fix_info, fix_events = fixation_events.get_fixation_events(meg_events,
                                                               eye_data,
                                                               behav)

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


def meg_filename(subj_fname): 
    subj_dir = f"{data_dir}raw/{subj_fname}" 
    dir_cont = os.listdir(subj_dir)
    assert len(dir_cont) == 1
    subj_dir = f"{subj_dir}/{dir_cont[0]}"
    raw_fname = f"{subj_dir}/{os.listdir(subj_dir)[0]}"
    return raw_fname
