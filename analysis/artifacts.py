"""
Tools to identify and reject artifacts
"""

import json
import numpy as np
import pandas as pd
import mne

expt_info = json.load(open('expt_info.json'))
subject_info = pd.read_csv(expt_info['data_dir'] + 'subject_info.csv')


def identify_artifacts(n):
    """ Identify artifacts for a given subject number.
    """
    data_dir = expt_info['data_dir']
    subj_fname = subject_info['meg'][n]

    # Read in the data
    raw_fname = f'{data_dir}raw/{subj_fname}.fif'
    raw = mne.io.read_raw_fif(raw_fname)

    # Manually mark bad segments
    annotations = identify_manual(raw)
    raw = reject_manual(raw, annotation) # Reject manual artifacts first

    # ICA
    raw_downsamp = downsample(raw, 10) # Downsample before ICA
    ica = identify_ica(raw_downsamp)
    raw = reject_ica(raw, ica)

    # Automatic artifact rejection
    #### Do this on the epoched data
    raw = identify_auto(raw)

    return raw


def downsample(raw, downsample_factor):
    """ Resample a raw data object without any filtering
    """
    assert type(downsample_factor) is int
    assert downsample_factor > 1
    d = raw.get_data()
    decim_inx = np.arange(d.shape[1], step=downsample_factor)
    d = d[:, decim_inx]
    info = raw.info.copy()
    raw_decim = mne.io.RawArray(d, info)
    return raw_decim


def identify_manual(raw):
    """ Manually identify raw artifacts
    First click on "Add label"
    Edit the label -- glitch, jump, etc
    Click and drag to set a new annotation (with some delay)
    """
    fig = raw.plot(butterfly=True)
    fig.canvas.key_press_event('a') # Press 'a' to start entering annotations 
    return raw.annotations
    #raw.annotations.save('saved-annotations.csv')
    #annot_from_file = mne.read_annotations('saved-annotations.csv') 


def reject_manual(raw, annotations):
    pass


def identify_ica(raw):
    """ Use ICA to reject artifacts
    """
    # Perform ICA
    ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
    ica.fit(raw)
    
    # Plot ICA results
    ica.plot_components(inst=raw) # Scalp topographies
    ica.plot_sources(raw) # Time-courses
    #####ica.plot_properties(raw, picks=[0, 2]) # What does this do?
    
    ###### Automatically find components that match the EOG recordings
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
    ###### A similar thing for heartbeat: ica.find_bads_ecg (method='correlation') 


def reject_ica(raw, ica, exclude):
    # Exclude the components that are artifacts
    ica.exclude = exclude
    ica.apply(raw) # Changes the `raw` object in place
    return raw
    
    # # Check how the data changes when components are excluded
    # ica.plot_overlay(raw, exclude=[2], picks='mag')
    # ica.plot_overlay(raw, exclude=[2], picks='grad')
    # 
    # ica.plot_properties(raw, picks=ica.exclude)
    # 
    # # Check whether ICA worked as expected
    # orig_raw = raw.copy()
    # raw.load_data()
    # ica.apply(raw) # This 
    # orig_raw.plot()
    # raw.plot()
