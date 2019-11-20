"""
Tools to identify and reject artifacts
"""

import os
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
    subj_fname = subj_fname.replace('/', '_')
    annot_fname = f'{data_dir}annotations/{subj_fname}.csv'
    if os.path.isfile(annot_fname):
        print(f'Artifact annotations already exist: {annot_fname}') 
        resp = input('Overwrite? (y/n): ')
        if resp in 'Nn':
            print('Loading old artifact annotations')
            annotations = mne.read_annotations(annot_fname)
        elif resp in 'Yy':
            print('Creating new artifact annotations')
        else:
            print(f'Option not recognized -- exiting') 
            return None
    else:
        annotations = identify_manual(raw)
        annotations.save(annot_fname)
    raw.set_annotations(annotations)
            
    # ICA
    #### Do this on the epoched data -- ???
    raw_downsamp = downsample(raw, 10) # Downsample before ICA
    ica = identify_ica(raw_downsamp)
    ica_fname = f'{data_dir}ica/{subj_fname}-ica.fif'
    ica.save(ica_fname)

    # # Check whether ICA worked as expected
    # orig_raw = raw.copy()
    # raw.load_data()
    # ica.apply(raw) # Changes the `raw` object in place
    # orig_raw.plot()
    # raw.plot()

    # Automatic artifact rejection
    #### Do this on the epoched data
    raw = identify_auto(raw)

    return raw


def downsample(raw, downsample_factor):
    """ Resample a raw data object without any filtering.
        This is only for use in ICA. Using this on other
        analyses could result in aliasing.
    """
    assert type(downsample_factor) is int
    assert downsample_factor > 1
    d = raw.get_data()
    decim_inx = np.arange(d.shape[1], step=downsample_factor)
    d = d[:, decim_inx]
    info = raw.info.copy()
    info['sfreq'] /= downsample_factor 
    first_samp = raw.first_samp / downsample_factor # Adj for beg of recording
    raw_downsamp = mne.io.RawArray(d, info, first_samp=first_samp) 
    raw_downsamp.set_annotations(raw.annotations)
    return raw_downsamp


def identify_manual(raw):
    """ Manually identify raw artifacts
    First click on "Add label"
    Edit the label -- glitch, jump, etc
    Click and drag to set a new annotation (with some delay)
    """
    raw_annot = raw.copy()
    raw_annot.load_data()
    raw_annot.pick(['meg', 'eog', 'stim'])
    raw_annot.filter(0.5, 40, picks=['meg', 'eog'])
    # Initialize an event
    init_annot = mne.Annotations(onset=[0],
                                 duration=0.001,
                                 description=['BAD_manual'])
    raw_annot.set_annotations(init_annot)
    # Plot the data
    fig = raw_annot.plot(butterfly=True)
    fig.canvas.key_press_event('a') # Press 'a' to start entering annotations 
    input('Press ENTER when finished tagging artifacts')
    raw_annot.annotations.delete(0) # Delete the annotation used for initializing
    return raw_annot.annotations


def identify_ica(raw):
    """ Use ICA to reject artifacts
    """
    # Perform ICA
    ica = mne.preprocessing.ICA(
            n_components=20, # Number of components to return
            max_pca_components=None, # Don't reduce dimensionality too much
            random_state=0,
            max_iter=800,
            verbose='INFO')
    ica.fit(raw, reject_by_annotation=True)
    
    # Plot ICA results
    ica.plot_components(inst=raw) # Scalp topographies - Click for more info
    ica.plot_sources(raw) # Time-courses - click on the ones to exclude
    
    # ###### Automatically find components that match the EOG recordings
    # ica.exclude = [] # Empty out the excluded comps (for testing the pipeline)
    # # find which ICs match the EOG pattern
    # eog_indices, eog_scores = ica.find_bads_eog(raw)
    # ica.exclude = eog_indices 
    # # barplot of ICA component "EOG match" scores
    # ica.plot_scores(eog_scores) 
    # # plot diagnostics
    # ica.plot_properties(raw, picks=eog_indices) 
    # # plot ICs applied to raw data, with EOG matches highlighted
    # ica.plot_sources(raw) 
    # # plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
    # ica.plot_sources(eog_evoked)
    # ###### A similar thing for heartbeat: ica.find_bads_ecg (method='correlation') 

    input('Press ENTER when finished marking bad components')
    return ica


#     # # Check how the data changes when components are excluded
#     # ica.plot_overlay(raw, exclude=[2], picks='mag')
#     # ica.plot_overlay(raw, exclude=[2], picks='grad')
#     # 
#     # ica.plot_properties(raw, picks=ica.exclude)
#     # 
