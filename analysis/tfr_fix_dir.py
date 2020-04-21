"""
Analyze TFRs around saccades to the right vs to the left.
Do we see alpha power asymmetries that predict the direction of the saccade?
"""

import json
import socket
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from mne.time_frequency import tfr_morlet
import load_data

expt_info = json.load(open('expt_info.json'))

if socket.gethostname() == 'colles-d164179':
    data_dir = expt_info['data_dir']['external']
else:
    data_dir = expt_info['data_dir']['standard']


def compute_alpha_asym(n):
    # Load the data
    d = load_data.load_data(n)
    # Select fixation offsets -- i.e. saccade onsets
    row_sel = d['fix_events'][:,2] == expt_info['event_dict']['fix_off']
    events = d['fix_events'][row_sel, :] 
    # Get the feature to reconstruct: the direction of the next saccade
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

    # Only keep trials that didn't have another eye movement too recently
    prior_saccade_thresh = 250 # In samples (i.e. ms)
    prior_saccade_time = events[1:,0] - events[:-1,0]
    too_close = prior_saccade_time < 250
    too_close = np.hstack([[False], too_close])
    
    # Apply the selections
    trial_sel = fix['saccade'] & ~too_close
    fix = fix.loc[trial_sel]
    events = events[trial_sel,:]
    
    # Reject ICA artifacts
    d['raw'].load_data()
    d['ica'].apply(d['raw']) 
    
    picks = mne.pick_types(d['raw'].info,
                        meg=True, eeg=False, eog=False,
                        stim=False, exclude='bads')
    
    # Compute power separately for saccades to the left and right
    epochs = {}
    power = {}
    for sac_direction in ('left', 'right'):
    
        # Select fixations in the chosen direction
        if sac_direction == 'left':
            trial_inx = fix['x_change'] < 0
        elif sac_direction == 'right':
            trial_inx = fix['x_change'] > 0
        
        # Epoch the data
        epochs[sac_direction] = mne.Epochs(d['raw'],
                                           events[trial_inx,:],
                                           reject_by_annotation=True,
                                           preload=True,
                                           baseline=None,
                                           picks=picks,
                                           proj=True,
                                           tmin=-1.0, tmax=0.5) 
        # Compute the TFR
        freqs = n_cycles = 3
        tfr_params = {'freqs': np.logspace(*np.log10([4, 40]), num=20),
                      'n_cycles': 3,
                      'use_fft': True,
                      'return_itc': False,
                      'decim': 10,
                      'n_jobs': 5}
        power[sac_direction] = tfr_morlet(epochs[sac_direction], **tfr_params)
    
    # Get the difference between left and right
    power['diff'] = power['left'] - power['right']

    return power


def plot(power):
    # Plot this difference
    # We're taking left - right, so we should expect to see a pattern
    # consistent with attention toward the left. At left occipital channels,
    # this should appear as an INCREASE in alpha power, and at right occipital
    # channels, this should appear as a DECREASE in alpha power.
    power['diff'].plot_topomap(ch_type='grad',
                               tmin=-0.3, tmax=-0.0,
                               fmin=8, fmax=12)

    # Plot the overall difference
    pick_grads = mne.pick_types(power['diff'].info, meg='grad') 
    power['diff'].plot_topo(picks=pick_grads)


def aggregate():
    # Combine across all participants
    import everyone
    def load_power(row):
        fname = f"{data_dir}tfr/{row['n']}_diff-tfr.h5"
        power = mne.time_frequency.read_tfrs(fname)[0]
        return power
    power_list = everyone.apply_fnc(load_power)
    power_all = np.full([len(power_list)] + list(power_list[0].data.shape), np.nan)
    for i_sub in range(len(power_list)):
        power_all[i_sub,...] = power_list[i_sub].data
    power_avg = mne.time_frequency.AverageTFR(power_list[0].info,
                                              power_all.mean(axis=0),
                                              power_list[0].times,
                                              power_list[0].freqs,
                                              len(power_list))

    # Plot the average of the difference between left- and right-saccades
    plot_params = dict(ch_type='grad',
                       tmin=-0.3, tmax=-0.15,
                       fmin=8, fmax=12,
                       vmax=lambda x: np.max(np.abs(x)), 
                       vmin=lambda x: -np.max(np.abs(x)), 
                       show=False,
                       contours=False,
                       colorbar=False)
    power_avg.plot_topomap(**plot_params)
    plt.tight_layout()
    fname = f"{data_dir}plots/alpha_asym/alpha_asym_avg.png"
    plt.savefig(fname)
    plt.show()

    # Plot the difference for each subject
    i_plot = 1
    for power in power_list:
        ax = plt.subplot(1, 5, i_plot)
        i_plot += 1
        power.plot_topomap(axes=ax, **plot_params)
    plt.tight_layout()
    fname = f"{data_dir}plots/alpha_asym/alpha_asym_by_subj.png"
    plt.savefig(fname)
    plt.show()


def overall_tfr(n):
    # Load the data
    d = load_data.load_data(n)
    # Select fixation offsets -- i.e. saccade onsets
    row_sel = d['fix_events'][:,2] == expt_info['event_dict']['fix_off']
    events = d['fix_events'][row_sel, :] 

    # Reject ICA artifacts
    d['raw'].load_data()
    d['ica'].apply(d['raw']) 
    
    # Select MEG channels
    picks = mne.pick_types(d['raw'].info,
                           meg=True, eeg=False, eog=False,
                           stim=False, exclude='bads')

    # Epoch the data
    epochs = mne.Epochs(d['raw'],
                        events,
                        reject_by_annotation=True,
                        preload=True,
                        baseline=None,
                        picks=picks,
                        proj=True,
                        tmin=-1.0, tmax=1.0)

    # Compute the TFR
    freqs = n_cycles = 3
    tfr_params = {'freqs': np.logspace(*np.log10([4, 40]), num=20),
                    'n_cycles': 3,
                    'use_fft': True,
                    'return_itc': True,
                    'decim': 10,
                    'n_jobs': 5}
    power, itc = tfr_morlet(epochs, **tfr_params)
    return power, itc


def aggreagate_itpc():
    """ Take the average ITPC across subjects
    """
    import everyone
    def load_itpc(row):
        fname = f"{data_dir}/tfr/{row['n']}_overall-itc-tfr.h5"
        itpc = mne.time_frequency.read_tfrs(fname)[0]
        return itpc
    itpc_list = everyone.apply_fnc(load_itpc) # Data for each subject
    # Combine into averaged data across subjects
    itpc_all = np.full([len(itpc_list)] + list(itpc_list[0].data.shape),
                       np.nan)
    for i_sub in range(len(itpc_list)):
        itpc_all[i_sub,...] = itpc_list[i_sub].data
    itpc_avg = mne.time_frequency.AverageTFR(itpc_list[0].info,
                                             itpc_all.mean(axis=0),
                                             itpc_list[0].times,
                                             itpc_list[0].freqs,
                                             len(itpc_list))

    # # Plot one subject
    # n = 2
    # itpc_list[3].plot_joint()

    # # Plot the average over subjects
    # itpc_avg.plot_joint()



if __name__ == '__main__':
    import sys
    try:
        n = sys.argv[1]
    except IndexError:
        n = input('Subject number: ')
    n = int(n)

    # # Alpha asymmetries
    # power = compute_alpha_asym(n)
    # for side in ('left', 'right', 'diff'):
    #     fname = f"{data_dir}tfr/{n}_{side}-tfr.h5"
    #     power[side].save(fname, overwrite=True)

    # Compute TFR and ITPC
    power, itc = overall_tfr(n)
    power.save(f"{data_dir}/tfr/{n}_overall-pow-tfr.h5",
               overwrite=True)
    itc.save(f"{data_dir}/tfr/{n}_overall-itc-tfr.h5",
             overwrite=True)

