"""
Compute the temporal response function (TRF) of saccades. 

Ole wants individual trials
- try locking this with the EOG (or diff(eog))
Compute TRF for each trial (many fixations per trial)
Then average over TRF

"""

import sys
import socket
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from pymtrf.mtrf import mtrf_predict, mtrf_crossval, mtrf_train, lag_gen
import load_data

expt_info = json.load(open('expt_info.json')) 

if socket.gethostname() == 'colles-d164179':
    data_dir = expt_info['data_dir']['external']
else:
    data_dir = expt_info['data_dir']['standard']


def simulation_test():
    """ Test the TRF script with simulated data
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pymtrf.mtrf import mtrf_train

    fs = 100. # Sampling rate
    dur = 1000 # Duration in s
    t = np.arange(0, dur, 1 / fs) # Time in s
    n = len(t) # Number of samples 
    k = 100 # Number of events
    kernel = np.hstack([np.zeros(25), np.hanning(25)]) # To convolve with stims
    y = np.zeros(n) # Make a train of stimulus onsets
    y[np.random.choice(n, size=k, replace=False)] = 1 # Inserts stim onsets
    x = np.convolve(y, kernel, 'same') # Make the output
    x = x + np.random.normal(size=x.shape, scale=0.2) # Add some noise

    plt.clf()

    # Compute the TRF for a range of regularization parameters
    y = np.reshape(y, [-1, 1]) # Reshape for mtrf_train()
    x = np.reshape(x, [-1, 1])
    lambda_exp = np.arange(-1, 5)
    colors = plt.cm.cool(np.linspace(0, 1, len(lambda_exp)))
    w_list = []
    for i_reg, reg_lambda_exp in enumerate(lambda_exp):
        w, t, i = mtrf_train(stim=y,
                            resp=x, # array (time, chan)
                            fs=fs, # Sampling freq
                            mapping_direction=1, # Forward model
                            tmin=-240, # Min time in ms
                            tmax=250, # Max time in ms
                            reg_lambda=10.0 ** reg_lambda_exp)
        w_list.append(w)

        plt.plot(t, w[0,:,:],
                color=colors[i_reg],
                label=f'$\lambda = 10^{{ {reg_lambda_exp} }}$')

    plt.plot(t, kernel, 'k', label='kernel')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Amp')
    plt.show()


def trf_eog_trialwise(n):
    """ Compute average TRF based on EOG for each trial (with multiple fixations).
    """

    d = load_data.load_data(n)
    raw = d['raw']
    raw.load_data()
    
    # Only keep some channels
    raw.pick_types(meg=True, eog=True, exclude='bads')
    
    # Reject artifacts using ICA
    d['ica'].apply(raw) 
    
    # BP filter MEG activity
    raw.filter(l_freq=1, h_freq=40, # band-pass filter 
               picks=['meg'],
               method='fir', phase='minimum' # causal filter
               n_jobs=5)

    # LP Filter EOG activity
    raw.filter(l_freq=None, h_freq=40, # band-pass filter 
               picks=['eog'],
               method='fir', phase='minimum' # causal filter
               )

    # Take the temporal derivative of EOG to look at changes in eye pos
    raw.apply_function(fun=np.gradient,
                       picks=['eog'])
    
    # Segment into epochs
    epochs = mne.Epochs(raw,
                        events=d['meg_events'],
                        event_id=expt_info['event_dict']['explore'],
                        tmin=0.5,
                        tmax=4.5,
                        baseline=None, #(None, 0),
                        detrend=1, # Linear detrending
                        reject_by_annotation=True, #FIXME How can we reject artifacts?
                        preload=True,
                        proj=True) # Should this be False after running ICA?
    raw.close()
    
    # Resample after epoching to make sure trigger times are correct
    epochs.resample(100, n_jobs=5)
    
    # Get the data. (epoch, channel, time)
    x = epochs.get_data()
    tmin = -0.5
    tmax = 0.5
    eog_chan_inx = mne.pick_types(epochs.info, meg=False, eog=True)
    meg_chan_inx = mne.pick_types(epochs.info, meg=True, eog=False)

    # # Run cross-validation to find the optimal lambda parameter
    # # Cross-validate for the time that's expected to show the strongest response
    # x_meg_cv = np.swapaxes(x[:, meg_chan_inx, :], 1, 2) # (trial, time, chan)
    # x_eog_cv = np.swapaxes(x[:, eog_chan_inx, :], 1, 2) # (trial, time, chan)
    # lambda_cv_exp = np.linspace(-10, 10, 5)
    # lambda_cv_vals = 10 ** lambda_cv_exp
    # r, p, mse, pred, model = mtrf_crossval(stim=x_eog_cv,
    #                                        resp=x_meg_cv,
    #                                        fs=epochs.info['sfreq'],
    #                                        mapping_direction=1,
    #                                        tmin=tmin,
    #                                        tmax=tmax,
    #                                        reg_lambda=lambda_cv_vals)
    # # Shape of r, p, mse:  (trial, lambda, channel)
    # # Shape of pred: list (1 elem for each trial) of arrays (lambda, ???, channel)
    # # model.shape : (trial, lambda, lag, target (?), feature)
    # best_chan = np.unravel_index(np.argmax(r), r.shape)
    # plt.subplot(2, 2, 1)
    # plt.title('CV accuracy')
    # plt.errorbar(lambda_cv_exp,
    #              np.mean(r, axis=(0, 2)),
    #              np.mean(np.std(r, axis=0), axis=-1) / np.sqrt(x.shape[0]))
    # plt.xlabel('Regularization ($10 ^ \lambda$)')
    # plt.ylabel('Correlation')
    # plt.subplt(2, 2, 2)
    # plt.errorbar(lambda_cv_exp,
    #              np.mean(mse, axis=(0, 2)),
    #              np.mean(np.std(mse, axis=0), axis=-1) / np.sqrt(x.shape[0]))

    # # Compute the TRF on each trial, then average across trials.
    # trf_epochs = []
    # for i_epoch in range(len(epochs)):
    #     x_meg = x[i_epoch, meg_chan_inx, :].T
    #     x_eog = x[i_epoch, eog_chan_inx, :].T
    #     w, t, i = mtrf_train(stim=x_eog, # Array (time, param)
    #                          resp=x_meg, # Array (time, chan)
    #                          fs=epochs.info['sfreq'], # Sampling freq
    #                          mapping_direction=1, # Forward model
    #                          tmin=(tmin * 1000), # Min time in ms
    #                          tmax=(tmax * 1000), # Max time in ms
    #                          reg_lambda=1e0)
    #     trf_epochs.append(w)
    # trf_arr = np.stack(trf_epochs)
    # trf_data = np.mean(trf_arr, axis=0)

    # Compute TRF on all trials concatenated together.
    # The results of this are almost exactly the same as the results for the
    # trial-wise analysis, but it runs much faster, and the magnitude of the
    # TRF is much higher.
    x_conc = np.concatenate([trial for trial in x], axis=-1)
    x_meg = x_conc[meg_chan_inx, :].T
    x_eog = x_conc[eog_chan_inx, :].T
    w, t, i = mtrf_train(stim=x_eog, # Array (time, param)
                            resp=x_meg, # Array (time, chan)
                            fs=epochs.info['sfreq'], # Sampling freq
                            mapping_direction=1, # Forward model
                            tmin=(tmin * 1000), # Min time in ms
                            tmax=(tmax * 1000), # Max time in ms
                            reg_lambda=1e0)
    trf_data = w

    # Combine across HEOG and VEOG
    trf_x = np.sqrt((trf_data[0,...] ** 2) + (trf_data[1,...] ** 2))
    trf_x = trf_data[0,...]
    
    # Convert the TRF to MNE format
    trf_info = epochs.info.copy()
    trf_info = mne.pick_info(trf_info,
                             mne.pick_types(epochs.info, meg=True, eog=False))
    trf_evoked = mne.EvokedArray(trf_x.T, info=trf_info, tmin=tmin, comment='TRF')
    trf_evoked.plot(spatial_colors=True)

    return trf_evoked


def trf_eyelink(n):
    """ Compute the TRF based on Eyelink events
        (Not working yet)
    """
    # Load the data
    d = load_data.load_data(n)
    raw = d['raw']
    raw.load_data()

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
    # Only keep events that are plausible saccades
    events = events[fix['saccade']] 
    fix = fix.loc[fix['saccade']]
    # Only keep saccade events that fall within the recording
    within_rec = events[:,0] < raw.times.size
    events = events[within_rec, :]
    fix = fix.loc[within_rec]
    
    # Only keep some channels
    raw.pick_types(meg=True, eeg=False, eog=False, stim=False, exclude='bads')

    # Reject artifacts using ICA
    d['ica'].apply(raw) 

    # LP filter
    raw.filter(l_freq=1, h_freq=40, # band-pass filter 
               method='fir', phase='minimum', # causal filter
               n_jobs=5)

    # Build the saccade timecourse to fit with the model
    saccade_onsets = np.zeros(raw.times.size) 
    saccade_onsets[events[:,0]] = 1

    # Add the saccades as a channel and downsample
    y = np.reshape(saccade_onsets, [1, -1])
    y_info = mne.create_info(['y'], raw.info['sfreq'], ['stim'])
    y_raw = mne.io.RawArray(y, y_info)
    raw.add_channels([y_raw], force_update_info=True)
    # Resample after epoching to make sure trigger times are correct
    raw.resample(100, n_jobs=5)

    # FIXME Replace artifact segments with NaNs
    # Alternatively, just delete triggers that appear in a trial marked as bad

    # Get the data
    x = raw.get_data().T
    y = np.reshape(x[:, -1], [-1, 1])
    x = x[:, :-1]

    # TODO figure out how to do multivariate predictors
    #   saccade direction
    #   saccade distance
    #   x vs y direction separately
    #   saccade angle



    # Compute the TFR
    tmin = -0.5
    tmax = 0.5
    w, t, i = mtrf_train(stim=y,
                         resp=x, # array (time, chan)
                         fs=raw.info['sfreq'], # Sampling freq
                         mapping_direction=1, # Forward model
                         tmin=(tmin * 1000), # Min time in ms
                         tmax=(tmax * 1000), # Max time in ms
                         reg_lambda=5e3)
 

    # Convert the TRF to MNE format
    trf_info = raw.info.copy()
    trf_info = mne.pick_info(trf_info, np.arange(len(raw.ch_names) - 1))
    trf = mne.EvokedArray(w[0,:,:].T, info=trf_info, tmin=tmin, comment='TRF')


def trf_simulation():
    """ Convolve random noise with a kernel, and reconstruct the kernel.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pymtrf.mtrf import mtrf_train

    plt.ion()

    n = int(1e3)
    fs = 100.0
    x = np.random.normal(size=n)
    kernel = np.zeros(41)
    midpoint = int(kernel.size / 2)
    kernel[midpoint-5:midpoint+5] = 1
    y = np.convolve(x, kernel, mode='same')
    y = y + np.random.normal(size=n)

    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(x, label='x')
    plt.plot(y, label='x * kernel')
    plt.legend()

    plt.subplot(2, 1, 2)
    w, t, i = mtrf_train(stim=np.reshape(x, [-1, 1]),
                        resp=np.reshape(y, [-1, 1]), # array (time, chan)
                        fs=fs, # Sampling freq
                        mapping_direction=1, # Forward model
                        tmin=-200, # Min time in ms
                        tmax=200, # Max time in ms
                        reg_lambda=1e0)
    plt.plot(t, np.squeeze(w), label='TRF')
    plt.plot(t, kernel, label='Kernel')
    plt.legend()
