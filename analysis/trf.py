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
import scipy
import matplotlib.pyplot as plt
import mne
from mne.externals.h5io import write_hdf5, read_hdf5
from pymtrf.mtrf import mtrf_predict, mtrf_crossval, mtrf_train, lag_gen
import load_data

plt.ion()

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


def trf_eog(n):
    """ Compute average TRF based on EOG for each trial (with multiple fixations).
    No need to take spatial derivative -- the TRF already accounts for
    autocorrelation in the EOG signals.

    TODO
    - absolute value of derivative of EOG
        - instead of modeling direction, just model saccade onsets

    """

    d = load_data.load_data(n)
    raw = d['raw']
    raw.load_data()
    
    # Only keep some channels
    raw.pick_types(meg=True, eog=True, exclude='bads')
    
    # # Reject artifacts using ICA
    # # !!! This obscures eye movements and changes the topographies, making it
    # harder to tell what activity is directly related to moving the eyes.
    # d['ica'].apply(raw) 
    
    # LP filter MEG and EOG
    raw.filter(l_freq=None, h_freq=40,
               picks=['meg', 'eog'],
               n_jobs=5,
               method='fir', phase='minimum') # causal filter
    
    # Segment into epochs
    epochs = mne.Epochs(raw,
                        events=d['meg_events'],
                        event_id=expt_info['event_dict']['explore'],
                        tmin=0.5,
                        tmax=4.5,
                        baseline=None,
                        detrend=None,
                        reject_by_annotation=True, # Reject trials with arts
                        preload=True,
                        proj=True)
    raw.close()
    
    # Resample after epoching to make sure trigger times are correct
    epochs.resample(100, n_jobs=5)
    
    # Get the data. (epoch, channel, time)
    x = epochs.get_data()
    tmin = -0.5
    tmax = 0.5
    eog_chan_inx = mne.pick_types(epochs.info, meg=False, eog=True)
    meg_chan_inx = mne.pick_types(epochs.info, meg=True, eog=False)

    # # Run cross-validation to find the optimal lambda parameter.
    # # Cross-validate for the time that's expected to show the strongest
    # # response. In Crosse et al 2016, they average r values over channels "such
    # # that model performance would be optimized in a more global manner." But
    # # they say: "Alternatively, one could average across only channels within a
    # # specified top percentile or based on a specific location."
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
    # plt.subplot(2, 2, 1)
    # plt.title('CV accuracy')
    # plt.errorbar(lambda_cv_exp,
    #              np.mean(r, axis=(0, 2)), # Average over trials and channels
    #              np.mean(np.std(r, axis=0), axis=-1) / np.sqrt(x.shape[0]))
    # plt.xlabel('Regularization ($10 ^ \lambda$)')
    # plt.ylabel('Correlation')
    # plt.subplot(2, 2, 2)
    # plt.title('MSE')
    # plt.errorbar(lambda_cv_exp,
    #              np.mean(mse, axis=(0, 2)),
    #              np.mean(np.std(mse, axis=0), axis=-1) / np.sqrt(x.shape[0]))
    # plt.xlabel('Regularization ($10 ^ \lambda$)')
    # plt.ylabel('MSE')


    # Compute TRF on each trial, or on concatenated data? They say: "Both of
    # these approaches yield the same results because the data are modeled
    # using a linear assumption" (ignoring artifacts between trials). But I did
    # find very slight differences. Those might be due to discontinuities where
    # trials are concatenated. Also, the parameter estimates are drastically
    # different in magnitude. Cross-validation is easier on trial-wise data

    # Compute the TRF on each trial, then average across trials.
    trf_epochs = []
    for i_epoch in range(len(epochs)):
        x_meg = x[i_epoch, meg_chan_inx, :].T
        x_eog = x[i_epoch, eog_chan_inx, :].T
        w, t, i = mtrf_train(stim=x_eog, # Array (time, param)
                             resp=x_meg, # Array (time, chan)
                             fs=epochs.info['sfreq'], # Sampling freq
                             mapping_direction=1, # Forward model
                             tmin=(tmin * 1000), # Min time in ms
                             tmax=(tmax * 1000), # Max time in ms
                             reg_lambda=1e-10)
        trf_epochs.append(w)
    trf_arr = np.stack(trf_epochs)
    trf_data = np.mean(trf_arr, axis=0)

    # # Compute TRF on all trials concatenated together.
    # # The results of this are almost exactly the same as the results for the
    # # trial-wise analysis, but it runs much faster, and the magnitude of the
    # # TRF is much higher.
    # x_conc = np.concatenate([trial for trial in x], axis=-1)
    # x_meg = x_conc[meg_chan_inx, :].T
    # x_eog = x_conc[eog_chan_inx, :].T
    # w, t, i = mtrf_train(stim=x_eog, # Array (time, param)
    #                         resp=x_meg, # Array (time, chan)
    #                         fs=epochs.info['sfreq'], # Sampling freq
    #                         mapping_direction=1, # Forward model
    #                         tmin=(tmin * 1000), # Min time in ms
    #                         tmax=(tmax * 1000), # Max time in ms
    #                         reg_lambda=1e0)
    # # When computing TRFs on concatenated data, the choice of lambda doesn't
    # make a big difference. Lower values mean less regularization, so that
    # might be a safer choice.
    # trf_data = w

    # Convert the TRF to MNE format and save it
    trf_info = epochs.info.copy()
    trf_info = mne.pick_info(trf_info,
                             mne.pick_types(epochs.info, meg=True, eog=False))
    for eog_direction in (0, 1):
        trf_evoked = mne.EvokedArray(trf_data[eog_dir, ...].T,
                                     info=trf_info,
                                     tmin=tmin,
                                     comment='TRF')
        fname = f"{data_dir}trf/eog/{n}_{eog_direction}-ave.fif"
        trf_evoked.save(fname)

    # Load them with: x = mne.Evoked(f"{data_dir}trf/{n}_{eog_dir}-ave.fif")
    # Then plot with: x.plot(spatial_colors=True)


def trf_plot(n, param_number=1, analysis_type='eyelink'):
    x = mne.Evoked(f"{data_dir}trf/{analysis_type}/{n}_{param_number}-ave.fif")
    x.plot(spatial_colors=True)


def trf_avg(param_number=1, analysis_type='eyelink'):
    """ Average over TRFs"""

    save_dir = f"{data_dir}trf/{analysis_type}"
    all_trfs = [mne.Evoked(f"{save_dir}/{n}_{param_number}-ave.fif")
                    for n in (2,3,4,5,6)]
    if analysis_type == 'eog':
        # For some TRFs, the polarity is reversed. Flip them so they all have the
        # same sign.
        weights = (1, -1, -1, 1, 1)
    elif analysis_type == 'eyelink':
        weights = (1, 1, 1, 1, 1)
    avg_trf = mne.combine_evoked(all_trfs, weights=weights)
    avg_trf.plot(spatial_colors=True)


def trf_eyelink(n):
    """ TRF between MEG and eye-tracker output
    To-do
    - interpolate between blinks
    - parameter for blink onset
    - absolute value of derivative of eye movement
    """
    subject_info = pd.read_csv(data_dir + 'subject_info.csv',
                            engine='python', sep=',')
    subj_fname = subject_info['meg'][n]
    raw = mne.io.read_raw_fif(load_data.meg_filename(subj_fname))
    raw.load_data()

    eyelink_chans = ['MISC001', 'MISC002', 'MISC003']
    eyes_raw = raw.copy()
    eyes_raw.pick_channels(eyelink_chans)

    # Add a channel for blinks
    blinks_raw = identify_blinks(eyes_raw, 100)
    eyes_raw.add_channels([blinks_raw], force_update_info=True)

    # Interpolate eye position over blinks
    eyes_clean = interpolate_over_blinks(eyes_raw)

    # Combine interpolated eye channels with raw data
    raw.drop_channels(eyelink_chans)
    raw.add_channels([eyes_clean], force_update_info=True)

    # LP filter MEG
    raw.filter(l_freq=None, h_freq=40,
               picks=['meg'],
               n_jobs=5,
               method='fir', phase='minimum') # causal filter

    # Make epochs
    meg_events = mne.find_events(raw, # Segment out the MEG events
                                 stim_channel='STI101',
                                 mask=0b00111111, # Ignore Nata button triggers
                                 shortest_event=1)
    # Segment into epochs
    epochs = mne.Epochs(raw,
                        events=meg_events,
                        event_id=expt_info['event_dict']['explore'],
                        picks=['meg', 'misc'],
                        tmin=0.5,
                        tmax=4.5,
                        baseline=None,
                        detrend=None,
                        reject_by_annotation=True, # Reject trials with arts
                        preload=True,
                        proj=True)

    # Downsample the data
    epochs.resample(100, n_jobs=5)

    # Get the data. (epoch, channel, time)
    x = epochs.get_data()

    # Model blinks as an impulse response instead of a sustained offset
    blink_inx = mne.pick_channels(epochs.ch_names, ('blink',))
    x_blink = x[:,blink_inx,:]
    x_blink = (np.diff(x_blink, prepend=0) > 0.5).astype(int)
    x[:,blink_inx,:] = x_blink

    tmin = -0.5
    tmax = 0.5
    eye_chan_inx = mne.pick_channels(epochs.ch_names,
                                     ('MISC001', 'MISC002', 'blink'))
    meg_chan_inx = mne.pick_types(epochs.info, meg=True, misc=False)

    # Compute the TRF on each trial, then average across trials.
    trf_epochs = []
    for i_epoch in range(len(epochs)):
        x_meg = x[i_epoch, meg_chan_inx, :].T
        x_eye = x[i_epoch, eye_chan_inx, :].T
        w, t, i = mtrf_train(stim=x_eye, # Array (time, param)
                             resp=x_meg, # Array (time, chan)
                             fs=epochs.info['sfreq'], # Sampling freq
                             mapping_direction=1, # Forward model
                             tmin=(tmin * 1000), # Min time in ms
                             tmax=(tmax * 1000), # Max time in ms
                             reg_lambda=1e0)
        trf_epochs.append(w)
    trf_arr = np.stack(trf_epochs)
    trf_data = np.mean(trf_arr, axis=0)

    # Convert the TRF to MNE format and save it
    trf_info = epochs.info.copy()
    trf_info = mne.pick_info(trf_info,
                             mne.pick_types(epochs.info, meg=True, eog=False))
    for eog_direction in range(trf_data.shape[0]):
        trf_evoked = mne.EvokedArray(trf_data[eog_direction, ...].T,
                                     info=trf_info,
                                     tmin=tmin,
                                     comment='TRF')
        fname = f"{data_dir}trf/eyelink/{n}_{eog_direction}-ave.fif"
        trf_evoked.save(fname)


def identify_blinks(raw, blink_padding=20):
    """
    Find time-points when the subject is blinking. Outputs an mne.Raw objects
    with a channel called "blink". This channel is 0 when the subject is not
    blinking, and 1 when the subject is blinking.

    When the eye is not found, the pupil channel MISC003 < -4.5.

    Parameters
    ----------

    raw : mne.Raw
        Raw data object

    blink_padding : int
        How much padding to add on either side of each blink (in samples)


    Returns
    -------

    blinks_raw : mne.Raw
        Data structure holding a channel showing where blinks occur.
    """

    # Get the pupil data
    pupil_raw = raw.copy()
    pupil_raw.load_data()
    pupil_raw.pick_channels(['MISC003'])
    x_pupil = pupil_raw.get_data()[0,:]

    # Check for blinks by seeing when the pupil is missing.
    blink = x_pupil < -4.5

    # Find small blinks that aren't caught above
    x_pupil[blink] = np.nan
    stdize = lambda a: (a - np.nanmean(a)) / np.nanstd(a)
    almost_blink = stdize(x_pupil) < -2.5

    # Combine blinks and other anomalies
    blink = blink | almost_blink

    # This misses the beginnings/endings of blinks. Extend it to either side.
    blink_labels = scipy.ndimage.label(blink)[0]
    blink_slices = scipy.ndimage.find_objects(blink_labels)
    blink_timecourse = np.zeros(x_pupil.shape)
    for sl in blink_slices:
        padded_slice = slice(sl[0].start - blink_padding,
                             sl[0].stop + blink_padding)
        blink_timecourse[padded_slice] = 1

    # Make it into an mne.Raw object
    blink_info = mne.create_info(['blink'], raw.info['sfreq'], ['misc'])
    blink_raw = mne.io.RawArray(np.reshape(blink_timecourse, [1, -1]),
                                blink_info)
    return blink_raw


def interpolate_over_blinks(raw):
    """
    Take time-points where subjects are blinking, and interpolate eye-tracker
    channels over those points.

    Parameters
    ----------

    raw : mne.Raw
        Raw data object containing eye-tracker channels (MISC001-3) and a
        channel called 'blink' from the function `identify_blinks`.


    Returns
    -------

    raw_clean : mne.Raw
        Raw data with eye-tracker chans interpolated over blinks.
    """

    assert 'blink' in raw.ch_names, 'Input `raw` must have a channel "blink"'
    x = raw.get_data()
    blink_inx = raw.ch_names.index('blink')
    blink = x[blink_inx, :]
    blink_labels = scipy.ndimage.label(blink)
    blink_slices = scipy.ndimage.find_objects(blink_labels[0])
    x_clean = x.copy()
    for chan in ('MISC001', 'MISC002', 'MISC003'):
        try:
            chan_inx = raw.ch_names.index(chan)
        except ValueError:
            continue

        for sl in blink_slices[:-1]:
            start = sl[0].start
            stop = sl[0].stop
            connector = np.linspace(x[chan_inx, start],
                                    x[chan_inx, stop],
                                    stop - start)
            x_clean[chan_inx, sl[0]] = connector

    raw_clean = mne.io.RawArray(x_clean, raw.info, raw.first_samp)
    return raw_clean


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
