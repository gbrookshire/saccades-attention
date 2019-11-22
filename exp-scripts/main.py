"""
Attention and preprocessing before a saccade
Fall 2019
G.Brookshire@bham.ac.uk

Design
- Show the same 6 stimuli in random locations on every trial
- At the end of each trial, probe memory
    - Ask which of 2 stimuli was present at a marked location
"""

# TODO
"""
Features to add
- Response with 2 fingers on the right hand (not 2 hands)

Tests
- Check all TODO tags
- Get a couple behavioral pilots
    - Is the task too hard?
    - Do people actually look at all the items?
        - Do we need to add a perceptual task
- MEG pilot: Test whether stimuli are discriminable using classifiers

"""

# Standard libraries
import os
import datetime
import numpy as np
import random

# Psychopy
from psychopy import parallel
from psychopy import visual, core, data, event, monitors

# Custom modules
import refcheck
import dist_convert as dc
import eye_wrapper
from lattice import lattice


############
# Settings #
############

# Set to False for testing without triggers, eye-tracker, etc
IN_MEG_LAB = True

START_TIME = datetime.datetime.now().strftime('%Y-%m-%d-%H%M')
RT_CLOCK = core.Clock() # for measuring response times

TRIGGERS = {'response': 1,
            'fixation': 2,
            'explore': 4,
            'mem_test': 8,
            'drift_correct_start': 16,
            'drift_correct_end': 32}

KEYS = {'break': 'escape',
        'drift': 'return',
        'accept': 'space',
        'left': '4',
        'right': '7'}

COLORS = {'cs': 'rgb255', # ColorSpace
          'white': [255, 255, 255],
          'grey': [128, 128, 128],
          'black': [0, 0, 0],
          'pink': [225, 10, 130],
          'blue': [35, 170, 230]}

FULL_SCREEN = True
if FULL_SCREEN:
    SCREEN_RES = [1920, 1080] # Full res on the Propixx projector
else:
    SCREEN_RES = [1000, 1000]
STIM_SIZE_DEG = 2.0
STIM_DIST_DEG = 3.0
STIM_SIZE = int(dc.deg2pix(STIM_SIZE_DEG)) # Size in pixels
STIM_DIST = int(dc.deg2pix(STIM_DIST_DEG)) # Min distance between images
EXPLORE_DUR = 5.0 # Explore stim screen for X seconds
RESPONSE_CUTOFF = 5.0 # Respond within this time
FIX_DUR = 0.5 # Hold fixation for X seconds before starting trial
FIX_THRESH_DEG = 1.0 # Subject must fixate w/in this distance to start trial
FIX_THRESH = int(dc.deg2pix(FIX_THRESH_DEG))
N_TRIALS = 300 # Should be about 45-50 min
REC_LENGTH = 15 * 60 # Length of each MEG recording (in seconds)

END_EXPERIMENT = 9999 # Numeric tag signals stopping expt early

LOG_DIR = '../logfiles/'
assert os.path.exists(LOG_DIR)

# Load instructions
with open('instruct.txt') as f:
    instruct_text = f.readlines()

# Initialize external equipment
if IN_MEG_LAB:
    refresh_rate = 120.0
    # Initialize parallel port
    port = parallel.ParallelPort(address=0xBFF8)
    port.setData(0)
    # Initialize eye-tracker
    el = eye_wrapper.SimpleEyelink(SCREEN_RES)
    el.startup()

    def eye_pos():
        """ Get the eye position
        """
        pos = el.el.getNewestSample()
        pos = pos.getRightEye()
        pos = pos.getGaze() # eye position in pix (origin: bottom right)
        return pos

    def send_trigger(trig):
        """ Send triggers to the MEG acquisition computer
        and the EyeLink computer.
        """
        t = TRIGGERS[trig]
        port.setData(t)
        el.trigger(t)

    def reset_port():
        """ Reset the parallel port to avoid overlapping triggers
        """
        wait_time = 0.003
        core.wait(wait_time)
        port.setData(0)
        core.wait(wait_time)

else:
    # Dummy functions for dry-runs on my office desktop
    refresh_rate = 60.0
    el = eye_wrapper.DummyEyelink()

    def eye_pos():
        pos = win_center
        return pos

    def send_trigger(trig):
        print('Trigger: {}'.format(trig))

    def reset_port():
        pass


######################
# Window and Stimuli #
######################

win_size = SCREEN_RES
win_center = (0, 0)

win = visual.Window(win_size,
                    monitor='propixxMonitor',
                    fullscr=FULL_SCREEN,
                    color=COLORS['grey'], colorSpace=COLORS['cs'],
                    allowGUI=False)

# parameters used across stimuli
stim_params = {'win': win, 'units': 'pix'}
circle_params = {'fillColor': COLORS['white'],
                 'lineColor': COLORS['white'],
                 'fillColorSpace': COLORS['cs'],
                 'lineColorSpace': COLORS['cs'],
                 **stim_params}

text_stim = visual.TextStim(pos=win_center, text='hello', # For instructions
                            color=COLORS['white'], colorSpace=COLORS['cs'],
                            height=32,
                            **stim_params)

fixation = visual.Circle(radius=10, pos=win_center, **circle_params)
drift_fixation = visual.Circle(radius=5, pos=win_center, **circle_params)
mem_probe = visual.Circle(radius=STIM_SIZE/3, **circle_params)

n_stimuli = 6
pic_stims = []
for n in range(n_stimuli):
    fname = '../stimuli/cropped/{}.jpg'.format(n)
    s = visual.ImageStim(
            image=fname,
            size=STIM_SIZE,
            name = str(n),
            **stim_params)
    pic_stims.append(s)

# Positions of the picture stimuli
stim_locs = lattice(center=win_center, n_fold=6, radius=STIM_DIST, n_layers=2)

# For marking the gaze position
eye_marker = visual.Circle(radius=20, pos=win_center, **circle_params)
eye_marker.fillColor = COLORS['pink']


###################
# Make the trials #
###################

choice = np.random.choice
trial_info = []
for n in range(N_TRIALS):
    d = {}
    stim_locations = choice(len(stim_locs), size=n_stimuli, replace=False)
    for i_stim in range(n_stimuli):
        tag = 'loc_{}'.format(i_stim)
        d[tag] = stim_locations[i_stim]
    # Only show four stimuli on each trial
    d['display_stims'] = choice(range(n_stimuli), size=4, replace=False)
    d['mem_target'] = choice(d['display_stims'])
    nontarget_stimuli = [e for e in d['display_stims'] if e != d['mem_target']]
    d['mem_distractor'] = choice(nontarget_stimuli)
    mem_loc_tag = 'loc_{}'.format(d['mem_target'])
    d['mem_loc'] = d[mem_loc_tag]
    d['mem_target_loc'] = choice(['right', 'left'])
    trial_info.append(d)

trials = data.TrialHandler(trial_info, nReps=1, method='sequential')


#########################
# Stimulus presentation #
#########################

def euc_dist(a, b):
    """ Euclidean distance between two (x,y) pairs
    """
    d = sum([(x1 - x2)**2 for x1,x2 in zip(a, b)]) ** (1/2)
    return d


def show_lattice():
    """ Show all the positions of the lattice for testing stim positions
    """
    lattice_stims = []
    for p in stim_locs:
        s = visual.Circle(radius=STIM_SIZE/2, pos=p, **circle_params)
        lattice_stims.append(s)
    for s in lattice_stims:
        s.draw()
    win.flip()
    event.waitKeys(keyList=['escape'])


def show_text(text):
    """ Show text at the center of the screen
    """
    text_stim.text = text
    text_stim.draw()
    win.flip()


def instructions(text):
    """ Show instructions and go on after pressing space
    """
    show_text(text)
    event.waitKeys(keyList=['space'])
    win.flip(clearBuffer=True) # clear the screen
    core.wait(0.2)


def drift_correct():
    """ Eye-tracker drift correction.
    Press SPACE on the Eyelink machine to accept the current position.
    """
    reset_port()
    core.wait(0.2)
    # Draw a fixation dot
    drift_fixation.draw() # TODO test this
    win.flip()
    send_trigger('drift_correct_start')
    reset_port()
    # Do the drift correction
    fix_pos = np.int64(dc.origin_psychopy2eyelink(drift_fixation.pos))
    el.drift_correct(fix_pos)
    send_trigger('drift_correct_end')
    reset_port()


def experimenter_control():
    """ Check for experimenter key-presses to pause/exit the experiment or
    correct drift in the eye-tracker.
    """
    r = event.getKeys(KEYS.values())
    if KEYS['break'] in r:
        show_text('End experiment? (y/n)')
        core.wait(1.0)
        event.clearEvents()
        r = event.waitKeys(keyList=['y', 'n'])
        if 'y' in r:
            return END_EXPERIMENT
    elif KEYS['drift'] in r:
        drift_correct()


def run_trial(trial):
    reset_port()
    event.clearEvents()

    # Wait for fixation and check for experimenter input
    fixation.draw()
    win.flip()
    send_trigger('fixation')
    reset_port()
    t_fix = core.monotonicClock.getTime() # Start a timer
    core.wait(0.2)

    while True:
        # Check for experimenter control to end the expt early
        if experimenter_control() == END_EXPERIMENT:
            return END_EXPERIMENT

        d = euc_dist(dc.origin_eyelink2psychopy(eye_pos()), win_center)
        # Reset timer if not looking at fixation
        if (d > FIX_THRESH):
            t_fix = core.monotonicClock.getTime()
        # If they are looking at the fixation, have they looked long enough?
        elif (core.monotonicClock.getTime() - t_fix) > FIX_DUR:
            break
        # If they are looking, but haven't held fixation long enough
        else:
            fixation.draw()
            win.flip()

    # Present the trial
    exploration_screen(trial)
    show_memory_trial(trial)

    return experimenter_control()


def exploration_screen(trial):
    """ Show all 6 stimuli at their respective locations
    """
    for n in range(len(pic_stims)):
        if n in trial['display_stims']:
            pos = stim_locs[trial['loc_{}'.format(n)]]
            pic_stims[n].pos = pos
            pic_stims[n].draw()
    win.flip()
    send_trigger('explore')
    reset_port()
    core.wait(EXPLORE_DUR)
    win.flip(clearBuffer=True)
    core.wait(0.5)


def show_memory_trial(trial):
    """ Show a probe at one location, and a 2AFC for which stim was there
    """
    # Positions of the two memory choices
    left_pos = [-700, -450]
    right_pos = [700, -450]

    # Draw the memory probe
    mem_probe.pos = stim_locs[trial['mem_loc']]
    mem_probe.draw()

    # Draw the two memory choices
    if trial['mem_target_loc'] == 'left':
        target_pos = left_pos
        distractor_pos = right_pos
    else:
        target_pos = right_pos
        distractor_pos = left_pos
    targ_stim = pic_stims[trial['mem_target']]
    dist_stim = pic_stims[trial['mem_distractor']]
    targ_stim.pos = target_pos
    dist_stim.pos = distractor_pos
    targ_stim.draw()
    dist_stim.draw()

    win.flip()
    send_trigger('mem_test')
    RT_CLOCK.reset()
    reset_port()
    core.wait(0.2)

    # Wait for a key press
    event.clearEvents()
    r = event.waitKeys(maxWait=RESPONSE_CUTOFF,
                       keyList=[KEYS['left'], KEYS['right']],
                       timeStamped=RT_CLOCK)
    if r is None:
        feedback_text = 'Too slow'
    else:
        send_trigger('response')
        reset_port()
        keypress, rt = r[0] #TODO Make sure responses are logging correctly
        trials.addData('resp', keypress)
        trials.addData('rt', rt)
        # Feedback # TODO Check whether this works
        # Check if answer was correct
        if (trial['mem_target_loc'] == 'right') & (keypress == KEYS['right']):
            feedback_text = 'Correct'
        elif (trial['mem_target_loc'] == 'left') & (keypress == KEYS['left']):
            feedback_text = 'Correct'
        else:
            feedback_text = 'Incorrect'

    show_text(feedback_text)
    core.wait(0.5) 


def eye_pos_check():
    """ Check whether the stimulus is following the eye position
    """
    event.clearEvents()
    while True:
        # Check for control keypreses
        r = event.getKeys(KEYS.values())
        if KEYS['break'] in r:
            break
        elif KEYS['accept'] in r:
            drift_correct()

        pos = eye_pos()
        pos = dc.origin_eyelink2psychopy(pos)
        eye_marker.pos = pos # Mark the current fixation

        # Draw the stimuli
        for s in eye_testers:
            s.draw()
        eye_marker.draw()
        win.flip()


def run_exp():
    """ Coordinate the different parts of the experiment
    """

    # A few tests before beginning the experiment
    refcheck.check_refresh_rate(win, refresh_rate)
    # show_lattice()
    # eye_pos_check()

    # Instructions
    for line in instruct_text:
        instructions(line)

    # Run the trials
    rec_start_time = core.monotonicClock.getTime()
    rec_number = 1 # Which recording number is this?
    for trial in trials:
        status = run_trial(trial)
        trials.addData('rec_number', rec_number) # TODO check this
        if status == END_EXPERIMENT:
            break
        # TODO check whether this works
        # Prompt to start a new MEG recording every so often
        curr_time = core.monotonicClock.getTime()
        if curr_time > (rec_start_time + REC_LENGTH):
            show_text('--- Start new MEG recording ---')
            event.waitKeys(keyList=['return'], maxWait=9999)
            show_text('Ready?')
            event.waitKeys(keyList=['return'], maxWait=9999)
            drift_correct() 

    # Save the data
    fname = '{}/{}.csv'.format(LOG_DIR, START_TIME)
    trials.saveAsWideText(fname, encoding='ASCII',
                          delim=',', fileCollisionMethod='rename')

    show_text('That was it -- thanks!')
    event.waitKeys(keyList=['escape'], maxWait=30)

    # Close everything down
    win.close()
    if IN_MEG_LAB:
        el.shutdown()
    core.quit()

