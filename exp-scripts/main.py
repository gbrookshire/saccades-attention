"""
Attention and preprocessing before a saccade
Fall 2019
G.Brookshire@bham.ac.uk
"""

# TODO
# - Show 6 stimuli over and over (and over (and over))
# - What memory test?
#   - Change 2 items at a time, per screen update?
#   - Perceptual manipulation of images? (e.g. blank out an eye in photoshop)
#   - Location test?
#   - Multiple items with a familiarity test?

import os
import datetime
import numpy as np
import random

# On Ubuntu, pyglet has to be imported first for psychopy to work
from sys import platform
if platform == 'linux':
    print('Running on the Ubuntu desktop...')
    import pyglet 

#import refcheck # To check the screen refresh rate
from psychopy import parallel
from psychopy import visual, core, data, event, monitors

import eye_wrapper
from lattice import lattice

############
# Settings #
############

# Set to False for testing without triggers, eye-tracker, etc
_IN_MEG_LAB = False


START_TIME = datetime.datetime.now().strftime('%Y-%m-%d-%H%M')
RT_CLOCK = core.Clock() # for measuring response times 

LOG_DIR = '../logfiles/'
assert os.path.exists(LOG_DIR)

# Load instructions
with open('instruct.txt') as f:
    instruct_text = f.readlines()

TRIGGERS = {'clear': 0,
            'fixation': 1,
            'explore': 3,
            'mem_test': 3,
            'response': 4,
            'drift_correct_start': 10,
            'drift_correct_end': 11}

KEYS = {'break': 'escape',
        'accept': 'space'}

COLORS = {'cs': 'rgb255', # ColorSpace
          'white': [255, 255, 255],
          'grey': [128, 128, 128],
          'black': [0, 0, 0],
          'pink': [225, 10, 130],
          'blue': [35, 170, 230]}

SCREEN_RES = [1920, 1080]
STIM_SIZE = 100 # Size in pixels
STIM_DIST = 150 # Min distance between images
EXPLORE_DUR = 5.0 # Explore stim screen for X seconds
N_TRIALS = 3# 300

# Initialize externals
if _IN_MEG_LAB:
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
        pos = pos.getGaze() # Get eye position in pix (origin: bottom right)
        return pos

    def send_trigger(trig):
        """ Send triggers to the MEG acquisition computer and the EyeLink computer.
        """
        port.setData(trig, 1) #TODO test with setData (as opposed to setPin)
        el.trigger(trig)

else:
    refresh_rate = 60.0
    el = eye_wrapper.DummyEyelink()

    def eye_pos():
        pos = eye_marker.pos + np.random.randint(-2, 3, size=2)
        return pos

    def send_trigger(trig):
        print('Trigger: {}'.format(trig))


######################
# Window and Stimuli #
######################

# win_size = (800, 800) # Pixels
win_size = SCREEN_RES
win_center = (0, 0)

win = visual.Window(win_size,
                    monitor='MEG_LAB_MONITOR', ## FIXME
                    fullscr=False,
                    color=COLORS['grey'], colorSpace=COLORS['cs'],
                    allowGUI=False)

# parameters used across stimuli
stim_params = {'win': win, 'units': 'pix'}
circle_params = {'fillColor': COLORS['white'],
                 'lineColor': COLORS['white'],
                 'fillColorSpace': COLORS['cs'],
                 'lineColorSpace': COLORS['cs'],
                 **stim_params}

fixation = visual.Circle(radius=10, pos=win_center, **circle_params)

# text_stim = visual.TextStim(pos=win_center, text='--', # For instructions
#                             color=COLORS['white'], colorSpace=COLORS['cs'],
#                             height=32, **stim_params) 

mem_probe_ring = visual.Circle(radius=STIM_SIZE/2, **circle_params)
# mem_probe = visual.TextStim(text='?',
#                            bold=True, height=STIM_SIZE*3/4,
#                            color=COLORS['white'],
#                            colorSpace=COLORS['cs'],
#                            **stim_params)

# eye_marker = visual.Circle(radius=20, pos=win_center, **circle_params)
# eye_marker.fillColor = COLORS['pink']

n_stimuli = 6
pic_stims = []
for n in range(n_stimuli):
    fname = '../stimuli/{}.jpg'.format(n)
    s = visual.ImageStim(
            image=fname,
            size=STIM_SIZE,
            name = str(n),
            **stim_params)
    pic_stims.append(s) 

# Positions of the stimuli
stim_locs = lattice(center=win_center, n_fold=6, radius=STIM_DIST, n_layers=2)

# Make the trials
choice = np.random.choice
trial_info = []
for n in range(N_TRIALS):
    d = {}
    d['fix_dur'] = np.random.uniform(1.0, 2.0)
    d['locs'] = choice(len(stim_locs), size=n_stimuli, replace=False)
    d['mem_target'] = choice(n_stimuli)
    nontarget_stimuli = [e for e in range(n_stimuli) if e != d['mem_target']]
    d['mem_distractor'] = choice(nontarget_stimuli)
    d['mem_loc'] = d['locs'][d['mem_target']] 
    d['mem_target_loc'] = choice(['right', 'left'])
    trial_info.append(d)

trials = data.TrialHandler(trial_info, nReps=1, method='sequential')
print(trial_info)


##################################
# Test eye-tracking and updating #
##################################

def origin_propixx2psychopy(pos):
    """ Convert coordinates for shifting the origin from the
        bottom right of the screen, with Y increasing upward,
        to the center of the screen, with Y increasing downward.
    """
    x,y = pos
    x -= SCREEN_RES[0] / 2
    y = (SCREEN_RES[1] / 2) - pos[1]
    return [x, y]


def origin_psychopy2propixx(pos):
    """ Convert in the opposite direction.
    """
    x,y = pos
    x += SCREEN_RES[0] / 2
    y = (SCREEN_RES[1] / 2) - pos[1]
    return [x, y]


def show_text(text):
    """ Show text at the center of the screen.
    """
    text_stim.set('text', text)
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
    send_trigger(TRIGGERS['drift_correct_start'])
    # Draw the fixation dot
    fixation.draw()
    win.flip() 
    # Do the drift correction
    fix_pos = np.int64(origin_psychopy2propixx(fixation.pos))
    el.drift_correct(fix_pos)
    send_trigger(TRIGGERS['drift_correct_end'])


def run_trial(trial):
    #send_trigger(TRIGGERS['clear']) 
    # Run the drift correction if necessary
    resps = event.getKeys(KEYS.values())
    if KEYS['accept'] in resps:
        drift_correct()

    # Variable fixation
    fixation.draw()
    win.flip()
    send_trigger(TRIGGERS['fixation'])
    core.wait(trial['fix_dur'])
    #TODO Wait for fixation

    exploration_screen(trial)
    show_mask_stimuli()
    show_memory_trial(trial)


def exploration_screen(trial):
    """ Show all 6 stimuli at their respective locations
    """
    for n in range(len(pic_stims)):
        pos = stim_locs[trial['locs'][n]]
        pic_stims[n].pos = pos
        pic_stims[n].draw()
    win.flip()
    send_trigger(TRIGGERS['explore'])
    core.wait(EXPLORE_DUR)
    win.flip(clearBuffer=True)


def show_mask_stimuli():
    pass


def show_memory_trial(trial):
    """ Show a probe at one location, and a 2AFC for which stim was there
    """
    left_pos = [-500, -300]
    right_pos = [500, -300]

    # Re-draw the memory probe
    #for s in (mem_probe, mem_probe_ring):
    for s in (mem_probe_ring,):
        s.pos = stim_locs[trial['mem_loc']]
        s.draw()

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
    send_trigger(TRIGGERS['mem_test'])
    
    # Wait for a key press
    r = event.waitKeys() #FIXME
    send_trigger(TRIGGERS['response'])


def eye_pos_check():
    """ Check whether the stimulus is following the eye position
    """
    event.clearEvents()
    while True:
        # Check for control keypreses
        resps = event.getKeys(KEYS.values())
        if KEYS['break'] in resps:
            break
        elif KEYS['accept'] in resps:
            drift_correct()

        pos = eye_pos()
        pos = origin_propixx2psychopy(pos)
        eye_marker.pos = pos # Mark the current fixation

        # Draw the stimuli
        for s in eye_testers:
            s.draw()
        eye_marker.draw()
        win.flip()


def run_exp():
    for trial in trials:
        run_trial(trial)
