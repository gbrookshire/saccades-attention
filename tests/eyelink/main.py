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

# import pyglet # On Ubuntu, this has to be imported first
#import refcheck # To check the screen refresh rate
from psychopy import parallel
from psychopy import visual, core, data, event, monitors
from . import eye_wrapper

############
# Settings #
############

SCREEN_RES = [1920, 1080]

# Set to False for testing without triggers, eye-tracker, etc
_IN_MEG_LAB = True

# Clocks
START_TIME = datetime.datetime.now().strftime('%Y-%m-%d-%H%M')
RT_CLOCK = core.Clock() # for measuring response times


LOG_DIR = '../logfiles/'
assert os.path.exists(LOG_DIR)

# # Load instructions
# with open('instruct.txt') as f:
#     instruct_text = f.readlines()

# Parallel port triggers - Send triggers on individual channels
TRIGGERS = {'stim': 3,
            'response': 5}

COLORS = {'cs': 'rgb255', # ColorSpace
          'white': [255, 255, 255],
          'grey': [128, 128, 128],
          'black': [0, 0, 0],
          'pink': [225, 10, 130],
          'blue': [35, 170, 230]}

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

# Initialize externals
if _IN_MEG_LAB:
    port = parallel.ParallelPort(address=0xBFF8)
    port.setData(0)
    el = eye_wrapper.SimpleEyelink(SCREEN_RES)
    el.startup()
    # core.wait(6)
    refresh_rate = 120.0

    def eye_pos():
        """ Get the eye position
        """
        pos = el.el.getNewestSample()
        pos = pos.getRightEye()
        pos = pos.getGaze() # Get eye position in pix (origin: bottom right)
        #pos = list(pos)
        #pos[0] -= SCREEN_RES[0]/2
        #pos[1] = (SCREEN_RES[1]/2) - pos[1]
        return pos
else:
    el = eye_wrapper.DummyEyelink()
    refresh_rate = 60.0
    def eye_pos():
        pos = eye_marker.pos + np.random.randint(-2, 3, size=2)
        return pos



######################
# Window and Stimuli #
######################

# win_size = (800, 800) # Pixels
win_size = SCREEN_RES
win_center = (0, 0)

win = visual.Window(win_size, monitor='personal-laptop',
                    fullscr=False,
                    color=COLORS['grey'], colorSpace=COLORS['cs'],
                    allowGUI=False, units='deg')

# parameters used across stimuli
stim_params = {'win': win, 'units': 'pix'}
circle_params = {'fillColor': COLORS['white'],
                 'lineColor': COLORS['white'],
                 'fillColorSpace': COLORS['cs'],
                 'lineColorSpace': COLORS['cs'],
                 **stim_params}

text_stim = visual.TextStim(pos=win_center, # For instructions
                            color=COLORS['white'], colorSpace=COLORS['cs'],
                            height=32, **stim_params)

fixation = visual.Circle(radius=10, pos=win_center, **circle_params)

eye_marker = visual.Circle(radius=20, pos=win_center, **circle_params)
eye_marker.fillColor = COLORS['pink']

eye_testers = []
for xpos in [-300, 0, 300]:
    for ypos in [-300, 0, 300]:
        s = visual.Circle(radius=20,
                          pos=[xpos, ypos],
                          **circle_params)
        eye_testers.append(s)

##################################
# Test eye-tracking and updating #
##################################

"""
# new_sample = EYELINK.getFloatData()
new_sample = el.el.getFloatData() # I think this will work?
r_eye = new_sample.getRightEye() # Get the sample info for the right eye
r_pos = r_eye.getGaze() # gaze position in pixel coordinates
r_pos = r_eye.getHREF() # gaze position in HREF angular coordinates
"""


def drift_correct():
    """ Eye-tracker drift correction
    """
    print('Performing drift correction')
    if _IN_MEG_LAB:
        # Draw the fixation dot
        fixation.draw()
        win.flip()

        # Maybe only this part is necessary?
        fix_pos = origin_psychopy2propixx(fixation.pos)
        el.el.doDriftCorrect(int(fix_pos[0]), int(fix_pos[1]), 0, 0)

        # Maybe this is the part that's necessary?
        # el.el.startDriftCorrect(fix_pos[0], fix_pos[1])
        # event.waitKeys(['enter'])
        # el.el.acceptTrigger()
        # print(el.el.getCalibrationMessage())
        # el.el.exitCalibration()
        el.el.applyDriftCorrect()
        #el.el.doTrackerSetup() # ??

        # Re-start the recording after drift correction
        error = el.el.startRecording(1,1,1,1) # startData()?
        print(error)

    else:
        print('Not using EyeLink!')


def eye_pos_check():
    """ Check whether the stimulus is following the eye position
    """
    KEYS = {'break': 'escape',
            'accept': 'space'}
    event.clearEvents()
    while True:
        # Check for control keypreses
        resps = event.getKeys(KEYS.values())
        if KEYS['break'] in resps:
            break
        elif KEYS['accept'] in resps:
            drift_correct()

        #pos = list(eye_pos())
        #pos[0] -= SCREEN_RES[0]/2
        # pos[1] -= SCREEN_RES[1]/2
        #pos[1] = (SCREEN_RES[1]/2) - pos[1]
        # print(type(pos))
        # print(pos)
        pos = eye_pos()
        pos = origin_propixx2psychopy(pos)
        eye_marker.pos = pos # Mark the current fixation

        # Draw the stimuli
        for s in eye_testers:
            s.draw()
        eye_marker.draw()
        win.flip()


def run_exp():
    eye_pos_check()
