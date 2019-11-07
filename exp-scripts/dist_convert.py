"""
Convert between pixels and degrees in MEG
"""

import numpy as np

# Distances (cm)
eye_screen = 147.5
screen_width = 70.3
screen_height = 39.5

# Screen resolution
screen_res = (1920, 1080)

def origin_eyelink2psychopy(pos):
    """ Convert coordinates for shifting the origin from the
        bottom right of the screen, with Y increasing upward,
        to the center of the screen, with Y increasing downward.
    """
    x,y = pos
    x -= screen_res[0] / 2
    y = (screen_res[1] / 2) - pos[1]
    return [x, y]


def origin_psychopy2eyelink(pos):
    """ Convert in the opposite direction.
    """
    x,y = pos
    x += screen_res[0] / 2
    y = (screen_res[1] / 2) - pos[1]
    return [x, y]


def pix2cm(x):
    return x * screen_width / screen_res[0]

def cm2pix(x):
    return x * screen_res[0] / screen_width

def deg2cm(theta):
    return eye_screen * np.tan(np.deg2rad(theta))

def cm2deg(x):
    return np.rad2deg(np.arctan(x / eye_screen))

def pix2deg(x):
    return cm2deg(pix2cm(x))

def deg2pix(theta):
    return cm2pix(deg2cm(theta))

