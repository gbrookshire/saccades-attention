"""
Check the screen refresh rate in Psychopy.
"""

from __future__ import division

def check_refresh_rate(win, expected, thresh=0.0001):
    """ Make sure the refresh rate is similar to the expected refresh rate.
    win: an instance of psychopy.visual.Window
    expected: the expected screen refresh rate in Hz
    thresh: the amount of difference allowed (in sec) before raising an error
    """
    fp = win.monitorFramePeriod
    print('Screen refresh rate:', round(1 / fp, 2), 'Hz')
    if abs((1 / expected) - fp) > thresh:
        msg = 'Frame period {} differs from expected {}.'
        msg = msg.format(round(fp, 4), round(1 / expected, 4))
        raise FrameRateException(msg)

class FrameRateException(Exception):
    """ Raised when the frame rate is wonky. """
    pass
