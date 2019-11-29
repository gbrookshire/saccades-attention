""" Simple wrapper around pylink to set up the eye tracker.
"""

import datetime
from time import sleep

try:
    import pylink as pl
except ImportError:
    print('pylink not found')

class DummyEyelink(object):
    """ Allows for testing scripts without using the Eyelink
    """
    def __init__(self, *args, **kwargs):
        pass
    def startup(self):
        pass
    def trigger(self, trig):
        pass
    def shutdown(self):
        pass
    def drift_correct(self, center_pos):
        sleep(5)

class SimpleEyelink(object):

    def __init__(self, screen_res):
        self.screen_res = screen_res
        self.fname = datetime.datetime.now().strftime("%y%m%d%H")

    def startup(self):
        # Open the calibration screen
        pl.flushGetkeyQueue()
        pl.openGraphics()

        # Calibrate the eyetracker
        el = pl.EyeLink()
        el.openDataFile(self.fname)

        ### vvv Parameters from Tjerk's setup file
        el.sendCommand('link_sample_data = \
                       LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT')

        # This Command is crucial to map the gaze positions from the tracker to
        # screen pixel positions to determine fixation
        el.sendCommand('screen_pixel_coords = %ld %ld %ld %ld' % \
                       tuple([0, 0] + self.screen_res))
        el.sendMessage('DISPLAY_COORDS %ld %ld %ld %ld' % \
                       tuple([0, 0] + self.screen_res))

        # Use Psychophysical setting
        el.sendCommand('recording_parse_type = GAZE')
        el.sendCommand('saccade_velocity_threshold = 22')
        el.sendCommand('saccade_acceleration_threshold = 3800')
        el.sendCommand('saccade_motion_threshold = 0.0')
        el.sendCommand('saccade_pursuit_fixup = 60')
        el.sendCommand('fixation_update_interval = 0')

        # Other tracker configurations
        el.sendCommand('heuristic_filter = 0')
        el.sendCommand('pupil_size_diameter = YES')
        #el.sendCommand('calibration_type = HV13')
        el.sendCommand('generate_default_targets = YES')
        el.sendCommand('enable_automatic_calibration = YES')
        #el.sendCommand('automatic_calibration_pacing = 1000')
        el.sendCommand('binocular_enabled = NO')
        el.sendCommand('use_ellipse_fitter = NO')
        el.sendCommand('sample_rate = 1000')
        el.sendCommand('elcl_tt_power = %d' % 2)
        el.sendCommand('file_event_filter = \
                       RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,INPUT')
        el.sendCommand('link_event_filter = \
                       RIGHT,FIXATION,FIXUPDATE,SACCADE,BLINK,MESSAGE,INPUT')
        el.sendCommand('file_sample_data = \
                       GAZE,GAZERES,HREF,PUPIL,AREA,STATUS,INPUT')
        ### ^^^ End of Tjerk's code

        el.doTrackerSetup(width=self.screen_res[0],
                          height=self.screen_res[1])
        el.enableAutoCalibration() # Go on automatically afer each fixation
        el.setCalibrationType('HV9') # show 9 targets
        el.setAutoCalibrationPacing(1000)

        # Close the calibration screen
        pl.closeGraphics()

        # Start getting data from the eye-tracker
        el.open()
        el.flushKeybuttons(0)
        el.startData(15, 1) # All event types
        el.startRecording(1,1,1,1) # Record eyetracker data
        el.waitForData(5000, 1, 0) # Wait up to 5 sec for data
        el.sendMessage('SYNCTIME') # From Tjerk's matlab code

        self.el = el # Hold onto the object

    def drift_correct(self, center_pos):
        """ Drift correction with a manually-drawn fixation
        Press ENTER on the Eyelink computer to accept the new fixation
        """
        self.el.doDriftCorrect(center_pos[0], center_pos[1], 0, 0)
        self.el.applyDriftCorrect()
        error = self.el.startRecording(1,1,1,1)
        return error

    def trigger(self, trig):
        self.el.sendMessage('Trigger %d' % trig)

    def shutdown(self):
        self.el.stopData()
        self.el.stopRecording() # Might be redundant?
        self.el.closeDataFile()
        self.el.receiveDataFile(self.fname, self.fname)
        self.el.close()
