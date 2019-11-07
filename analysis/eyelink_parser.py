"""
Parse eye-link files. Construct pd.DataFrame objects for fixations and triggers
"""

import pandas as pd


def _get_entries(lines, start_str):
    """ Get lines of the object `lines` that start with `start_str`,
        splitting the lines on whitespace.
    """ 
    s = [e.split() for e in lines if e.startswith(start_str)]
    return s


class EyelinkData(object):
    """
    Initialized with the filename of the eyelink .asc file.
    Has the following attributes.
    - lines: All lines from the data file
    - fixations: pd.DataFrame of fixations
    - triggers: pd.DataFrame of triggers
    """

    def __init__(self, fname):
        with open(fname, 'r') as f:
            lines = f.readlines()
        self.lines = lines

        # Fixations
        colnames = ['EFIX', 'eye_side', 'start', 'end', 'dur', 'x_avg', 'y_avg', 'pupil']
        fix = _get_entries(self.lines, 'EFIX')
        fix = pd.DataFrame(fix, columns=colnames)
        fix = fix.drop(columns='EFIX')
        for col in ['start', 'end', 'dur', 'pupil']:
            fix[col] = fix[col].astype(int)
        for col in ['x_avg', 'y_avg']:
            fix[col] = fix[col].astype(float)
        self.fixations = fix

        # Triggers
        colnames = ['MSG', 'time_stamp', 'type', 'value']
        msg = _get_entries(self.lines, 'MSG')
        msg = [m for m in msg if m[2] == 'Trigger']
        msg = pd.DataFrame(msg, columns=colnames)
        msg = msg.drop(columns=['MSG', 'type'])
        msg = msg.astype(int)
        self.triggers = msg

