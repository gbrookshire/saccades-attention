"""
This is necessary for plots to appear in Castles.
"""

import tkinter
import matplotlib

def setup(): 
    matplotlib.use('TkAgg')

if __name__ == '__main__':
    setup()
