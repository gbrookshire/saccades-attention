"""
This is necessary for plots to appear in Castles.
"""

import tkinter
import matplotlib
import socket

def setup(): 
    if socket.gethostname() == 'vm-jenseno-meg.novalocal':
        matplotlib.use('TkAgg')

if __name__ == '__main__':
    setup()
