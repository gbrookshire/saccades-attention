""" Load eyelink data
"""

import matplotlib.pyplot as plt
import eyelink_parser

# Get the positions of the stimuli
import sys
sys.path.append('../exp-scripts')
from lattice import lattice
import dist_convert as dc 
STIM_SIZE_DEG = 2.0
STIM_DIST_DEG = 3.0
STIM_SIZE = int(dc.deg2pix(STIM_SIZE_DEG)) # Size in pixels
STIM_DIST = int(dc.deg2pix(STIM_DIST_DEG)) # Min distance between images
win_center = (0, 0) 
stim_locs = lattice(center=win_center, n_fold=6, radius=STIM_DIST, n_layers=2)
stim_locs = [dc.origin_psychopy2eyelink(pos) for pos in stim_locs]

fname = '../data/eyelink/ascii/19110415.asc'
edata = eyelink_parser.EyelinkData(fname)

# Only keep fixations to the screen
fix = edata.fixations
fix = fix[(fix['x_avg'] > 0) & (fix['y_avg'] > 0)]

# Plot fixations
plt.plot(fix['x_avg'], fix['y_avg'],
         'o', fillstyle='none', alpha=0.5)

# Plot lattice locations
x,y = list(zip(*stim_locs))
plt.plot(x, y, '*r')

plt.show()
