""" Load eyelink data
"""

import json
import matplotlib.pyplot as plt
import eyelink_parser
import load_data

# Stim libraries for the positions of the stimuli
import sys
sys.path.append('../exp-scripts')
from lattice import lattice
import dist_convert as dc 

# Get the positions of the stimuli
STIM_SIZE_DEG = 2.0
STIM_DIST_DEG = 3.0
STIM_SIZE = int(dc.deg2pix(STIM_SIZE_DEG)) # Size in pixels
STIM_DIST = int(dc.deg2pix(STIM_DIST_DEG)) # Min distance between images
win_center = (0, 0) 
stim_locs = lattice(center=win_center, n_fold=6, radius=STIM_DIST, n_layers=2)
stim_locs = [dc.origin_psychopy2eyelink(pos) for pos in stim_locs]

expt_info = json.load(open('expt_info.json'))

n = 1
d = load_data.load_data(n)

# Select fixation onsets
row_sel = d['fix_events'][:,2] == expt_info['event_dict']['fix_on']
d['fix_events'] = d['fix_events'][row_sel, :]


fix = d['fix_info']

# Set up the grid of subplots
f = plt.figure(constrained_layout=True,
               figsize=(6, 3))
gs0 = f.add_gridspec(1, 2)
gs_left = gs0[0].subgridspec(1, 1)
gs_right = gs0[1].subgridspec(2, 1)

# Plot fixation positions
f.add_subplot(gs_left[0,0])
plt.plot(fix['x_avg'], fix['y_avg'],
         'o', fillstyle='none', alpha=0.5) 
# Plot lattice locations
x,y = list(zip(*stim_locs))
plt.plot(x, y, '*r')
ax = plt.gca()
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)

# Histogram of fixation duration
f.add_subplot(gs_right[0,0])
plt.hist(fix['dur'] / 1000,
         bins=25, density=True)
plt.xlabel('Fixation Duration (s)')
plt.ylabel('Density')

# Plot histogram of inter-saccade interval
f.add_subplot(gs_right[1,0])
onsets = fix['start'][1:].to_numpy()
offsets = fix['end'][:-1].to_numpy()
isi = onsets - offsets
isi = isi / 1000
isi = isi[isi < 0.15]
plt.hist(isi,
         bins=25, density=True) 
plt.xlabel('Saccade Duration (s)')
plt.ylabel('Density')
plt.xlim(0, 0.150)

plt.show()
