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

def plot_eyetracking(n):
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
    # TODO Add a box where each stimulus is instead of just a point
    
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
    fname = f"{expt_info['data_dir']}plots/behavior/fixations/{n}.pdf" 
    plt.savefig(fname)


# Look for serial dependence in saccade directions

# Get the direction of the next saccade
# Check the timing between fixations to make sure
# that the next item on the list is actually the next fixation.
import numpy as np
import pandas as pd
fix = d['fix_info'].copy()
onsets = fix['start'][1:].to_numpy()
offsets = fix['end'][:-1].to_numpy()
saccade_dur = onsets - offsets
saccade_dur = saccade_dur / 1000
saccade_dur = np.hstack((saccade_dur, np.inf))
plausible_saccade = saccade_dur < 0.15
x_change = fix['x_avg'][1:].to_numpy() - fix['x_avg'][:-1].to_numpy()
x_change = np.hstack((x_change, np.nan))
y_change = fix['y_avg'][1:].to_numpy() - fix['y_avg'][:-1].to_numpy()
y_change = np.hstack((y_change, np.nan))
saccade_movement = x_change + (y_change * 1j)
saccade_angle = np.angle(saccade_movement)
saccade_dist = np.abs(saccade_movement) 

new_col = lambda col: pd.Series(col, index=fix.index)
fix['x_change'] = new_col(x_change)
fix['y_change'] = new_col(y_change)
fix['saccade_dur'] = new_col(saccade_dur)
fix['saccade'] = new_col(plausible_saccade)
fix['saccade_angle'] = new_col(saccade_angle)
fix['saccade_dist'] = new_col(saccade_dist) 

# Only keep saccades to new objects
# To do this, check whether the next fixated stim is different
new_object = np.diff(fix['closest_loc']) != 0
new_object = np.hstack([new_object, False])
fix['new_object'] = new_col(new_object)

# Only keep real saccades
fix = fix.loc[fix['saccade']]
# Only keep saccades to a new object
fix = fix.loc[fix['new_object']]


# Plot all the autocorrelations

maxlag = 20

def acorr(x, **kwargs):
    plt.acorr(x,
              maxlags=maxlag, 
              detrend=lambda x: x - x.mean(),
              usevlines=False,
              linestyle='-',
              **kwargs)

plt.figure()

# plt.subplot(2, 1, 1)
acorr(fix['x_change'], label='Horiz') 
acorr(fix['y_change'], label='Vert')
plt.plot([0, maxlag], [0, 0], '-k')
plt.xticks([0, 10, 20])
plt.xlim(0, maxlag)
plt.xlabel('Lag')
plt.ylabel('Corr')
plt.legend()

# Negative autocorrelations at low lags indicate that people are
# fixating back and forth
# How can we make sure that these autocorrelations are removed?
# Randomly drop trials until they disappear?

# Do people switch back and forth between stimuli?
# If so, then we should see a concentration of points along the line
# through the origin with slope = -1.
# But maybe a negative correlation here just indicates that subjects
# are moving their eyes to a new object. What's the alternative? If there
# was a correlation of r=0, that would indicate random eye movements.
# But there are only a few images on the screen, so people have to saccade
# back and forth between them. 
from scipy import stats
plt.clf()
fields = ['x_change', 'y_change', 'saccade_dist']
for i_plot, var in enumerate(fields):
    plt.subplot(1, 3, i_plot + 1)
    # Two copies of the same data, lagged by one item
    v1 = fix[var].iloc[1:]
    v2 = fix[var].iloc[:-1]
    # Get the line of best fit
    b = np.polynomial.polynomial.polyfit(v1, v2, deg=1)
    stat = stats.pearsonr(v1, v2)
    # Plot the correlation line
    plt.plot(v1, b[0] + b[1] * v1, '-r')
    # Plot the subsequent saccades
    plt.plot(v1, v2, 'o', alpha=0.1)
    plt.title(var)
    text = f"r = {stat[0]:.2f}\np = {stat[1]:1.0e}"
    plt.text(v1.min(), v1.min(), text,
             horizontalalignment='left',
             verticalalignment='bottom')
plt.tight_layout()
