""" Load up the positions of the stimuli
"""

import json
import sys
sys.path.append('../exp-scripts')
from lattice import lattice
import dist_convert as dc 

def stim_positions():
    expt_info = json.load(open('expt_info.json')) 
    win_center = (0, 0) 
    stim_dist = int(dc.deg2pix(expt_info['stim_dist_deg'])) # Min distance between images
    stim_locs = lattice(center=win_center, n_fold=6, radius=stim_dist, n_layers=2)
    stim_locs = [dc.origin_psychopy2eyelink(pos) for pos in stim_locs]
    return stim_locs

