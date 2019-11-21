"""
Plot behavioral results for one subject
"""

import json
import pandas as pd
import matplotlib.pyplot as plt

# Which subject to run
n = 1

expt_info = json.load(open('expt_info.json'))
subject_info = pd.read_csv(expt_info['data_dir'] + 'subject_info.csv')

data_dir = expt_info['data_dir']

# Read in the behavioral logfile
behav_fname = f'{data_dir}logfiles/{subject_info["behav"][n]}.csv' 
behav = pd.read_csv(behav_fname)

# Only keep trials that were actually run
behav = behav.loc[behav['ran'] == 1.0]

behav['rt'] = behav['rt'].astype(float)

# Get response accuracy
behav['resp_side'] = ''
behav.loc[behav['resp'] == '4', 'resp_side'] = 'left'
behav.loc[behav['resp'] == '7', 'resp_side'] = 'right'
behav['correct'] = behav['mem_target_loc'] == behav['resp_side']

# Set up plot layout
f, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 3]})

# Plot accuracy
acc = behav['correct'].mean()
a0.bar(0.5, acc)
a0.set_xlim([-0.5, 1.5])
a0.set_xlabel('')
a0.set_xticks([])
a0.set_ylabel('Accuracy')
a0.text(0.5, 0.2, f'{acc*100:.1f}%', horizontalalignment='center')

# Plot histogram of RTs
a1.hist(behav['rt'])
a1.set_xlim([0, 4])
a1.set_xlabel('Time (s)')
a1.set_ylabel('Count')

f.set_size_inches(6, 2)
f.tight_layout()
plt.show()
