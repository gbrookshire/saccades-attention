import pandas as pd
import matplotlib.pyplot as plt

fnames = {'meg': '191104/yalitest.fif',
          'eye': '19110415.asc',
          'behav': '2019-11-04-1527.csv'}

data_dir = '../data/'

behav = pd.read_csv(data_dir + 'logfiles/' + fnames['behav'])

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
a0.bar(0.5, behav['correct'].mean())
a0.set_xlim([-1, 2])
a0.set_xlabel('')
a0.set_xticks([])
a0.set_ylabel('Accuracy')

# Plot histogram of RTs
a1.hist(behav['rt'])
a1.set_xlim([0, 4])
a1.set_xlabel('Time (s)')
a1.set_ylabel('Count')

f.tight_layout()
plt.show()
