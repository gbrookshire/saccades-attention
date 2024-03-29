"""
Plot behavioral results for one subject
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import plot_setup

plot_setup.setup()

expt_info = json.load(open('expt_info.json'))

if socket.gethostname() == 'colles-d164179':
    data_dir = expt_info['data_dir']['external']
else:
    data_dir = expt_info['data_dir']['standard']

subject_info = pd.read_csv(data_dir + 'subject_info.csv') 

def coerce_numeric(x):
    """ Coerce a vector of strings into numeric data
    """
    if x.dtype == np.number:
        return x.to_numpy()
    else:
        return np.genfromtxt(x)

def analyze_behavior(n):
    """ Analyze data for one subject 
    Return accuracy and RT
    """ 
    # Read in the behavioral logfile
    behav_fname = f'{data_dir}logfiles/{subject_info["behav"][n]}.csv' 
    col_dtypes = {'resp': str}
    behav = pd.read_csv(behav_fname, dtype=col_dtypes)
    # Only keep trials that were actually run
    behav = behav.loc[behav['ran'] == 1.0] 
    # Get response accuracy
    behav['resp_side'] = ''
    behav.loc[behav['resp'] == '7', 'resp_side'] = 'left'
    behav.loc[behav['resp'] == '8', 'resp_side'] = 'right'
    behav['correct'] = behav['mem_target_loc'] == behav['resp_side'] 
    # Get variables to return
    acc = behav['correct'].mean() 
    rt = coerce_numeric(behav['rt'])
    return acc, rt


def analyze_all_subjects():
    results = []
    for n in subject_info.index:
        if subject_info['exclude'][n]:
            continue
        else:
            results.append(analyze_behavior(n))

    # Get lists of accuracy and RT distributions for each subject
    acc, rt = zip(*results)
    plot_results(acc, rt)


def plot_results(acc, rt):
    assert len(acc) < 10 # Need to choose a different colormap for >10 subs
    colors = cm.tab10(np.arange(10))

    # Set up plot layout
    f, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 3]})

    # Plot accuracy
    a0.bar(0.5, np.mean(acc),
           color='none', edgecolor='black') 
    for i_subj in range(len(acc)):
        a0.plot(0.5, acc[i_subj], 'o',
                fillstyle='none',
                color=colors[i_subj])
    a0.text(0.5, 0.2,
            f'{np.mean(acc)*100:.1f}%',
            horizontalalignment='center')
    a0.set_xlim([-0.5, 1.5])
    a0.set_xticks([])
    a0.set_xlabel('')
    a0.set_ylim(0, 1.0)
    a0.set_yticks([0, 0.5, 1.0])
    a0.set_ylabel('Accuracy')

    # Plot RT histograms for each subject
    for i_subj,subj_rt in enumerate(rt):
        subj_rt = subj_rt[~np.isnan(subj_rt)]
        a1.hist(subj_rt,
                bins=20,
                histtype='step',
                color=colors[i_subj])
    a1.set_xlim([0, 4])
    a1.set_xlabel('Time (s)')
    a1.set_ylabel('Count')

    # Format the plot
    f.set_size_inches(6, 2)
    f.tight_layout()
    plt.show()
    fname = f"{data_dir}plots/behavior/acc_rt.pdf"
    plt.savefig(fname)


if __name__ == '__main__':
    analyze_all_subjects()





