"""
Do something for each subject
"""

import json
import pandas as pd

expt_info = json.load(open('expt_info.json'))
subject_info = pd.read_csv(expt_info['data_dir'] + 'subject_info.csv',
                           engine='python', sep=',')

def apply(fnc):
    """ Apply the function to each non-excluded subject.
    """
    results = []
    for inx, row in subject_info.iterrows():
        if not row['exclude']:
            res = fnc(row)
            results.append(res)
    return results

