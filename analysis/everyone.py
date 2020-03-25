#!/usr/bin/env python3
"""
Do something for each subject
"""

import os
import sys
import json
import csv
import socket

expt_info = json.load(open('expt_info.json'))

if socket.gethostname() == 'colles-d164179':
    data_dir = expt_info['data_dir']['external']
else:
    data_dir = expt_info['data_dir']['standard']

with open(data_dir + 'subject_info.csv', 'r') as f:
    reader = csv.DictReader(f)
    subject_info = list(reader)

def apply_fnc(fnc):
    """ Apply the function to each non-excluded subject.

    fnc: A function that takes a row of `subject_info.csv`
    """
    results = []
    for row in subject_info:
        if row['exclude'] != '1':
            res = fnc(row)
            results.append(res)
    return results


def slurm(cmd):
    """
    Pass arguments on to sbatch for all subjects

    Args:
    cmd: A command to pass to `slurmit`. Write `<N>` for subject number (str)

    Example:
    > python3 everyone.py "'rsa_item.py <N>' -t 15:00 -m 30G -c 5"
    Make sure to use double quotes around the whole slurm string, and single
    quotes around the part that goes to the python call.
    """
    cmd_template = './slurmit ' + cmd
    number_tag = '<N>'
    def slurm_call(row):
        c = cmd_template.replace(number_tag, str(row['n']))
        print(c)
        os.system(c)
    results = apply_fnc(slurm_call)


if __name__ == '__main__':
    slurm(sys.argv[1])
