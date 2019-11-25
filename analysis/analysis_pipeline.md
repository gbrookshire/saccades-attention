# Analysis pipeline

This is the general analysis pipeline for the experiment.


## Bookkeeping

Store all experimental data in `saccades-attention/data/`.

Keep track of filenames and subject details in `saccades-attention/data/subject_info.csv`. 


## Setup

- On the computing clusters (Castles or Bluebear), run this in terminal to access `IPython` and `mne`:
    - `source load_python.sh`
- To make plots work on the clusters, run this in an IPython session:
    - `run plot_setup.py`


## Pipeline

- Convert EyeLink files to plain-text: run `source eyelink_ascii.sh` in the terminal.
    - Requires `edf2asc` in the developer tools from EyeLink. 
- Behavior
- MEG 
    - Reject artifacts from MEG data
    - Evoked responses
    - Classification analyses
        - Stimulus identity (time-locked to fixation start)
        - Saccade direction (time-locked to fixation end)
