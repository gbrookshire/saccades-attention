# Analysis pipeline

This is the general analysis pipeline for the experiment.


## Bookkeeping

Store all experimental data in `data/`.

Keep track of filenames and subject details in `data/subject_info.csv`. 


## Setup

- On the computing clusters (Castles or Bluebear), run this in terminal to access IPython and mne:
    - `source load_python.sh`
- On my desktop, run `conda activate mne` before starting IPython.
- To make plots work on the clusters, run this in an IPython session:
    - `run plot_setup.py`


## Pipeline

- Convert EyeLink files to plain-text
    - `$ source eyelink_ascii.sh`
    - Requires `edf2asc` in the developer tools from EyeLink. 
- Behavior
    - `$ python behavior.py`
- MEG 
    - Reject artifacts from MEG data
        - `artifacts.py`
    - Evoked responses
    - Classification analyses
        - Stimulus identity (time-locked to fixation start)
            - `classify_items.py`
        - Saccade direction (time-locked to fixation end)
