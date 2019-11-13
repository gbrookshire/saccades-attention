# Analysis pipeline

This is the general analysis pipeline for the experiment.

## Bookkeeping

Store all experimental data in `../data/`.

Keep track of filenames in `../data/subject_info.csv`. 

## Pipeline

- Convert EyeLink files to plain-text: run `source eyelink_ascii.sh` in the terminal.
    - Requires `edf2asc`, in the developer tools from EyeLink. 
- Behavior
    - Time between fixations
    - Accuracy in the memory test
- MEG 
    - Reject artifacts from MEG data
    - (Source reconstruction?)
    - Evoked responses
    - Classification analyses
        - Stimulus identity (time-locked to fixation start)
        - Saccade direction (time-locked to fixation end)
