# saccades-attention

How does the brain gather information before an eye movement?

This repo has the code used in a study I did at the University of Birmingham in Ole Jensen's lab. [This presentation](https://docs.google.com/presentation/d/1glRY3vG054nz8KTlRrEk7d5SLgWP8zoJ5Xdoj2XtCPU/edit?usp=sharing) gives some information about the study.

# Design

- Show 4 randomly selected images (from 6 total images) in random locations on the screen.
    - Images remain on screen for 5 seconds.
- At the end of each trial, test memory for the locations of the images.
    - After a visual masking stimulus, a dot appears at the location of one of the stimuli.
    - Two images are displayed at the bottom of the screen.
    - Participants report which of the two displayed images was at the location of the dot.


# Requirements

## Stimulus presentation
- [Psychopy 3.2.3](https://www.psychopy.org/download.html)

## Analysis
- [mne-python 0.19.2](https://mne.tools/stable/install/mne_python.html) 
Detailed requirements in analysis/requirements.txt.
