""" Minimal example raising a RuntimeError in Psychopy
"""
from psychopy import visual, core, data, event, monitors

win = visual.Window((500, 500),
                    fullscr=False,
                    color=(0, 0, 0),
                    colorSpace='rgb',
                    allowGUI=False)

text_stim = visual.TextStim(win, pos=(0, 0), text='hello')

text_stim.draw()
win.flip()
core.wait(4)
win.close()
