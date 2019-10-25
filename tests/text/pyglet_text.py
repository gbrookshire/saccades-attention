""" Test the text function in pyglet
Geoff Brookshire
G.Brookshire@bham.ac.uk
"""

import pyglet

win = pyglet.window.Window(800, 800)
text = pyglet.text.Label(text='hello', x=100, y=100)

@win.event
def on_draw():
    win.clear()
    text.draw()


if __name__ == '__main__':
    pyglet.app.run()
