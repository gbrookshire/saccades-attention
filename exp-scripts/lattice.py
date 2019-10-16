"""
Geometric lattices
"""

import numpy as np


def lattice(center, n_fold, radius, n_layers): 
    """ Make a regularly tiled lattice
    Return a numpy.array of [x,y] coordinates
    center: Position around which the points are arranged
    n_fold: Degree of radial symmetry (4 makes squares, 6 hexagons, etc)
    radius: Distance between each point
    n_layers: How many concentric arrangements of points
    """
    angles = np.linspace(0, 2*np.pi, n_fold, endpoint=False)
    vertices = radius * np.exp(1j * angles)
    vertices = np.array([np.real(vertices), np.imag(vertices)]).transpose()

    pos = np.array([center])
    for n in range(n_layers):
        for p in pos:
            v = vertices + p # Vertices around this position
            pos = np.vstack([pos, v])
            # Remove duplicates
            pos = np.round(pos, decimals=3)
            pos = np.unique(pos, axis=0)
    return pos


def demo():
    """ Demo the library with a little plot
    """
    center = [0, 0] # Position of the center
    n_fold = 6 # How many folds of symmetry
    radius = 100 # Distance between vertices
    n_layers = 2 # How many times to spiral outward from the center
    pos = lattice(center, n_fold, radius, n_layers)

    import matplotlib.pyplot as plt
    plt.plot(pos[:,0], pos[:,1], 'o')
    plt.show()

if __name__ == '__main__':
    demo()
