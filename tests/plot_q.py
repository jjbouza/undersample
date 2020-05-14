from dipy.io import read_bvals_bvecs
import numpy as np

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt


def plot(fbval, fbvec):
    """
    Plots the q-space vectors from bvec, bval data.
    """
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for bval, bvec in zip(bvals, bvecs):
        q = bval*bvec

        c = 'blue'
        if bval <= 5:
            c = 'red'
        elif bval <= 1005:
            c = 'green'
        elif bval <= 2005:
            c = 'yellow'
        elif bval <= 3010:
            c = 'orange'

        ax.scatter(q[0], q[1], q[2], c=c)


    plt.show()

if __name__ == "__main__":
    plot('./test_data/david_data/3112_BL_bvals', './test_data/david_data/3112_BL_bvecs')
