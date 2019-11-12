'''
plot surface perf results
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


def make_plot(csv_file):

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    res = np.loadtxt(csv_file, delimiter=',', dtype=float)

    # Make data.
    X = res[:, 1]
    Y = res[:, 0]
    Z = res[:, 2]
    Z /= max(res[:, 2])

    # Plot the surface.
    ax.plot_trisurf(X, Y, Z, cmap='viridis')

    ax.set_xlabel('Number of Neurons')
    ax.set_ylabel('Number of Layers')
    ax.set_title("Normalized Time For DNN Results")


    plt.show()

if __name__ == "__main__":
    make_plot("vec_perf.csv")