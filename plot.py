import numpy as np
from matplotlib import pyplot as plt

import functions
from functions import Function2D


def plot_surface(fn: Function2D, x_max=5.12, n_points=100) -> None:
    r_min, r_max = -x_max, x_max
    spacing = 2 * x_max / n_points
    xaxis = np.arange(r_min, r_max, spacing)
    yaxis = np.arange(r_min, r_max, spacing)
    x, y = np.meshgrid(xaxis, yaxis)
    results: np.ndarray = fn(x, y)
    figure = plt.figure()
    axis = figure.gca(projection='3d')
    axis.plot_surface(x, y, results, cmap='jet', shade="false")
    plt.show()
    # plt.contour(x, y, results)
    # plt.show()
    # plt.scatter(x, y, results)
    # plt.show()


if __name__ == '__main__':
    plot_surface(fn=functions.ackley, x_max=32.768)
    plot_surface(fn=functions.rosenbrock, x_max=32.768)
    plot_surface(fn=functions.rastrigin)
    plot_surface(fn=functions.sum_squares)
