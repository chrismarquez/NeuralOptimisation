from __future__ import annotations
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


def ackley(x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray | float:
    terms = [
        - 20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))),
        - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))),
        np.e,
        20
    ]
    return sum(terms)


def plot(fn: Callable[[np.ndarray | float, np.ndarray | float], np.ndarray | float]):
    r_min, r_max = -32.768, 32.768
    xaxis = np.arange(r_min, r_max, 2.0)
    yaxis = np.arange(r_min, r_max, 2.0)
    x, y = np.meshgrid(xaxis, yaxis)
    results: np.ndarray = fn(x, y)
    figure = plt.figure()
    axis = figure.gca(projection='3d')
    axis.plot_surface(x, y, results, cmap='jet', shade="false")
    plt.show()
    plt.contour(x, y, results)
    plt.show()
    plt.scatter(x, y, results)
    plt.show()


if __name__ == '__main__':
    print(ackley(0.0, 0.0))
    plot(fn=ackley)
