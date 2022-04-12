from __future__ import annotations
from typing import Callable, Union

import numpy as np
import matplotlib.pyplot as plt

Real = Union[np.ndarray, float]
Function2D = Callable[[Real, Real], Real]


def ackley(x: Real, y: Real) -> Real:
    terms = [
        - 20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))),
        - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))),
        np.e,
        20
    ]
    return sum(terms)


def rosenbrock(x: Real, y: Real):
    a = 1
    b = 100
    return (a - x) ** 2 + b * (y - x ** 2) ** 2


def plot(fn: Function2D):
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
    plot(fn=rosenbrock)
