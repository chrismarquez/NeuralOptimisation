from __future__ import annotations
from typing import Callable, Union

import numpy as np

Real = Union[np.ndarray, float]
Function2D = Callable[[Real, Real], Real]


def sum_squares(x: Real, y: Real) -> Real:
    return x ** 2 + 2 * y ** 2


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


def rastrigin(x: Real, y: Real) -> Real:
    term = x ** 2 + y ** 2 - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))
    return 20 + term





if __name__ == '__main__':
    print(sum_squares(0.0, 0.0))