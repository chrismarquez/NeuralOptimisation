from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy


@dataclass
class Polynomial:
    values: List[float]

    @staticmethod
    def make_cnn_polynomial(n: int, f: int, l: int) -> Polynomial:
        result = Polynomial([0])
        result += 3.0 * n ** 2
        result += Polynomial._inner_cnn_polynomial(f, l)
        result += Polynomial._final_fc_polynomial(n, f, l)
        return result

    @staticmethod
    def _inner_cnn_polynomial(f: int, l: int) -> Polynomial:
        result = Polynomial([0])
        k_values = [0] + [2 ** i for i in range(l)]
        for i in range(1, l + 1):
            if i == 1:
                result += Polynomial([f ** 2 + 1.0, 0.0])
            else:
                result += Polynomial([k_values[i - 1] * k_values[i] * f ** 2, k_values[i], 0.0])
        return result

    @staticmethod
    def _final_fc_polynomial(n: int, f: int, l: int) -> Polynomial:
        coeff = Polynomial.final_fc_coeff(n, f, l)
        return Polynomial([2.0 ** (l - 1) * coeff ** 3, coeff])

    @staticmethod
    def final_fc_coeff(n: int, f: int, i: int) -> int:
        if i == 0:
            return n
        else:
            res = int(math.floor(0.5 * (Polynomial.final_fc_coeff(n, f, i - 1) - f + 1)))
            return max(1, res)

    def __neg__(self):
        return Polynomial([-it for it in self.values])

    def __add__(self, other: Polynomial | float) -> Polynomial:
        return self.add(other)

    def __sub__(self, other: Polynomial | float):
        return self.add(-other)

    def add(self, other: Polynomial | float) -> Polynomial:
        if type(other) is Polynomial:
            return self._add_poly(other)
        else:
            return self._add_scalar(other)

    def _add_scalar(self, other: float) -> Polynomial:
        result = self.values[:]
        result[-1] += other
        return Polynomial(result)

    def _add_poly(self, other: Polynomial) -> Polynomial:
        largest_degree = max(len(self.values), len(other.values))
        result = [0.0 for _ in range(largest_degree)]
        for i in range(-1, - largest_degree - 1, -1):
            if i >= -len(self.values):
                result[i] += self.values[i]
            if i >= -len(other.values):
                result[i] += other.values[i]
        return Polynomial(result)

    def largest_root(self) -> float:
        roots = numpy.roots(self.values)
        return max(*roots)


if __name__ == '__main__':
    f_s = [3, 5, 7]
    for f in f_s:
        x = Polynomial.make_cnn_polynomial(20, f, 2)
        print(x)
        for i in range(1, 10):
            print((x - 10_000 * i).largest_root())
