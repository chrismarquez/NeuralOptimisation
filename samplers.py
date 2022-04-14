import numpy as np
from scipy.stats import qmc

import functions
from functions import Function2D


def sobel_sample(fn: Function2D, sample_amount: int) -> np.ndarray:
    power = int(np.log2(sample_amount - 1) + 1)
    sampler = qmc.Sobol(d=2, scramble=True)
    samples = sampler.random_base2(m=power)
    samples = np.array(samples[:sample_amount])
    x, y = samples[:, 0], samples[:, 1]
    z = fn(x, y)
    return np.array(list(zip(x, y, z)))


if __name__ == '__main__':
    z = sobel_sample(functions.sum_squares, 16)
    print(z)