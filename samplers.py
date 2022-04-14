import os.path

import numpy as np
from scipy.stats import qmc

import functions
from functions import Function2D


def sobel_sample(fn: Function2D, sample_amount: int, x_max=1.0) -> np.ndarray:
    power = int(np.log2(sample_amount / 2 - 1) + 1)  # Divide by two so when mirrored the intended sample amount is used
    sampler = qmc.Sobol(d=2, scramble=True)
    samples = sampler.random_base2(m=power)
    samples = np.array(samples[:sample_amount]) * x_max  # Scale points
    samples = np.concatenate([samples, -samples], axis=0)
    x, y = samples[:, 0], samples[:, 1]
    z = fn(x, y)
    return np.array(list(zip(x, y, z)))


if __name__ == '__main__':
    for fn, x_max in functions.pool:
        samples = sobel_sample(fn, sample_amount=2**15, x_max=x_max)
        name = fn.__name__
        path = "./samples/"
        if not os.path.exists(path):
            os.mkdir(path)
        np.savetxt(f"samples/{name}.csv", X=samples, delimiter=",")
