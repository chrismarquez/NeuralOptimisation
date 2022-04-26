import os.path
import typing

import numpy as np
from scipy.stats import qmc

from src.data import functions
from src.data.functions import Function2D


def sobel_sample(fn: Function2D, density = 128, x_max=1.0) -> np.ndarray:
    sample_amount = int(density * x_max ** 2)  # Samples per quadrant up to x_max
    print(f"Generating {sample_amount} samples with density {density} per unit squared and range up to {x_max}")
    power = int(np.ceil(np.log2(sample_amount)))
    print(f"Closest power of two: {power} --> {2 ** power}")
    sampler = qmc.Sobol(d=2, scramble=True)
    quadrant_samples = sampler.random_base2(m=power)
    quadrant_samples = np.array(quadrant_samples[:sample_amount]) * x_max  # Scale points
    print(f"Filling remaining quadrants, resulting in  {sample_amount * 4} samples")
    samples = []
    for sample in quadrant_samples:
        x, y = sample
        samples.append(sample)
        samples.append([-x, y])
        samples.append([x, -y])
        samples.append(-sample)
    samples = np.array(samples)
    x, y = samples[:, 0], samples[:, 1]
    z = fn(x, y)
    return np.array(list(zip(x, y, z)))


if __name__ == '__main__':
    for fn, x_max in functions.pool.items():
        samples = sobel_sample(fn, x_max=x_max)
        name = fn.__name__
        path = "../../resources/samples/"
        if not os.path.exists(path):
            os.mkdir(path)
        np.savetxt(f"samples/{name}.csv", X=typing.cast(samples, int), delimiter=",")
