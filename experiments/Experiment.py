import math
from dataclasses import dataclass


def _get_hyper_params():
    sizes_2 = [int(math.sqrt(2 * 10_000 * i + 15) - 4) for i in range(1, 10)]
    sizes_4 = [int(math.sqrt(32.0 / 21.0 * 10_000 * i - 32.0 / 21.0 + (64.0 / 21.0) ** 2) - 64.0 / 21.0) for i in
               range(1, 9)]

    sizes_2 = [(it, 2) for it in sizes_2]
    sizes_4 = [(it, 4) for it in sizes_4]

    return {
        "learning_rate": [1E-6, 3E-7],  # Evenly spaced lr in log scale
        "batch_size": [128, 512],
        "network_shape": sizes_2 + sizes_4,
        "activation_fn": ["ReLU", "Sigmoid"],
    }


@dataclass
class Experiment:
    exp_id: str
    hyper_params = _get_hyper_params()
