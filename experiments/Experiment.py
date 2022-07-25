import math
from dataclasses import dataclass
from typing import Literal, Dict

NeuralType = Literal["Feedforward", "Convolutional"]


def _get_fnn_hyper_params():
    sizes_2 = [int(math.sqrt(2 * 10_000 * i + 15) - 4) for i in range(1, 10)]
    sizes_4 = [int(math.sqrt(32.0 / 21.0 * 10_000 * i - 32.0 / 21.0 + (64.0 / 21.0) ** 2) - 64.0 / 21.0) for i in
               range(1, 10)]

    sizes_2 = [(it, 2) for it in sizes_2]
    sizes_4 = [(it, 4) for it in sizes_4]

    return {
        "learning_rate": [1E-6, 3E-7],
        "batch_size": [128, 512],
        "network_shape": sizes_2 + sizes_4,
        "activation_fn": ["ReLU", "Sigmoid", "Tanh", "Softplus"],
    }


def _get_cnn_hyper_params():
    return {
        "learning_rate": [1E-6, 3E-7],
        "batch_size": [128, 512],
        "filter_size": [3, 5, 7],
        "depth": [2, 4],
        "learnable_parameters": [10_000 * i for i in range(1, 10)],
        "activation_fn": ["ReLU", "Sigmoid"],
    }


@dataclass
class Experiment:
    exp_id: str
    type: NeuralType
    epochs: int

    def get_hyper_params(self) -> Dict:
        if self.type == "Feedforward":
            return _get_fnn_hyper_params()
        elif self.type == "Convolutional":
            return _get_cnn_hyper_params()
