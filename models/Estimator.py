from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin

from models.FNN import FNN
from models.LoadableModule import LoadableModule
from models.Regressor import Regressor
from models.Trainer import Trainer
from repositories.db_models import FeedforwardNeuralConfig, NeuralProperties

LayerSize = int
NetworkDepth = int
NetworkShape = Tuple[LayerSize, NetworkDepth]


class Estimator(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        name: str = "",
        config: FeedforwardNeuralConfig = FeedforwardNeuralConfig(1E-3, 128, 138, 2, "ReLU"),
        epochs: int = 200
    ):
        self.name = name
        self.regressor = None
        self.config = config
        self.epochs = epochs

    def from_existing(self, net: LoadableModule) -> Estimator:
        self.regressor = Regressor(net)
        return self

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> Trainer:
        learning_rate, batch_size, network_size, depth, activation_fn = self.config
        net = FNN(nodes=network_size, depth=depth, activation=activation_fn)
        trainable_params = net.count_parameters()
        params_class = round(trainable_params / 10_000.0) * 10
        trainer = Trainer(net, lr=learning_rate, batch_size=batch_size)
        details = f"Size [{network_size}] Depth [{depth}] Params [{trainable_params}]  Class[{params_class} k]  Activation [{activation_fn}] "
        trainer.train(x_train, y_train, self.epochs, details=details)
        self.regressor = Regressor(net)
        return trainer

    def predict(self, X):
        return self.regressor.predict(torch.tensor(X))

    def score(self, x_test: np.ndarray, y_test: np.ndarray, sample_weight=None) -> NeuralProperties:
        score, r2 = self.regressor.evaluate(torch.tensor(x_test), torch.tensor(y_test))  # Neg RMSE
        return NeuralProperties(score, r2)


if __name__ == '__main__':
    est = Estimator(name="ackley", config=FeedforwardNeuralConfig(1E-3, 128, 138, 2, "ReLU"))
