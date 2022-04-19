from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from torch import nn

from models.FNN import FNN
from models.Regressor import Regressor
from models.Trainer import Trainer


class Estimator(BaseEstimator, RegressorMixin):

    def __init__(
        self,
        learning_rate=10E-3,
        batch_size=128,
        network_size=50,
        depth=3,
        activation_fn=lambda: nn.ReLU()
    ):
        self.regressor = None
        self.trainer = None
        self.net = None
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.network_size = network_size
        self.depth = depth
        self.activation_fn = activation_fn
        self.activation_fn_name = type(activation_fn()).__name__

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> Estimator:
        self.net = FNN(nodes=self.network_size, depth=self.depth, activation_fn=self.activation_fn)
        self.trainer = Trainer(self.net, lr=self.learning_rate, batch_size=self.batch_size)
        details = f"Size [{self.network_size}] Depth [{self.depth}] Activation [{self.activation_fn_name}]"
        self.trainer.train(x_train, y_train, details=details)
        self.regressor = Regressor(self.net)
        return self

    def predict(self, X):
        return self.regressor.predict(torch.tensor(X))

    def score(self, x_test: np.ndarray, y_test: np.ndarray, sample_weight=None):
        return -1 * self.regressor.evaluate(x_test, y_test)  # Neg RMSE
