from __future__ import annotations

import os
from typing import Union, Mapping

import numpy as np
import torch
import uuid as uuid
from sklearn.base import BaseEstimator, RegressorMixin
from torch import nn

from models.FNN import FNN
from models.Regressor import Regressor
from models.Trainer import Trainer



class Estimator(BaseEstimator, RegressorMixin):
    _activations: Mapping[str, nn.Module] = {
        "ReLU": lambda: nn.ReLU(),
        "Tanh": lambda: nn.Tanh()
    }

    def __init__(
        self,
        name="",
        learning_rate=1E-3,
        batch_size=128,
        network_size=50,
        depth=3,
        activation_fn="ReLU",
        should_save=False,
    ):
        self.name = name
        self.regressor = None
        self.trainer: Union[Trainer, None] = None
        self.net = None
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.network_size = network_size
        self.depth = depth
        self.activation_fn = activation_fn
        self.should_save = should_save

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> Estimator:
        self.net = FNN(nodes=self.network_size, depth=self.depth,
                       activation_fn=Estimator._activations[self.activation_fn])
        self.trainer = Trainer(self.net, lr=self.learning_rate, batch_size=self.batch_size)
        details = f"Size [{self.network_size}] Depth [{self.depth}] Activation [{self.activation_fn}]"
        self.trainer.train(x_train, y_train, details=details)
        self.regressor = Regressor(self.net)
        return self

    def predict(self, X):
        return self.regressor.predict(torch.tensor(X))

    def score(self, x_test: np.ndarray, y_test: np.ndarray, sample_weight=None):
        score, r2 = self.regressor.evaluate(torch.tensor(x_test), torch.tensor(y_test))  # Neg RMSE
        if self.should_save:
            self._log_model(score, r2)
        return -1 * score

    def _log_model(self, score, r2):
        id = uuid.uuid4().__str__()
        path = "./trained/metadata/"
        if not os.path.exists(path):
            os.mkdir(path)
        filename = f"{path}/{self.name}.csv"
        if not os.path.exists(filename):
            with open(filename, 'a') as f:
                f.write("id,learning_rate,batch_size,network_size,depth,activation_fn,rmse,r2\n")
        with open(filename, 'a') as f:
            model_params = f"{id},{self.learning_rate},{self.batch_size},{self.network_size},{self.depth},{self.activation_fn},{score},{r2}\n"
            f.write(model_params)
        self.trainer.save(self.name, id)
