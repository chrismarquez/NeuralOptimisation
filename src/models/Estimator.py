from __future__ import annotations

import gc
import os
import uuid as uuid
from typing import Union, Tuple

import numpy as np
import pandas
import torch
from sklearn.base import BaseEstimator, RegressorMixin

from src.models.FNN import FNN, Activation
from src.models.Regressor import Regressor
from src.models.Trainer import Trainer

LayerSize = int
NetworkDepth = int
NetworkShape = Tuple[LayerSize, NetworkDepth]


class Estimator(BaseEstimator, RegressorMixin):

    def __init__(
        self,
        name: str = "",
        learning_rate: float = 1E-3,
        batch_size: int = 128,
        network_shape: NetworkShape = (138, 2),
        activation_fn: Activation = "ReLU",
        epochs: int = 200,
        should_save: bool = False
    ):
        self.name = name
        self.regressor = None
        self.trainer: Union[Trainer, None] = None
        self.net = None
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.network_shape = network_shape
        self.activation_fn = activation_fn
        self.epochs = epochs
        self.should_save = should_save

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> Estimator:
        self.net = self.load()
        size, depth = self.network_shape
        if self.net is None:
            self.net = FNN(nodes=size, depth=depth, activation=self.activation_fn)
            trainable_params = self.net.count_parameters()
            params_class = round(trainable_params / 10_000.0) * 10
            self.trainer = Trainer(self.net, lr=self.learning_rate, batch_size=self.batch_size)
            details = f"Size [{size}] Depth [{depth}] Params [{trainable_params}]  Class[{params_class} k]  Activation [{self.activation_fn}] "
            self.trainer.train(x_train, y_train, self.epochs,details=details)
            self.regressor = Regressor(self.net)
        else:
            self.should_save = False
            self.trainer = Trainer(self.net, lr=self.learning_rate, batch_size=self.batch_size)
            self.regressor = Regressor(self.net)
        return self

    def load(self):
        path = f"./trained/metadata/{self.name}.csv"
        nodes, depth = self.network_shape
        if os.path.exists(path):
            models = pandas.read_csv(path)
            df = models[models['learning_rate'] == self.learning_rate]
            df = df[df['batch_size'] == self.batch_size]
            df = df[df['network_size'] == nodes]
            df = df[df['depth'] == depth]
            df = df[df['activation_fn'] == self.activation_fn]
            if not df.empty:
                id = df['id'].values[0]
                model_path = f"./trained/{self.name}/{id}.pt"
                net = FNN(nodes, depth, self.activation_fn).load(model_path)
                gc.collect()
                torch.cuda.empty_cache()  # Clean GPU memory before use
                return net
        return None

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
            network_size, depth = self.network_shape
            model_params = f"{id},{self.learning_rate},{self.batch_size},{network_size},{depth},{self.activation_fn},{score},{r2}\n"
            f.write(model_params)
        self.trainer.save(self.name, id)


if __name__ == '__main__':
    est = Estimator(name="ackley", learning_rate=0.0001, batch_size=128, network_shape=(138, 2), activation_fn="ReLU")
    est.load()
