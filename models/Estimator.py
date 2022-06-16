from __future__ import annotations

import gc
from typing import Union, Tuple

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin

from models.FNN import FNN
from models.Regressor import Regressor
from models.Trainer import Trainer
from repositories.NeuralModelRepository import NeuralModelRepository
from repositories.db_models import NeuralConfig, NeuralProperties, NeuralModel

LayerSize = int
NetworkDepth = int
NetworkShape = Tuple[LayerSize, NetworkDepth]


class Estimator(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        repo: NeuralModelRepository,
        name: str = "",
        config: NeuralConfig = NeuralConfig(1E-3, 128, 138, 2, "ReLU"),
        epochs: int = 200,
        should_save: bool = False
    ):
        self.name = name
        self.regressor = None
        self.trainer: Union[Trainer, None] = None
        self.net = None
        self.config = config
        self.epochs = epochs
        self.should_save = should_save
        self.repo = repo

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> Estimator:
        self.net = self.load()
        learning_rate, batch_size, network_size, depth, activation_fn = self.config
        if self.net is None:
            self.net = FNN(nodes=network_size, depth=depth, activation=activation_fn)
            trainable_params = self.net.count_parameters()
            params_class = round(trainable_params / 10_000.0) * 10
            self.trainer = Trainer(self.net, lr=learning_rate, batch_size=batch_size)
            details = f"Size [{network_size}] Depth [{depth}] Params [{trainable_params}]  Class[{params_class} k]  Activation [{activation_fn}] "
            self.trainer.train(x_train, y_train, self.epochs, details=details)
            self.regressor = Regressor(self.net)
        else:
            self.should_save = False
            self.trainer = Trainer(self.net, lr=learning_rate, batch_size=batch_size)
            self.regressor = Regressor(self.net)
        return self

    def load(self):
        existing_models = self.repo.get_by_config(self.name, self.config)
        if len(existing_models) == 0:
            return None
        model = existing_models[0]
        net = FNN(self.config.network_size, self.config.depth, self.config.activation_fn).load_bytes(model.model_data)
        gc.collect()
        torch.cuda.empty_cache()  # Clean GPU memory before use
        return net

    def predict(self, X):
        return self.regressor.predict(torch.tensor(X))

    def score(self, x_test: np.ndarray, y_test: np.ndarray, sample_weight=None):
        score, r2 = self.regressor.evaluate(torch.tensor(x_test), torch.tensor(y_test))  # Neg RMSE
        if self.should_save:
            self._log_model(score, r2)
        return -1 * score

    def _log_model(self, score, r2): #  "id,learning_rate,batch_size,network_size,depth,activation_fn,rmse,r2\n"
        props = NeuralProperties(score, r2)
        model_data = self.trainer.get_model_data()
        model = NeuralModel(self.name, self.config, props, model_data)
        self.repo.save(model)


if __name__ == '__main__':
    repo = NeuralModelRepository()
    est = Estimator(repo, name="ackley", config=NeuralConfig(1E-3, 128, 138, 2, "ReLU"))
    est.load()
