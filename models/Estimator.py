from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin

from cluster.JobInit import init_container
from models.CNN import CNN
from models.FNN import FNN
from models.LoadableModule import LoadableModule
from models.Regressor import Regressor
from models.Trainer import Trainer, NanException
from repositories.db_models import FeedforwardNeuralConfig, NeuralProperties, NeuralConfig, ConvolutionalNeuralConfig

LayerSize = int
NetworkDepth = int
NetworkShape = Tuple[LayerSize, NetworkDepth]


class Estimator(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        name: str = "",
        config: NeuralConfig = FeedforwardNeuralConfig(1E-3, 128, 138, 2, "ReLU"),
        epochs: int = 200,
        l1_reg_lambda: Optional[float] = None
    ):
        self.name = name
        self.regressor = None
        self.config = config
        self.epochs = epochs
        self.l1_reg_lambda = l1_reg_lambda

    def from_existing(self, net: LoadableModule) -> Estimator:
        self.regressor = Regressor(net)
        return self

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> Trainer:
        net = self._build_net()
        trainable_params = net.count_parameters()
        params_class = round(trainable_params / 10_000.0) * 10
        trainer = Trainer(
            net, lr=self.config.learning_rate, batch_size=self.config.batch_size, l1_reg_lambda=self.l1_reg_lambda
        )
        details = f"Type [{type(self.config)}] Depth [{self.config.depth}] Params [{trainable_params}]  Class[{params_class} k]  Activation [{self.config.activation_fn}] "
        trainer.train(x_train, y_train, self.epochs, details=details)
        self.regressor = Regressor(net)
        return trainer

    def _build_net(self) -> LoadableModule:
        if type(self.config) is FeedforwardNeuralConfig:
            _, _, network_size, depth, activation_fn = self.config
            return FNN(nodes=network_size, depth=depth, activation=activation_fn)
        elif type(self.config) is ConvolutionalNeuralConfig:
            return CNN(start_size=self.config.start_size, filter_size=self.config.filter_size, filters=self.config.filters, depth=self.config.depth, activation=self.config.activation_fn)
        else:
            raise RuntimeError("Unrecognized Network Type")

    def predict(self, X):
        return self.regressor.predict(torch.tensor(X))

    def score(self, x_test: np.ndarray, y_test: np.ndarray, sample_weight=None) -> NeuralProperties:
        score, r2 = self.regressor.evaluate(torch.tensor(x_test), torch.tensor(y_test))  # Neg RMSE
        return NeuralProperties(score, r2)


if __name__ == '__main__':
    l1_reg_lambda = 0.0005
    container = init_container()
    sample_repo = container.sample_repository()
    neural_repo = container.neural_repository()
    model = neural_repo.get("62fcf647cc4b54de3c2efd5a") # '62fcf647cc4b54de3c2efd56'
    est = Estimator(
        name=model.function,
        config=model.neural_config,
        epochs=5000,
        #l1_reg_lambda=l1_reg_lambda
    )
    dataset_id = sample_repo.get_id_by_name(model.function)
    dataset = sample_repo.get(dataset_id).to_dataset()
    x_train, y_train = dataset.train
    x_test, y_test = dataset.test
    while True:
        try:
            trainer = est.fit(x_train, y_train)
            neural_props = est.score(x_test, y_test)
            print(neural_props)
            break
        except NanException as e:
            model.neural_config.learning_rate *= 0.9
            lr = "{:e}".format(model.neural_config.learning_rate)
            print(f"New Learning Rate: {lr}")
