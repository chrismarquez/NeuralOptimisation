import os

import numpy as np
import torch
from sklearn.model_selection import GridSearchCV, ParameterGrid

from Estimator import Estimator
from dataset import Dataset
from models.FNN import FNN
from models.Regressor import Regressor
from models.Trainer import Trainer


def train(dataset):
    trainer = Trainer(FNN())
    x_train, y_train = dataset.train
    trained_net = trainer.train(x_train, y_train, 400)
    trainer.save("sum_squares")
    return trained_net


def hyperparameter_search(x_train: np.ndarray, y_train: np.ndarray, hyper_params, name):
    # np.exp(numpy.linspace(np.log(10E-4), np.log(10E-6), 3))

    estimator = Estimator(name=name)
    searcher = GridSearchCV(estimator, param_grid=hyper_params, scoring=Estimator.score, cv=2, n_jobs=4, verbose=10)
    searcher.fit(x_train, y_train)


def test(trained_net, dataset):
    predictor = Regressor(trained_net)
    x = np.array([[1.0, 2.0]])
    test = torch.tensor(x).double()

    out = predictor.predict(test)
    # real = functions.sum_squares(x[:, 0], x[:, 1])

    print(out)

    x_dev, y_dev = dataset.dev
    dev_error = predictor.evaluate(torch.tensor(x_dev).double(), torch.tensor(y_dev).double())
    print(dev_error)


def main():
    hyper_params = {
        "learning_rate": [1E-4, 1E-5, 1E-6],  # Evenly spaced lr in log scale
        "batch_size": [128, 512, 2048],
        "network_size": [25, 50, 75],
        "depth": [2, 3, 4],
        "activation_fn": ["ReLU", "Tanh"],
        "should_save": [True]
    }

    dummy_params = {
        "learning_rate": [1E-4],  # Evenly spaced lr in log scale
        "batch_size": [2048],
        "network_size": [25],
        "depth": [2],
        "activation_fn": ["ReLU", "Tanh"],
        "should_save": [True]
    }

    for file in os.listdir("samples/"):
        raw_dataset = np.loadtxt(f"samples/{file}", delimiter=",")
        dataset = Dataset.create(raw_dataset)
        x_train, y_train = dataset.train

        name, ext = file.split(".")

        print(f"Computing params of function: {name}")

        hyperparameter_search(x_train, y_train, hyper_params, name)


if __name__ == '__main__':
    main()
