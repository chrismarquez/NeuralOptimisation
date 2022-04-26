import math
import os

import numpy as np
import pandas as pd
import sklearn.metrics
import torch
from sklearn.model_selection import GridSearchCV
from torch import nn
from tqdm import tqdm

from sklearn import metrics

from models.Estimator import Estimator
from dataset import Dataset
from models.FNN import FNN
from models.Regressor import Regressor
from models.Trainer import Trainer
from optimisation.Optimiser import Optimiser


def train(dataset: Dataset):
    trainer = Trainer(FNN.instantiate())
    x_train, y_train = dataset.train
    trained_net = trainer.train(x_train, y_train, epochs=400)
    trainer.save("test", "sum_squares")
    return trained_net


def hyperparameter_search(x_train: np.ndarray, y_train: np.ndarray, hyper_params, name):
    # np.exp(numpy.linspace(np.log(10E-4), np.log(10E-6), 3))
    estimator = Estimator(name=name)
    searcher = GridSearchCV(estimator, param_grid=hyper_params, scoring=Estimator.score, cv=2, n_jobs=2, verbose=10)
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


def train_all_models():

    sizes_2 = [int(math.sqrt(2 * 10_000 * i + 15) - 4) for i in range(1, 10)]
    sizes_4 = [int(math.sqrt(32.0/21.0 * 10_000 * i - 32.0/21.0 + (64.0/21.0) ** 2) - 64.0/21.0) for i in range(1, 9)]

    sizes_2 = [(it, 2) for it in sizes_2]
    sizes_4 = [(it, 4) for it in sizes_4]

    hyper_params = {
        "learning_rate": [1E-6, 3E-7],  # Evenly spaced lr in log scale
        "batch_size": [128, 512],
        "network_config": sizes_2 + sizes_4,
        "activation_fn": ["ReLU", "Sigmoid"],
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
    #train_all_models()
    # main()
    # raw_dataset = np.loadtxt(f"samples/sum_squares.csv", delimiter=",")
    # dataset = Dataset.create(raw_dataset)
    # train(dataset)
    optimise_all_models()
