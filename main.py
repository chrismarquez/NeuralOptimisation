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
        "learning_rate": [1E-4, 3E-5],  # Evenly spaced lr in log scale
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


def finished_optimisations(function: str):
    filename = f"trained/optimisation/{function}.csv"
    df = pd.read_csv(filename, delimiter=',')
    return list(df['id'].values)


def optimise_all_models():
    path = "trained/metadata"
    input_bounds = {
        i: (-0.2, 0.2) for i in range(2)
    }
    activations = {
        "ReLU": lambda: nn.ReLU(),
        "Tanh": lambda: nn.Tanh()
    }
    for file in os.listdir(path):
        function, _ = file.split(".")
        finished = finished_optimisations(function)
        df = pd.read_csv(f"{path}/{file}", delimiter=",")
        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            id, _, _, nodes, depth, activation_fn, _, _ = row
            if id not in finished and depth == 2:
                net = FNN(nodes, depth, activations[activation_fn])
                optimiser = Optimiser.load(f"trained/{function}/{id}.pt", input_bounds, lambda: net)
                x_opt, y_opt, z_opt = optimiser.solve()
                location_error = metrics.mean_squared_error([0.0, 0.0], [x_opt, y_opt], squared=False)
                optimum_error = metrics.mean_squared_error([0.0], [z_opt], squared=False)
                computation_time = optimiser.optimisation_time
                filename = f"trained/optimisation/{function}.csv"
                if not os.path.exists(filename):
                    with open(filename, 'a') as f:
                        f.write("id,x,y,location_error, optimum_error, computation_time\n")
                with open(filename, 'a') as f:
                    model_params = f"{id},{x_opt},{y_opt},{location_error},{optimum_error},{computation_time}\n"
                    f.write(model_params)
            else:
                print(f"Skip {id}")


if __name__ == '__main__':
    train_all_models()
    # main()
    # raw_dataset = np.loadtxt(f"samples/sum_squares.csv", delimiter=",")
    # dataset = Dataset.create(raw_dataset)
    # train(dataset)
    # optimise_all_models()
