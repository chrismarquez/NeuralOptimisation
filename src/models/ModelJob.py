from typing import Dict

import numpy as np
from sklearn.model_selection import GridSearchCV

from src.cluster.Job import Job
from src.models.Estimator import Estimator


def hyperparameter_search(x_train: np.ndarray, y_train: np.ndarray, hyper_params: Dict, function: str):
    # np.exp(numpy.linspace(np.log(10E-4), np.log(10E-6), 3))
    estimator = Estimator(name=function)
    searcher = GridSearchCV(estimator, param_grid=hyper_params, scoring=Estimator.score, cv=2, n_jobs=2, verbose=10)
    searcher.fit(x_train, y_train)


class ModelJob(Job):

    def __init__(self, function: str, x_train: np.ndarray, y_train: np.ndarray, hyper_params):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.hyper_params = hyper_params
        self.function = function

    def run(self):
        hyperparameter_search(self.x_train, self.y_train, self.hyper_params, self.function)


if __name__ == '__main__':  # Prepare this to be used as job trigger-able
    pass
