from dataclasses import dataclass
from typing import Dict

from cluster.Job import Job
from data.Dataset import Dataset
from models.Estimator import Estimator
from models.GridSearch import GridSearch
from repositories.NeuralModelRepository import NeuralModelRepository


@dataclass
class ModelJob(Job):
    function: str
    dataset: Dataset
    hyper_params: Dict
    _repository: NeuralModelRepository

    def run(self):
        # np.exp(numpy.linspace(np.log(10E-4), np.log(10E-6), 3))

        searcher = GridSearch()
        config_pool = searcher.get_sequence(self.hyper_params)

        estimator_pool = [
            Estimator(self._repository, name=self.function, config=config, epochs=5, should_save=True)
            for config in config_pool
        ]

        x_train, y_train = self.dataset.train
        x_test, y_test = self.dataset.train

        for estimator in estimator_pool:
            estimator.fit(x_train, y_train)
            estimator.score(x_test, y_test)





if __name__ == '__main__':  # Prepare this to be used as job trigger-able
    pass
