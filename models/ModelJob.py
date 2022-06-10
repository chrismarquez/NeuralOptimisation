import sys
from dataclasses import dataclass

from cluster.Job import Job
from models.Estimator import Estimator
from repositories.NeuralModelRepository import NeuralModelRepository
from repositories.SampleDatasetRepository import SampleDatasetRepository
from repositories.db_models import NeuralConfig


@dataclass
class ModelJob(Job):
    dataset_id: str
    config: NeuralConfig
    _neural_repo: NeuralModelRepository
    _sample_repo: SampleDatasetRepository

    def run(self):
        # np.exp(numpy.linspace(np.log(10E-4), np.log(10E-6), 3))
        sample_dataset = self._sample_repo.get(self.dataset_id)
        function_name = sample_dataset.function
        dataset = sample_dataset.to_dataset()

        x_train, y_train = dataset.train
        x_test, y_test = dataset.test

        estimator = Estimator(self._neural_repo, name=function_name, config=self.config, epochs=5, should_save=True)
        estimator.fit(x_train, y_train)
        estimator.score(x_test, y_test)


if __name__ == '__main__':  # Prepare this to be used as job trigger-able
    args = sys.argv
    print("Invoked by sbatch")
