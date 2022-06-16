from dataclasses import dataclass

from cluster.Job import Job, JobType
from models.Estimator import Estimator
from repositories.db_models import NeuralConfig

from cluster.JobContainer import JobContainer
from cluster.JobInit import init_job


@dataclass
class ModelJob(Job):
    dataset_id: str
    config: NeuralConfig

    def run(self, container: JobContainer):
        neural_repo = container.neural_repository()
        sample_repo = container.sample_repository()

        # np.exp(numpy.linspace(np.log(10E-4), np.log(10E-6), 3))
        sample_dataset = sample_repo.get(self.dataset_id)
        function_name = sample_dataset.function
        dataset = sample_dataset.to_dataset()

        x_train, y_train = dataset.train
        x_test, y_test = dataset.test

        estimator = Estimator(neural_repo, name=function_name, config=self.config, epochs=5, should_save=True)
        estimator.fit(x_train, y_train)
        estimator.score(x_test, y_test)

    def as_command(self) -> str:
        return f"python3 -m models.ModelJob --job {self.encode()}"

    def get_job_type(self) -> JobType:
        return "GPU"


if __name__ == '__main__':  # Prepare this to be used as job trigger-
    init_job("ModelJob")
