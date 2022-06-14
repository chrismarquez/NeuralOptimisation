import sys
import argparse
from dataclasses import dataclass

from dependency_injector.wiring import Provide, inject

from cluster.Job import Job
from cluster.JobContainer import JobContainer
from models.Estimator import Estimator
from repositories.db_models import NeuralConfig


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


@inject
def main(encoded_job: str, container: JobContainer = Provide[JobContainer]):
    job = Job.decode(encoded_job)
    job.run(container)


if __name__ == '__main__':  # Prepare this to be used as job trigger-

    container = JobContainer()
    container.init_resources()
    container.wire(modules=[__name__])

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        '--job',
        type=str,
        help="ModelJob encoded as b64 pickle"
    )
    args = parser.parse_args()
    main(args.job)
