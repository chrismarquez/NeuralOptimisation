import sys
from time import sleep
from typing import Optional

from cluster.Job import Job, JobType, UnnecessaryJobException
from cluster.JobContainer import JobContainer
from cluster.JobInit import init_job
from data.Dataset import Dataset
from models.Estimator import Estimator
from models.Trainer import Trainer
from repositories.NeuralModelRepository import NeuralModelRepository
from repositories.SampleDatasetRepository import SampleDatasetRepository
from repositories.db_models import NeuralProperties, NeuralModel, NeuralConfig


class ModelJob(Job):

    def __init__(self, model_id: str, config: NeuralConfig):
        super().__init__(model_id)

        self.config = config

        self.sample_repo: Optional[SampleDatasetRepository] = None
        self.neural_repo: Optional[NeuralModelRepository] = None

    def _pre_run(self, container: JobContainer) -> (Dataset, NeuralModel):
        self.neural_repo = container.neural_repository()
        self.sample_repo = container.sample_repository()
        model = self.neural_repo.get(self.model_id)

        if model.neural_properties is not None:
            raise UnnecessaryJobException()

        dataset_id = self.sample_repo.get_id_by_name(model.function)
        sample_dataset = self.sample_repo.get(dataset_id)
        return sample_dataset.to_dataset(), model

    def _run(self, container: JobContainer):
        dataset, model = self._pre_run(container)
        x_train, y_train = dataset.train
        x_test, y_test = dataset.test
        epochs = int(container.config.training.epochs())
        estimator = Estimator(name=model.function, config=self.config, epochs=epochs)
        trainer = estimator.fit(x_train, y_train)
        neural_props = estimator.score(x_test, y_test)
        self.save_neural_props(trainer, neural_props)
        print(f"NEURAL_MODEL_ID:{self.model_id}", end="")
        sys.stdout.flush()
        sleep(0.5)

    def save_neural_props(self, model: NeuralModel, trainer: Trainer, props: NeuralProperties):
        model.neural_properties = props
        model.model_data = trainer.get_model_data()
        self.neural_repo.update(model)

    def as_command(self) -> str:
        return f"python3 -m models.ModelJob --job {self.encode()}"

    def get_job_type(self) -> JobType:
        return "GPU"


if __name__ == '__main__':  # Prepare this to be used as job trigger-
    init_job("ModelJob")
