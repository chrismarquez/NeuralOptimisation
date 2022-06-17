from dataclasses import dataclass
from typing import Optional

from cluster.Job import Job, JobType
from data.Dataset import Dataset
from models.Estimator import Estimator
from models.FNN import FNN
from models.LoadableModule import LoadableModule
from models.Trainer import Trainer
from repositories.db_models import NeuralConfig, NeuralProperties, NeuralModel

from cluster.JobContainer import JobContainer
from cluster.JobInit import init_job


@dataclass
class ModelJob(Job):
    dataset_id: str
    config: NeuralConfig

    def __init__(self, uuid):
        super().__init__(uuid)
        self.function_name = None
        self.sample_repo = None
        self.neural_repo = None

    def _pre_run(self, container: JobContainer) -> Dataset:
        self.neural_repo = container.neural_repository()
        self.sample_repo = container.sample_repository()

        # np.exp(numpy.linspace(np.log(10E-4), np.log(10E-6), 3))
        sample_dataset = self.sample_repo.get(self.dataset_id)
        self.function_name = sample_dataset.function
        return sample_dataset.to_dataset()

    def run(self, container: JobContainer):
        dataset = self._pre_run(container)
        model_result = self.search_existing()
        if model_result is None:
            x_train, y_train = dataset.train
            x_test, y_test = dataset.test
            estimator = Estimator(name=self.function_name, config=self.config, epochs=5)
            trainer = estimator.fit(x_train, y_train)
            neural_props = estimator.score(x_test, y_test)
            neural_model_id = self.save_model(trainer, neural_props)
        else:
            neural_model_id = model_result.id
        print(f"NEURAL_MODEL_ID:{neural_model_id}")

    def load(self) -> Optional[LoadableModule]:
        existing_models = self.neural_repo.get_by_config(self.function_name, self.config)
        model = existing_models[0]
        return FNN(
            self.config.network_size, self.config.depth, self.config.activation_fn
        ).load_bytes(model.model_data)

    def search_existing(self) -> Optional[NeuralModel]:
        existing_models = self.neural_repo.get_by_config(self.function_name, self.config)
        if len(existing_models) != 0:
            return existing_models[0]
        else:
            return None

    def save_model(self, trainer: Trainer, props: NeuralProperties) -> str:
        model = NeuralModel(self.function_name, self.config, props, trainer.get_model_data())
        return self.neural_repo.save(model)

    def as_command(self) -> str:
        return f"python3 -m models.ModelJob --job {self.encode()}"

    def get_job_type(self) -> JobType:
        return "GPU"


if __name__ == '__main__':  # Prepare this to be used as job trigger-
    init_job("ModelJob")
