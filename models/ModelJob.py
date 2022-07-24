import sys
from time import sleep
from typing import Optional

from cluster.Job import Job, JobType
from cluster.JobContainer import JobContainer
from cluster.JobInit import init_container
from data.Dataset import Dataset
from experiments.Experiment import NeuralType
from models.Estimator import Estimator
from models.FNN import FNN
from models.LoadableModule import LoadableModule
from models.Trainer import Trainer
from repositories.NeuralModelRepository import NeuralModelRepository
from repositories.SampleDatasetRepository import SampleDatasetRepository
from repositories.db_models import FeedforwardNeuralConfig, NeuralProperties, NeuralModel, ConvolutionalNeuralConfig, \
    NeuralConfig


class ModelJob(Job):

    def __init__(self, dataset_id: str, config: NeuralConfig, experiment_id: str):
        super().__init__(experiment_id)
        self.dataset_id = dataset_id
        self.config = config

        self.function_name: Optional[str] = None
        self.sample_repo: Optional[SampleDatasetRepository] = None
        self.neural_repo: Optional[NeuralModelRepository] = None

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
            estimator = Estimator(name=self.function_name, config=self.config, epochs=100)
            trainer = estimator.fit(x_train, y_train)
            neural_props = estimator.score(x_test, y_test)
            neural_model_id = self.save_model(trainer, neural_props)
        else:
            neural_model_id = model_result.id
        print(f"NEURAL_MODEL_ID:{neural_model_id}", end="")
        sys.stdout.flush()
        sleep(0.5)

    def load(self) -> Optional[LoadableModule]:
        existing_models = self.neural_repo.get_by_config(self.function_name, self.config, self.experiment_id)
        model = existing_models[0]
        return FNN(
            self.config.network_size, self.config.depth, self.config.activation_fn
        ).load_bytes(model.model_data)

    def search_existing(self) -> Optional[NeuralModel]:
        existing_models = self.neural_repo.get_by_config(self.function_name, self.config, self.experiment_id)
        if len(existing_models) != 0:
            return existing_models[0]
        else:
            return None

    def save_model(self, trainer: Trainer, props: NeuralProperties) -> str:
        if type(self.config) is FeedforwardNeuralConfig:
            neural_type: NeuralType = "Feedforward"
        elif type(self.config) is ConvolutionalNeuralConfig:
            neural_type: NeuralType = "Convolutional"
        else:
            raise RuntimeError("Unrecognized Network Type")
        model = NeuralModel(
            function=self.function_name,
            type=neural_type,
            neural_config=self.config,
            neural_properties=props,
            model_data=trainer.get_model_data(),
            experiment_id=self.experiment_id
        )
        return self.neural_repo.save(model)

    def as_command(self) -> str:
        return f"python3 -m models.ModelJob --job {self.encode()}"

    def get_job_type(self) -> JobType:
        return "GPU"


if __name__ == '__main__':  # Prepare this to be used as job trigger-
    repo = SampleDatasetRepository(uri="mongodb://cloud-vm-42-88.doc.ic.ac.uk:27017/")
    dataset_id = repo.get_all_dataset_id()[0]
    config = FeedforwardNeuralConfig(1E7, 128, 420, 2, "ReLU")
    job = ModelJob(dataset_id, config, "neural-test")
    container = init_container()
    job.run(container)
    #init_job("ModelJob")
