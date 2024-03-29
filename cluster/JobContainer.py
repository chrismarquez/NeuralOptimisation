from dependency_injector import containers, providers

from constants import get_config, get_env, ROOT_DIR
from repositories.NeuralModelRepository import NeuralModelRepository
from repositories.SampleDatasetRepository import SampleDatasetRepository


class JobContainer(containers.DeclarativeContainer):

    env = get_env()
    config_file = get_config(env)
    path = f"{ROOT_DIR}/resources/{config_file}"
    config = providers.Configuration(ini_files=[path])

    # Repositories

    neural_repository = providers.Singleton(NeuralModelRepository, uri=config.database.uri)
    sample_repository = providers.Singleton(SampleDatasetRepository, uri=config.database.uri)