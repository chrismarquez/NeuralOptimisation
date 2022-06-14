
from dependency_injector import containers, providers

from constants import get_config, get_env
from repositories.NeuralModelRepository import NeuralModelRepository
from repositories.SampleDatasetRepository import SampleDatasetRepository


class JobContainer(containers.DeclarativeContainer):

    print("Initialising dependencies...")

    env = get_env()
    config_file = get_config(env)
    print(f"Environment: {env}. Using config file: {config_file}")

    path = f"../resources/{config_file}"
    config = providers.Configuration(ini_files=[path])

    # Repositories

    neural_repository = providers.Singleton(NeuralModelRepository, uri=config.database.uri)
    sample_repository = providers.Singleton(SampleDatasetRepository, uri=config.database.uri)

    print("Dependencies ready.")
