from dependency_injector import containers, providers

from repositories.NeuralModelRepository import NeuralModelRepository
from repositories.SampleDatasetRepository import SampleDatasetRepository


class JobContainer(containers.DeclarativeContainer):
    print("Initialising dependencies...")
    path = "resources/config.ini"
    config = providers.Configuration(ini_files=[path])

    # Repositories

    neural_repository = providers.Singleton(NeuralModelRepository, uri=config.database.uri)
    sample_repository = providers.Singleton(SampleDatasetRepository, uri=config.database.uri)

    print("Dependencies ready.")