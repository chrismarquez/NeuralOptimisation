

from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject

from cluster.Cluster import Cluster
from models.ModelsExecutor import ModelsExecutor
from optimisation.OptimisationExecutor import OptimisationExecutor
from repositories.NeuralModelRepository import NeuralModelRepository
from repositories.SampleDatasetRepository import SampleDatasetRepository

from constants import ROOT_DIR, get_env, get_config


# TODO: Put relevant Containers to dependency inject both jobs inside cluster and the main driver program

class Container(containers.DeclarativeContainer):
    print("Initialising dependencies...")

    env = get_env()
    config_file = get_config(env)
    path = f"resources/{config_file}"
    config = providers.Configuration(ini_files=[path])

    # Cluster

    cluster = providers.Singleton(Cluster, root_dir=ROOT_DIR)

    # Repositories

    neural_repository = providers.Singleton(NeuralModelRepository, uri=config.database.uri)
    sample_repository = providers.Singleton(SampleDatasetRepository, uri=config.database.uri)

    # Executors

    optimisation_executor = providers.Singleton(OptimisationExecutor, repository=neural_repository)
    models_executor = providers.Singleton(ModelsExecutor, neural_repo=neural_repository, sample_repo=sample_repository)

    print("Dependencies ready.")


@inject
def main(container: Container = Provide[Container]):
    #optimisation_executor = container.optimisation_executor()
    #optimisation_executor.run_all_jobs()
    #print("Hello there")
    cluster = container.cluster()
    config = cluster.get_job_config()
    print(config)
    cluster.exec(config)


if __name__ == '__main__':
    print("Loading Container...")
    container = Container()
    container.init_resources()
    container.wire(modules=[__name__])
    print("Container ready.")
    main()
