import argparse
import asyncio
from typing import cast

from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject

from cluster.Cluster import Cluster
from constants import ROOT_DIR, get_env, get_config
from experiments.Experiment import Experiment, NeuralType
from experiments.ExperimentExecutor import ExperimentExecutor
from repositories.NeuralModelRepository import NeuralModelRepository
from repositories.SampleDatasetRepository import SampleDatasetRepository


class Container(containers.DeclarativeContainer):
    print("Initialising dependencies...")

    env = get_env()
    config_file = get_config(env)
    path = f"resources/{config_file}"
    config = providers.Configuration(ini_files=[path])

    # Cluster

    cluster = providers.Singleton(
        Cluster,
        root_dir=ROOT_DIR,
        condor_server=config.condor_server.uri,
        raw_debug=config.log.debug
    )

    # Repositories

    neural_repository = providers.Singleton(NeuralModelRepository, uri=config.database.uri)
    sample_repository = providers.Singleton(SampleDatasetRepository, uri=config.database.uri)

    # Executors

    experiment_executor = providers.Singleton(
        ExperimentExecutor,
        cluster=cluster,
        neural_repo=neural_repository,
        sample_repo=sample_repository
    )

    print("Dependencies ready.")


@inject
async def main(experiment: Experiment, test_run: bool, container: Container = Provide[Container]):
    executor = container.experiment_executor()
    await executor.run_experiment(experiment, test_run=test_run)


if __name__ == '__main__':
    print("Loading Container...")
    container = Container()
    container.init_resources()
    container.wire(modules=[__name__])
    print("Container ready.")

    parser = argparse.ArgumentParser(description='Neural Optimisation Experiment Runner')

    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        help=f"ID of the Experiment to Run"
    )

    parser.add_argument(
        '--type',
        type=str,
        choices=["Feedforward", "Convolutional"],
        required=True,
        help=f"Type of Network to Use"
    )

    parser.add_argument(
        '--test',
        action="store_true",
        help=f"Run only a few examples to perform a quick test"
    )

    args = parser.parse_args()

    experiment = Experiment(args.experiment, cast(NeuralType, args.type))

    asyncio.run(main(experiment, args.test))
