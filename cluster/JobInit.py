import argparse

from dependency_injector.wiring import inject, Provide

from cluster.Job import Job
from cluster.JobContainer import JobContainer


@inject
def main(encoded_job: str, container: JobContainer = Provide[JobContainer]):
    job = Job.decode(encoded_job)
    job.run(container)


def init_container() -> JobContainer:
    container = JobContainer()
    container.init_resources()
    container.wire(modules=[__name__])
    return container


def init_job(job_type: str):
    init_container()
    parser = argparse.ArgumentParser(description='Runnable job task')
    parser.add_argument(
        '--job',
        type=str,
        help=f"{job_type} encoded as b64 pickle"
    )
    args = parser.parse_args()
    main(args.job)
