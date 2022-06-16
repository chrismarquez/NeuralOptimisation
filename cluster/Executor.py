from abc import ABC, abstractmethod
from typing import List

from cluster.Cluster import Cluster
from cluster.Job import Job
from cluster.JobContainer import JobContainer
from constants import ROOT_DIR


class Executor(ABC):

    def __init__(self):
        self.cluster = Cluster(ROOT_DIR)
        pass

    @abstractmethod
    def _get_jobs(self) -> List[Job]:
        pass

    def run_all_jobs(self, use_cluster: bool = True, test_run: bool = False):
        jobs = self._get_jobs()
        if test_run:
            jobs = [jobs[0]]
        if use_cluster:
            self._run_at_cluster(jobs)
        else:
            Executor._run_locally(jobs)

    def _run_at_cluster(self, jobs: List[Job]):
        for job in jobs:
            self.cluster.submit(job)

    @staticmethod
    def _run_locally(jobs: List[Job]):
        container = JobContainer()
        container.init_resources()
        container.wire(modules=[__name__])
        for job in jobs:
            print(job.encode())
            job.run(container)
