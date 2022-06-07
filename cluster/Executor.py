from abc import ABC, abstractmethod
from typing import List

from cluster.Job import Job


class Executor(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def _get_jobs(self) -> List[Job]:
        pass

    def run_all_jobs(self):
        for job in self._get_jobs():
            job.run()
