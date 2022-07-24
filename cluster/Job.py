from __future__ import annotations

import base64
import pickle
import uuid
from abc import ABC, abstractmethod
from typing import Literal

from cluster.JobContainer import JobContainer

JobType = Literal["GPU", "CPU"]


class UnnecessaryJobException(Exception):
    pass


class Job(ABC):

    def __init__(self, model_id: str):
        self.uuid: str = str(uuid.uuid4())
        self.model_id = model_id
        pass

    def run(self, container: JobContainer):
        try:
            self._run(container)
        except UnnecessaryJobException:
            print("Job is unnecessary")

    @abstractmethod
    def _run(self, container: JobContainer):
        pass

    @abstractmethod
    def as_command(self) -> str:
        pass

    @abstractmethod
    def get_job_type(self) -> JobType:
        pass

    def encode(self) -> str:
        return base64.b64encode(pickle.dumps(self)).decode()

    @staticmethod
    def decode(encoded_job: str) -> Job:
        return pickle.loads(base64.b64decode(encoded_job.encode()))

    def requires_gurobi_license(self) -> bool:
        return False
