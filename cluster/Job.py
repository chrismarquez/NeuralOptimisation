from __future__ import annotations

import base64
import pickle
from abc import ABC, abstractmethod
from typing import Literal

from cluster.JobContainer import JobContainer

JobType = Literal["GPU", "CPU"]

class Job(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def run(self, container: JobContainer):
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