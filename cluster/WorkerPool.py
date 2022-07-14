import inspect
import tempfile
from abc import ABC, abstractmethod
from asyncio import Queue
from typing import Awaitable

from cluster.Job import Job, JobType


class WorkerPool(ABC):

    def __init__(self, capacity: int, root_dir: str):
        self.root_dir = root_dir
        self.capacity = capacity
        self._slots_queue: Queue[int] = Queue(maxsize=capacity)

    async def _request_slot(self):
        await self._slots_queue.put(0)

    async def _release_slot(self):
        await self._slots_queue.get()
        self._slots_queue.task_done()

    def _runnable_script_from(self, job: Job) -> str:
        cmd = inspect.cleandoc(
            f"""
                #!/bin/bash
                source {self.root_dir}/venv/bin/activate
                APP_ENV=PROD {job.as_command()}
            """
        )
        return self._write_script_file(cmd)

    def _write_script_file(self, content: str, suffix=".sh") -> str:
        path = f"{self.root_dir}/temp"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w", dir=path) as file:
            file.write(content)
            script = file.name
            print(script)
            return script

    @abstractmethod
    async def submit(self, job: Job) -> Awaitable[str]:
        pass

    @abstractmethod
    def job_type(self) -> JobType:
        pass
