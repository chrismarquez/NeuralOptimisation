import inspect
import os
import stat
import tempfile
from abc import ABC, abstractmethod
from asyncio import Queue
from typing import Awaitable, List

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
                #!/bin/zsh
                cd {self.root_dir}
                source ~/.zshrc
                source {self.root_dir}/venv/bin/activate
                GRB_LICENSE_FILE="/homes/csm21/gurobi_licenses/$(hostname)/gurobi.lic" APP_ENV=PROD {job.as_command()}
            """
        )
        return self._write_script_file(cmd)

    def _write_script_file(self, content: str, suffix=".sh") -> str:  # Returns the file name of the created script
        path = f"{self.root_dir}/temp"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w", dir=path) as file:
            file.write(content)
            script = file.name
        st = os.stat(script)
        os.chmod(script, st.st_mode | stat.S_IEXEC)
        return script

    @staticmethod
    def find_model_id(lines: List[str]) -> str:
        return lines[-1].split("NEURAL_MODEL_ID:")[-1].removesuffix("\\n")

    @abstractmethod
    async def submit(self, job: Job) -> Awaitable[bool]:
        pass

    @abstractmethod
    def job_type(self) -> JobType:
        pass
