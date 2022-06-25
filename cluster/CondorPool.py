import asyncio
import inspect
from dataclasses import dataclass
from typing import List, Awaitable

from cluster.Job import JobType, Job
from cluster.WorkerPool import WorkerPool
import paramiko


@dataclass
class CondorConfig:
    server_list: List[str]
    job_type: JobType


class CondorPool(WorkerPool):

    def __init__(self, root_dir: str,capacity: int, condor_server: str, config: CondorConfig):
        super().__init__(capacity)
        self.root_dir = root_dir
        self.config = config
        self.condor_server = condor_server

        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.load_system_host_keys()
        self.ssh_client.connect(self.condor_server)


    async def submit(self, job: Job) -> Awaitable[str]:
        pass

    async def _submit(self, job: Job):
        await self._request_slot()
        cmd = inspect.cleandoc(
            f"""
                #!/bin/bash
                source {self.root_dir}/venv/bin/activate
                APP_ENV=PROD {job.as_command()}
            """
        )
        script = WorkerPool._write_script_file(cmd)
        condor_submit = f"condor_submit {script}"
        stdin, stdout, stderr = self.ssh_client.exec_command(condor_submit)

    def test(self):
        stdin, stdout, stderr = self.ssh_client.exec_command("ls /usr/local/condor")
        print(str(stdin.read()))

    def job_type(self) -> JobType:
        return self.config.job_type


if __name__ == '__main__':
    config = CondorConfig([], "CPU")
    pool = CondorPool("/vol/bitbucket/csm21/NeuralOptimisation", 2, "shell1.doc.ic.ac.uk", config)
    pool.test()