import inspect
from dataclasses import dataclass
from typing import List, Awaitable

from cluster.Job import JobType, Job
from cluster.WorkerPool import WorkerPool
import paramiko

CONDOR_PATH = "/usr/local/condor/release/bin"


@dataclass
class CondorConfig:
    server_list: List[str]
    job_type: JobType


class CondorPool(WorkerPool):

    def __init__(self, root_dir: str, capacity: int, condor_server: str, config: CondorConfig):
        super().__init__(capacity, root_dir)
        self.config = config
        self.condor_server = condor_server

        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.load_system_host_keys()
        self.ssh_client.connect(self.condor_server, username="csm21")

    async def submit(self, job: Job) -> Awaitable[str]:
        pass

    async def _submit(self, job: Job):
        await self._request_slot()
        script = self._to_runnable_script(job)
        condor_spec = self.get_condor_spec(script)
        condor_submit = f"{CONDOR_PATH}/condor_submit {condor_spec}"
        try:
            stdin, stdout, stderr = self.ssh_client.exec_command(condor_submit)
        except ValueError:
            await self._release_slot()

    def get_condor_spec(self, script: str) -> str:
        cmd = inspect.cleandoc(
            f"""
                universe = vanilla
                executable = {script}
                output = uname.$(Process).out
                error = uname.$(Process).err
                log = uname.log
                queue
            """
        )
        return self._write_script_file(cmd, suffix=".cmd")

    def test(self):
        stdin, stdout, stderr = self.ssh_client.exec_command(f"{CONDOR_PATH}/condor_submit {self.root_dir}/test.job")
        print(str(stdout.read()))

    def job_type(self) -> JobType:
        return self.config.job_type


if __name__ == '__main__':
    config = CondorConfig([], "CPU")
    pool = CondorPool("/vol/bitbucket/csm21/NeuralOptimisation", 2, "shell1.doc.ic.ac.uk", config)
    pool.test()
