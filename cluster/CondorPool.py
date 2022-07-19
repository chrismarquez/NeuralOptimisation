import asyncio
import inspect
from asyncio import Future
from dataclasses import dataclass
from typing import List, Awaitable, Optional

from cluster.CondorJobStatus import CondorJobStatus
from cluster.Job import JobType, Job
from cluster.WorkerPool import WorkerPool
import paramiko

CONDOR_PATH = "/usr/local/condor/release/bin"


@dataclass
class CondorConfig:
    user: str
    job_type: JobType


class CondorPool(WorkerPool):

    @staticmethod
    def _parse_job_id(result: List[str]) -> str:
        raw_job_id = result[-1].rstrip("\\n").split("submitted to cluster ")[-1]
        return CondorJobStatus.format_job_id(raw_job_id)

    def __init__(self, root_dir: str, capacity: int, condor_server: str, config: CondorConfig):
        super().__init__(capacity, root_dir)
        self.config = config
        self.condor_server = condor_server

        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.load_system_host_keys()
        self.ssh_client.connect(self.condor_server, username=config.user)

    async def submit(self, job: Job) -> Awaitable[str]:
        condor_job_id = await self._submit(job)
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        task = self._post_process(future, condor_job_id)
        asyncio.create_task(task)
        return future

    async def _post_process(self, future: Future[str], condor_job_id: str):
        while True:
            status = await self.status(condor_job_id)
            print(f"Job Status {condor_job_id}: {status}")
            if status is None:
                break
            await asyncio.sleep(3)
        if self.job_type() == "CPU":
            lines = self.get_job_output(condor_job_id)
            model_id = CondorPool.find_model_id(lines)
            future.set_result(model_id)
        else:
            future.set_result(str(condor_job_id))

    async def _submit(self, job: Job) -> str:
        await self._request_slot()
        script = self._runnable_script_from(job)
        condor_spec = self.get_condor_spec(script)
        condor_submit = f"{CONDOR_PATH}/condor_submit {condor_spec}"
        try:
            _, stdout, _ = self.ssh_client.exec_command(condor_submit)
            lines = stdout.readlines()
            return CondorPool._parse_job_id(lines)
        except ValueError:
            await self._release_slot()

    async def status(self, job_id: str) -> Optional[CondorJobStatus]:
        cmd = f"{CONDOR_PATH}/condor_q {self.config.user} -nobatch"
        _, stdout, _ = self.ssh_client.exec_command(cmd)
        log = stdout.read().decode("utf-8")
        job_status_list = CondorJobStatus.from_log(log)
        filtered_status = [status for status in job_status_list if status.job_id == job_id]
        if len(filtered_status) == 0:
            return None
        else:
            return filtered_status[0]

    def get_job_output(self, job_id: int) -> List[str]:
        file = f"~/uname.{job_id}.out"
        with open(file) as f:
            return f.readlines()

    def get_condor_spec(self, script: str) -> str:
        cmd = inspect.cleandoc(
            f"""
                universe = vanilla
                executable = {script}
                output = uname.$(Process).out
                error = uname.$(Process).err
                Requirements = {self._get_node_req()}
                log = uname.log
                queue
            """
        )
        return self._write_script_file(cmd, suffix=".cmd")

    def _get_node_req(self):
        if self.config.job_type == "GPU":
            return """regexp("^(gpu)[0-9][0-9]", TARGET.Machine) == True"""
        elif self.config.job_type == "CPU":
            return """regexp("^(ray)0[1-8]", TARGET.Machine) == True"""

    def test(self):
        _, stdout, _ = self.ssh_client.exec_command(f"{CONDOR_PATH}/condor_submit {self.root_dir}/test.job")
        print(str(stdout.read()))

    def job_type(self) -> JobType:
        return self.config.job_type


if __name__ == '__main__':
    config = CondorConfig("csm21", "CPU")
    pool = CondorPool("/vol/bitbucket/csm21/NeuralOptimisation", 2, "shell1.doc.ic.ac.uk", config)

