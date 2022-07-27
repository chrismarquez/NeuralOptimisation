import asyncio
from asyncio import Queue, Future, Task
from typing import Awaitable, List, Mapping, Tuple, Optional

import paramiko

from cluster.CondorPool import CondorPool, CondorConfig
from cluster.Job import Job, JobType
from cluster.SlurmPool import SlurmPool
from cluster.WorkerPool import WorkerPool


class Cluster:

    def __init__(self, root_dir, condor_server: str, raw_debug: str):
        user = "csm21"
        self.root_dir = root_dir
        debug = raw_debug == "True"
        self.pools: List[WorkerPool] = [
            SlurmPool(root_dir, capacity=2, debug=debug),
            CondorPool(root_dir, capacity=40, condor_server=condor_server, config=CondorConfig(user, "CPU", debug)),
            CondorPool(root_dir, capacity=25, condor_server=condor_server, config=CondorConfig(user, "GPU", debug))
        ]

        print("[Cluster] Connecting to Shell Server.")
        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.load_system_host_keys()
        self.ssh_client.connect(condor_server, username=user)
        print("Connected.")

        self.consumers: List[Task] = []
        self.kerberos_request: Optional[Task] = None

        self.type_queues: Mapping[JobType, Queue[Tuple[Future[bool], Job]]] = {
            "CPU": Queue(),
            "GPU": Queue()
        }

    # Coroutine launch method
    async def submit(self, job: Job) -> Awaitable[bool]:
        job_type = job.get_job_type()
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self.type_queues[job_type].put((future, job))
        return future

    def start(self):
        for pool in self.pools:
            consumer = asyncio.create_task(self._consume(pool))
            self.consumers.append(consumer)
        self.kerberos_request = asyncio.create_task(self._request_kerberos_ticket())

    def stop(self):
        for consumer in self.consumers:
            consumer.cancel()
        self.kerberos_request.cancel()

    async def _consume(self, pool: WorkerPool):
        job_type = pool.job_type()
        queue = self.type_queues[job_type]
        while True:
            (future, job) = await queue.get()
            try:
                pool_future = await pool.submit(job)
                task = self._on_complete(future, pool_future)
                asyncio.create_task(task)
            except RuntimeError as e:
                future.set_exception(e)

    async def _on_complete(self, future: Future, pool_future: Awaitable):
        result = await pool_future
        future.set_result(result)

    async def _request_kerberos_ticket(self):
        minutes = 60
        sleep_period = 15 * minutes
        while True:
            command = f"{self.root_dir}/../cronjobs/kerberos.sh"
            _, stdout, _ = self.ssh_client.exec_command(command)
            _ = stdout.read().decode("utf-8")
            await asyncio.sleep(sleep_period)
