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
        self.debug = raw_debug == "True"
        self.pools: List[WorkerPool] = [
            SlurmPool(root_dir, capacity=2, debug=self.debug),
            CondorPool(root_dir, capacity=40, condor_server=condor_server, config=CondorConfig(user, "CPU", self.debug))
            #CondorPool(root_dir, capacity=25, condor_server=condor_server, config=CondorConfig(user, "GPU", self.debug))
        ]

        print("[Cluster] Connecting to Shell Server.")
        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.load_system_host_keys()
        self.ssh_client.connect(condor_server, username=user)
        transport = self.ssh_client.get_transport()
        transport.set_keepalive(interval=300)
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
                task = self._on_complete(future, job, pool_future)
                asyncio.create_task(task)
            except RuntimeError as e:
                await self._reschedule_failed(future, job, e)

    async def _on_complete(self, future: Future, job: Job, pool_future: Awaitable):
        try:
            result = await pool_future
            future.set_result(result)
        except RuntimeError:
            future.set_result(False)

    async def _reschedule_failed(self, future: Future, job: Job, e: Exception):
        future.set_exception(e)
        if self.debug:
            print(f"Failed Job {job.uuid} with model {job.model_id}. Adding to queue again")
        job_type = job.get_job_type()
        await self.type_queues[job_type].put((future, job))

    async def _request_kerberos_ticket(self):
        minutes = 60
        hours = 60 * minutes
        sleep_period = 4 * hours
        while True:
            command = f"{self.root_dir}/../cronjobs/kerberos.sh"
            _, stdout, _ = self.ssh_client.exec_command(command)
            _ = stdout.read().decode("utf-8")
            await asyncio.sleep(sleep_period)
