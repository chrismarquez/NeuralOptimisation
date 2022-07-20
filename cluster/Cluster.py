import subprocess
from asyncio import Queue, Future, Task

import asyncio
from typing import Awaitable, List, Mapping, Tuple, Optional

from cluster.CondorPool import CondorPool, CondorConfig
from cluster.Job import Job, JobType
from cluster.SlurmPool import SlurmPool
from cluster.WorkerPool import WorkerPool


class Cluster:

    def __init__(self, root_dir, condor_server: str, raw_debug: str):
        self.root_dir = root_dir
        debug = raw_debug == "True"
        self.pools: List[WorkerPool] = [
            SlurmPool(root_dir, capacity=2, debug=debug),
            CondorPool(root_dir, capacity=8, condor_server=condor_server, config=CondorConfig("csm21", "CPU", debug)),
            CondorPool(root_dir, capacity=25, condor_server=condor_server, config=CondorConfig("csm21", "GPU", debug))
        ]

        self.consumers: List[Task] = []
        self.kerberos_request: Optional[Task] = None

        self.type_queues: Mapping[JobType, Queue[Tuple[Future[str], Job]]] = {
            "CPU": Queue(),
            "GPU": Queue()
        }

    # Coroutine launch method
    async def submit(self, job: Job) -> Awaitable[str]:
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
            result = subprocess.run(f"{self.root_dir}/../cronjobs/kerberos.sh", shell=True, capture_output=True)
            print(result.stdout.decode("utf-8"))
            await asyncio.sleep(sleep_period)
