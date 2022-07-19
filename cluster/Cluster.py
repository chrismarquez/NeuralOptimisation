from asyncio import Queue, Future, Task
from time import sleep

import asyncio
from typing import Awaitable, List, Mapping, Tuple

from cluster.CondorPool import CondorPool, CondorConfig
from cluster.Job import Job, JobType
from cluster.SlurmPool import SlurmPool
from cluster.WorkerPool import WorkerPool


class Cluster:

    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.pools: List[WorkerPool] = [
            SlurmPool(root_dir, capacity=2),
            CondorPool(root_dir, capacity=8, condor_server="shell1.doc.ic.ac.uk", config=CondorConfig("csm21", "CPU")),
            CondorPool(root_dir, capacity=10, condor_server="shell1.doc.ic.ac.uk", config=CondorConfig("csm21", "GPU"))
        ]

        self.consumers: List[Task] = []

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

    def stop(self):
        for consumer in self.consumers:
            consumer.cancel()

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
