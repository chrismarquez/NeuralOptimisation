from abc import ABC
from asyncio import Queue
from typing import Awaitable

from cluster.Job import Job, JobType


class WorkerPool(ABC):

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._slots_queue: Queue[int] = Queue(maxsize=capacity)

    async def _request_slot(self):
        await self._slots_queue.put(0)

    async def _release_slot(self):
        await self._slots_queue.get()
        self._slots_queue.task_done()

    async def submit(self, job: Job) -> Awaitable[str]:
        pass

    def job_type(self) -> JobType:
        pass
