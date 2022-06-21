from abc import ABC
from asyncio import Queue


class WorkerPool(ABC):

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._slots_queue: Queue[int] = Queue(maxsize=capacity)

    async def _request_slot(self):
        await self._slots_queue.put(0)

    async def _release_slot(self):
        await self._slots_queue.get()
        self._slots_queue.task_done()
