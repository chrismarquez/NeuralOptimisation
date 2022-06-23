import asyncio
from asyncio import Queue, Task
from typing import TypeVar, List, Callable, Optional, Awaitable

from cluster.Job import Job

T = TypeVar('T')

Submit = Callable[[Job], Awaitable[Task]]
Pipe = Callable[[Task], Awaitable[Optional[Job]]]


async def _end_pipe(task: Awaitable[None]) -> Optional[Job]:
    await task
    return None


class Segment:

    def __init__(
        self,
        name: str,
        capacity: int,
        submit: Submit,
        pipe: Pipe = _end_pipe
    ):
        self.name = name
        self.job_completed: int = 0
        self.job_queue: Queue[T] = Queue(maxsize=capacity)
        self.submit = submit
        self.pipe = pipe
        self.next: Optional[Segment] = None
        self._consumer: Optional[Task] = None

    async def enqueue(self, job: T):
        await self.job_queue.put(job)

    def start(self):
        self._consumer = asyncio.create_task(self._consume())

    async def finish(self):
        await self.job_queue.join()
        self._consumer.cancel()

    async def _pipe_and_mark(self, task: Task) -> Optional[Job]:
        try:
            job = await self.pipe(task)
            self.job_completed += 1
        except RuntimeError:
            print(f"Could not pipe task {task}")
            job = None
        self.job_queue.task_done()
        return job

    async def _consume(self):
        while True:
            job = await self.job_queue.get()
            task = await self.submit(job)
            next_job = await self._pipe_and_mark(task)
            if self.next is not None and next_job is not None:
                await self.next.enqueue(next_job)


class Pipeline:

    def __init__(self, segments: Optional[List[Segment]] = None):
        if segments is None:
            segments = []
        self.job_batch_size: int = 0
        self.segments: List[Segment] = []
        for segment in segments:
            self.add(segment)

    def add(self, segment: Segment):
        if len(self.segments) != 0:
            last = self.segments[-1]
            last.next = segment
        self.segments.append(segment)

    async def execute(self, jobs: List[Job]):
        self.job_batch_size = len(jobs)
        producer = asyncio.create_task(self._produce(jobs))
        reporter = asyncio.create_task(self._report_progress())
        for segment in self.segments:
            segment.start()
        await producer
        for segment in self.segments:
            await segment.finish()
        reporter.cancel()

    async def _produce(self, jobs: List[Job]):
        first_segment = self.segments[0]
        for job in jobs:
            await first_segment.enqueue(job)

    async def _report_progress(self):
        # tqdm test
        total = self.job_batch_size
        while True:
            await asyncio.sleep(1)
            for segment in self.segments:
                completed = segment.job_completed
                print(f"Segment {segment.name} progress: Completed {completed} / {total} jobs \t", end="")
            print()
