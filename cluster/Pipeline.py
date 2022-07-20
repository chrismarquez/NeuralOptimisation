import asyncio
from asyncio import Queue, Task
from typing import TypeVar, List, Callable, Optional, Awaitable

from tqdm import tqdm

from cluster.Job import Job, JobType
from cluster.JobContainer import JobContainer

T = TypeVar('T')

Submit = Callable[[Job], Awaitable[Task]]
Pipe = Callable[[Task], Awaitable[List[Job]]]


async def _end_pipe(task: Awaitable[None]) -> List[Job]:
    await task
    return []


class Segment:

    def __init__(
        self,
        name: str,
        capacity: int,
        expected_jobs: int,
        submit: Submit,
        pipe: Pipe = _end_pipe
    ):
        self.name = name
        self.expected_jobs = expected_jobs
        self.job_completed: int = 0
        self.job_queue: Queue[T] = Queue(maxsize=capacity)
        self.submit = submit
        self.pipe = pipe
        self.next_segment: Optional[Segment] = None
        self._consumer: Optional[Task] = None

    async def enqueue(self, job: T):
        await self.job_queue.put(job)

    def start(self):
        self._consumer = asyncio.create_task(self._consume())

    async def finish(self):
        await self.job_queue.join()
        self._consumer.cancel()

    async def _pipe_and_mark(self, task: Task) -> List[Job]:
        try:
            jobs = await self.pipe(task)
            self.job_completed += 1
        except RuntimeError:
            print(f"Could not pipe task {task}")
            jobs = []
        self.job_queue.task_done()
        return jobs

    async def _consume(self):
        while True:
            job = await self.job_queue.get()
            task = await self.submit(job)
            await asyncio.sleep(0.3)
            asyncio.create_task(self._post_process(task))

    async def _post_process(self, task: Task):
        next_jobs = await self._pipe_and_mark(task)
        if self.next_segment is not None:
            for job in next_jobs:
                await self.next_segment.enqueue(job)


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
            last.next_segment = segment
        self.segments.append(segment)

    async def execute(self, jobs: List[Job]):
        self.job_batch_size = len(jobs)
        producer = asyncio.create_task(self._produce(jobs))
        reporters = self._spawn_reporters()
        for segment in self.segments:
            segment.start()
        await producer
        for segment in self.segments:
            await segment.finish()
        for reporter in reporters:
            await reporter

    async def _produce(self, jobs: List[Job]):
        first_segment = self.segments[0]
        for job in jobs:
            await first_segment.enqueue(job)

    def _spawn_reporters(self) -> List[Task]:
        return [
            asyncio.create_task(self._report_segment(segment, i))
            for i, segment in enumerate(self.segments)
        ]

    @staticmethod
    async def _report_segment(segment: Segment, position: int):
        with tqdm(total=segment.expected_jobs, position=position, leave=True) as pbar:
            pbar.set_description(segment.name)
            previous_completed = 0
            while True:
                update = segment.job_completed - previous_completed
                pbar.update(update)
                previous_completed += update
                if segment.expected_jobs == previous_completed:
                    break
                await asyncio.sleep(1)


class DummyJob(Job):
    def run(self, container: JobContainer):
        pass

    def as_command(self) -> str:
        return ""

    def get_job_type(self) -> JobType:
        return "CPU"


async def _mock_submit(job: Job) -> Awaitable[Task]:
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    container = JobContainer()
    container.init_resources()
    container.wire(modules=[__name__])
    future.set_result(job.run(container))
    return future

async def _mock_pipe(task: Awaitable[None]) -> List[Job]:
    await task
    return [DummyJob("")]


async def main():
    pipeline = Pipeline([
        Segment(name="Train", capacity=50, expected_jobs=50, submit=_mock_submit, pipe=_mock_pipe),
        Segment(name="Opt", capacity=50, expected_jobs=50, submit=_mock_submit)
    ])
    jobs = [DummyJob("") for _ in range(50)]
    await pipeline.execute(jobs)


if __name__ == '__main__':
    asyncio.run(main())
