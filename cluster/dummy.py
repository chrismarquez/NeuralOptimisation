
import asyncio

async def main():
    items = [i for i in range(10)]
    queue = asyncio.Queue(maxsize=3)

    prod = asyncio.create_task(produce(items, queue))
    cons = asyncio.create_task(consume(queue))

    await asyncio.gather(prod)
    print("Finish Production")
    await queue.join()
    print("Finish Consumer")
    cons.cancel()


async def produce(items, q):
    for item in items:
        await q.put(item)
        print(f"Insert {item}")


async def consume(q):
    while True:
        item = await q.get()
        await asyncio.sleep(1)
        q.task_done()
        print(f"Processed {item}")

if __name__ == '__main__':
    asyncio.run(main())


