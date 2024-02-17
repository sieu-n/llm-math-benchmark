import asyncio
from typing import Union

import aiojobs
from tqdm import tqdm

INF = 999999999999999


def run_batch_aiojobs_sync(*args):
    asyncio.run(run_batch_aiojobs(*args))


async def run_batch_aiojobs(
    f: callable,
    args: list[Union[object, tuple]] = None,
    kwargs: list[dict] = None,
    post_process: callable = None,
    limit: int = 100,
    pbar: bool = True,
) -> list[object]:
    """
    Wrapper for aiojobs to run a batch of jobs in parallel.

    Args:
        f (callable): Function to be called in parallel.
        args (list[object, tuple], optional): List of positional arguments for the function.
        kwargs (list[dict], optional): List of keyword arguments for the function.
        post_process (callable, optional): Function applied on the result of each job, before being stored in the results list.
        limit (int, optional): Maximum number of concurrent jobs.
        pbar (bool, optional): Whether to show a progress bar.
    Returns:
        list[object]: List of results from the function calls.
    """
    use_pbar: bool = pbar
    n = len(args or kwargs)

    if n == 0:
        return []
    if args is None:
        args = [() for _ in range(n)]
    elif type(args[0]) != tuple:
        args = [(a,) for a in args]
    if kwargs is None:
        kwargs = [{} for _ in range(n)]

    if use_pbar:
        pbar = tqdm(total=n)

    results = [None] * n

    async def coro(i, *args, **kwargs):
        assert results[i] is None
        res = await f(*args, **kwargs)
        if post_process is not None:
            res = post_process(res)
        if use_pbar:
            pbar.update(1)
        results[i] = res

    scheduler = aiojobs.Scheduler(limit=limit, pending_limit=INF, close_timeout=INF)

    for i in range(n):
        await scheduler.spawn(coro(i, *args[i], **kwargs[i]))

    # wait for all jobs to finish
    while len(scheduler) > 0:  # number of active or scheduled jobs
        await asyncio.sleep(0.1)
    await scheduler.close()

    if use_pbar:
        pbar.close()
    return results
