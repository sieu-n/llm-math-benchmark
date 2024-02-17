import asyncio

import pytest

from src.aiojobs import run_batch_aiojobs  # Adjust the import according to your file structure


async def _dummy_job(a):
    if a == 63:
        raise ValueError("dummy error")
    await asyncio.sleep(0.1)
    return a


@pytest.mark.asyncio
async def test_run_batch_aiojobs():
    start_time = asyncio.get_event_loop().time()

    args = list(range(100))
    args.remove(63)

    results = await run_batch_aiojobs(_dummy_job, args=args, limit=10, pbar=False)

    end_time = asyncio.get_event_loop().time()
    total_time = end_time - start_time

    expected_results = args
    assert all([a == b for a, b in zip(results, expected_results)]), "The job results are not as expected."

    # Ensure the jobs are running in parallel by checking the total execution time.
    # If 99 jobs take 0.1 seconds each sequentially, it would take 9.9 seconds.
    # Running in parallel with a limit of 10, it should take significantly less time.
    assert 0.7 < total_time < 1.3, "Jobs are not running in parallel as expected."


@pytest.mark.asyncio
async def test_run_batch_aiojobs_format():
    results1 = await run_batch_aiojobs(_dummy_job, args=list(range(100)), limit=100, pbar=False)
    results2 = await run_batch_aiojobs(_dummy_job, args=[(i,) for i in range(100)], limit=100, pbar=False)
    results3 = await run_batch_aiojobs(_dummy_job, kwargs=[{"a": i} for i in range(100)], limit=100, pbar=False)

    assert all([a == b and b == c for a, b, c in zip(results1, results2, results3)]), "The job results are not as expected."


@pytest.mark.asyncio
async def test_run_batch_aiojobs_error():
    args = list(range(100))
    results = await run_batch_aiojobs(_dummy_job, args=args, limit=10, pbar=False)

    assert len(results) == 100, "Should return 100 results"
    assert all(arg == 63 or result == arg for result, arg in zip(results, args)), "Results should match the input arguments"
