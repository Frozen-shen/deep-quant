"""
Parallel computation helpers for quant-starter.

Provides a simple, robust interface for parallelizing independent tasks
(e.g., per-stock factor computation, per-date backtesting).

Features:
    - Simple parallel_map(func, items) interface
    - Progress bar via tqdm
    - Per-task error handling (one failure doesn't kill the pool)
    - Configurable worker count with sensible defaults
    - Timeout support per task

Usage:
    from quant.utils.parallel import parallel_map

    results = parallel_map(
        compute_factor,
        stock_list,
        n_workers=4,
        desc="Computing factors",
    )
"""

from __future__ import annotations

import multiprocessing
import traceback
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional, Sequence, TypeVar

from loguru import logger

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class TaskResult:
    """Result of a single parallel task."""

    index: int
    item: Any
    result: Any = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    success: bool = True


@dataclass
class ParallelResult:
    """Aggregate result from a parallel_map call."""

    results: List[TaskResult] = field(default_factory=list)
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    elapsed_seconds: float = 0.0

    @property
    def values(self) -> List[Any]:
        """Return results in original order (None for failed tasks)."""
        ordered = sorted(self.results, key=lambda r: r.index)
        return [r.result for r in ordered]

    @property
    def successful_values(self) -> List[Any]:
        """Return only successful results, in original order."""
        ordered = sorted(self.results, key=lambda r: r.index)
        return [r.result for r in ordered if r.success]

    @property
    def failures(self) -> List[TaskResult]:
        """Return only failed task results."""
        return [r for r in self.results if not r.success]

    def summary(self) -> str:
        """Human-readable summary of the parallel execution."""
        lines = [
            f"Parallel execution: {self.succeeded}/{self.total} succeeded "
            f"({self.failed} failed) in {self.elapsed_seconds:.2f}s"
        ]
        if self.failures:
            lines.append("Failures:")
            for f in self.failures[:5]:  # Show first 5 failures
                lines.append(f"  [{f.index}] {f.error}")
            if len(self.failures) > 5:
                lines.append(f"  ... and {len(self.failures) - 5} more")
        return "\n".join(lines)


def _execute_task(args: tuple) -> TaskResult:
    """
    Worker function that executes a single task with error handling.

    This runs in a separate process/thread and must be a top-level
    function for pickling compatibility.
    """
    index, item, func = args
    try:
        result = func(item)
        return TaskResult(index=index, item=item, result=result, success=True)
    except Exception as e:
        return TaskResult(
            index=index,
            item=item,
            error=str(e),
            traceback=traceback.format_exc(),
            success=False,
        )


def parallel_map(
    func: Callable[[T], R],
    items: Sequence[T],
    n_workers: Optional[int] = None,
    desc: str = "Processing",
    use_threads: bool = False,
    timeout: Optional[float] = None,
    chunksize: int = 1,
    raise_on_error: bool = False,
    progress: bool = True,
) -> ParallelResult:
    """
    Apply a function to items in parallel with progress tracking.

    Each task is independent; one failure does not affect others.
    Results are returned in the original order of items.

    Args:
        func: Function to apply to each item. Must be picklable if
              use_threads=False (i.e., a top-level function, not a lambda).
        items: Sequence of items to process.
        n_workers: Number of parallel workers. Defaults to min(cpu_count, len(items)).
        desc: Description for the progress bar.
        use_threads: If True, use threads instead of processes.
                     Use for I/O-bound tasks (network, disk).
        timeout: Per-task timeout in seconds. None means no timeout.
        chunksize: Number of items per worker batch (processes only).
                   Larger values reduce IPC overhead for many small tasks.
        raise_on_error: If True, raise the first exception encountered
                        instead of collecting it in the result.
        progress: Whether to show a tqdm progress bar.

    Returns:
        ParallelResult containing all task outcomes.

    Raises:
        RuntimeError: If raise_on_error=True and any task fails.

    Example:
        def compute_factor_for_stock(symbol: str) -> pd.DataFrame:
            ...

        result = parallel_map(
            compute_factor_for_stock,
            ["000001", "000002", "600000"],
            n_workers=4,
            desc="Computing factors",
        )
        print(result.summary())
        dataframes = result.successful_values
    """
    import time

    if not items:
        return ParallelResult(results=[], total=0, succeeded=0, failed=0)

    total = len(items)
    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count(), total)
    n_workers = max(1, n_workers)

    # Prepare task arguments
    task_args = [(i, item, func) for i, item in enumerate(items)]

    start_time = time.time()
    results: List[TaskResult] = []

    # Choose executor
    ExecutorClass = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

    # For threads, use direct submission; for processes, use map with chunksize
    if use_threads:
        results = _run_with_thread_pool(
            ExecutorClass, task_args, n_workers, timeout, desc, progress
        )
    else:
        results = _run_with_process_pool(
            ExecutorClass, task_args, n_workers, timeout, chunksize, desc, progress
        )

    elapsed = time.time() - start_time

    succeeded = sum(1 for r in results if r.success)
    failed = total - succeeded

    parallel_result = ParallelResult(
        results=results,
        total=total,
        succeeded=succeeded,
        failed=failed,
        elapsed_seconds=elapsed,
    )

    # Log summary
    if failed > 0:
        logger.warning(
            "Parallel '{}' completed: {}/{} succeeded, {} failed ({:.2f}s)",
            desc,
            succeeded,
            total,
            failed,
            elapsed,
        )
    else:
        logger.debug(
            "Parallel '{}' completed: {}/{} succeeded ({:.2f}s)",
            desc,
            succeeded,
            total,
            elapsed,
        )

    if raise_on_error and failed > 0:
        first_failure = next(r for r in results if not r.success)
        raise RuntimeError(
            f"Task {first_failure.index} failed: {first_failure.error}\n"
            f"{first_failure.traceback}"
        )

    return parallel_result


def _run_with_thread_pool(
    executor_class,
    task_args: List[tuple],
    n_workers: int,
    timeout: Optional[float],
    desc: str,
    progress: bool,
) -> List[TaskResult]:
    """Execute tasks using a thread pool with progress tracking."""
    results: List[TaskResult] = [None] * len(task_args)  # type: ignore[list-item]

    with executor_class(max_workers=n_workers) as executor:
        future_to_index = {}
        for args in task_args:
            future = executor.submit(_execute_task, args)
            future_to_index[future] = args[0]

        if progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(
                    as_completed(future_to_index),
                    total=len(task_args),
                    desc=desc,
                    unit="task",
                )
            except ImportError:
                iterator = as_completed(future_to_index)
        else:
            iterator = as_completed(future_to_index)

        for future in iterator:
            idx = future_to_index[future]
            try:
                task_result = future.result(timeout=timeout)
                results[idx] = task_result
            except Exception as e:
                results[idx] = TaskResult(
                    index=idx,
                    item=task_args[idx][1],
                    error=str(e),
                    traceback=traceback.format_exc(),
                    success=False,
                )

    return results


def _run_with_process_pool(
    executor_class,
    task_args: List[tuple],
    n_workers: int,
    timeout: Optional[float],
    chunksize: int,
    desc: str,
    progress: bool,
) -> List[TaskResult]:
    """Execute tasks using a process pool with progress tracking."""
    results: List[TaskResult] = [None] * len(task_args)  # type: ignore[list-item]

    with executor_class(max_workers=n_workers) as executor:
        if progress:
            try:
                from tqdm import tqdm

                # Use submit pattern for progress tracking
                future_to_index = {}
                for args in task_args:
                    future = executor.submit(_execute_task, args)
                    future_to_index[future] = args[0]

                for future in tqdm(
                    as_completed(future_to_index),
                    total=len(task_args),
                    desc=desc,
                    unit="task",
                ):
                    idx = future_to_index[future]
                    try:
                        task_result = future.result(timeout=timeout)
                        results[idx] = task_result
                    except Exception as e:
                        results[idx] = TaskResult(
                            index=idx,
                            item=task_args[idx][1],
                            error=str(e),
                            traceback=traceback.format_exc(),
                            success=False,
                        )
            except ImportError:
                # No tqdm, fall back to map
                for task_result in executor.map(
                    _execute_task, task_args, chunksize=chunksize, timeout=timeout
                ):
                    results[task_result.index] = task_result
        else:
            for task_result in executor.map(
                _execute_task, task_args, chunksize=chunksize, timeout=timeout
            ):
                results[task_result.index] = task_result

    return results


def parallel_map_simple(
    func: Callable[[T], R],
    items: Sequence[T],
    n_workers: int = 4,
) -> List[R]:
    """
    Simplified parallel map that returns just the values.

    Raises on first error. Use parallel_map() for more control.

    Args:
        func: Function to apply to each item.
        items: Items to process.
        n_workers: Number of workers.

    Returns:
        List of results in original order.

    Raises:
        RuntimeError: If any task fails.
    """
    result = parallel_map(
        func, items, n_workers=n_workers, raise_on_error=True, progress=False
    )
    return result.values  # type: ignore[return-value]
