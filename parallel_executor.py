"""Parallel execution engine for plot generation."""

import multiprocessing as mp
import os
import time
from typing import Callable, List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import traceback

from logger_config import get_logger

logger = get_logger(__name__)


class TaskStatus(Enum):
    """Status of a plot generation task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlotTask:
    """Represents a plot generation task."""

    name: str  # Function/plot name
    func: Callable  # Function to execute
    args: Tuple = field(default_factory=tuple)  # Function arguments
    kwargs: Dict[str, Any] = field(default_factory=dict)  # Function keyword arguments
    check_existing: bool = True  # Skip if output file exists
    output_file: Optional[Path] = None  # Expected output file path

    status: TaskStatus = field(default=TaskStatus.PENDING, init=False)
    result: Optional[Dict] = field(default=None, init=False)
    error: Optional[str] = field(default=None, init=False)
    start_time: Optional[float] = field(default=None, init=False)
    end_time: Optional[float] = field(default=None, init=False)

    @property
    def duration(self) -> Optional[float]:
        """Get task duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    def __repr__(self) -> str:
        """String representation."""
        return f"PlotTask({self.name}: {self.status.value})"


@dataclass
class ExecutionResult:
    """Result of executing a plot task."""

    task_name: str
    status: TaskStatus
    result: Optional[Dict] = None
    error: Optional[str] = None
    duration: Optional[float] = None


def _execute_task(
    task: PlotTask,
    check_existing: bool = True,
) -> ExecutionResult:
    """Execute a single plot task in subprocess.

    Args:
        task: PlotTask to execute
        check_existing: Whether to check for existing output files

    Returns:
        ExecutionResult with status and output
    """
    try:
        # Check if output file already exists
        if check_existing and task.output_file and Path(task.output_file).exists():

            logger.info(f"Skipping {task.name} (file exists: {task.output_file})")
            return ExecutionResult(
                task_name=task.name,
                status=TaskStatus.SKIPPED,
            )

        logger.info(f"Starting task: {task.name}")
        start_time = time.time()

        # Execute function
        result = task.func(*task.args, **task.kwargs)

        duration = time.time() - start_time
        logger.info(f"Completed task: {task.name} ({duration:.2f}s)")

        return ExecutionResult(
            task_name=task.name,
            status=TaskStatus.COMPLETED,
            result=result,
            duration=duration,
        )

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        logger.error(f"Task failed: {task.name}\n{error_msg}")

        return ExecutionResult(
            task_name=task.name,
            status=TaskStatus.FAILED,
            error=error_msg,
        )


class ParallelPlotExecutor:
    """Manages parallel execution of plot generation tasks."""

    def __init__(
        self,
        max_workers: Optional[int] = None,
        timeout_seconds: float = 300,
        check_existing: bool = True,
    ):
        """Initialize ParallelPlotExecutor.

        Args:
            max_workers: Maximum number of worker processes.
                        If None, uses number of CPU cores.
            timeout_seconds: Timeout per task
            check_existing: Skip execution if output file exists
        """
        if max_workers is None:
            max_workers = os.cpu_count() or 4
        elif max_workers <= 0:
            max_workers = os.cpu_count() or 4

        self.max_workers = min(max_workers, os.cpu_count() or 4)
        self.timeout_seconds = timeout_seconds
        self.check_existing = check_existing

        self.tasks: List[PlotTask] = []
        self.results: List[ExecutionResult] = []

        logger.info(
            f"ParallelPlotExecutor initialized: "
            f"max_workers={self.max_workers}, "
            f"timeout={timeout_seconds}s"
        )

    def add_task(
        self,
        name: str,
        func: Callable,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        output_file: Optional[Path] = None,
    ) -> None:
        """Add a plot generation task.

        Args:
            name: Task identifier
            func: Function to execute
            args: Function positional arguments
            kwargs: Function keyword arguments
            output_file: Expected output file path
        """
        task = PlotTask(
            name=name,
            func=func,
            args=args,
            kwargs=kwargs or {},
            output_file=output_file,
        )
        self.tasks.append(task)
        logger.debug(f"Task added: {name}")

    def add_tasks(self, tasks: List[PlotTask]) -> None:
        """Add multiple tasks.

        Args:
            tasks: List of PlotTask objects
        """
        self.tasks.extend(tasks)
        logger.info(f"Added {len(tasks)} tasks")

    def execute(self) -> List[ExecutionResult]:
        """Execute all tasks in parallel.

        Returns:
            List of ExecutionResult objects
        """
        if not self.tasks:
            logger.warning("No tasks to execute")
            return []

        logger.info(
            f"Starting parallel execution: {len(self.tasks)} tasks, "
            f"{self.max_workers} workers"
        )

        start_time = time.time()

        # Use ProcessPoolExecutor for CPU-bound work
        with mp.Pool(processes=self.max_workers) as pool:
            # Execute tasks
            async_results = []
            for task in self.tasks:
                async_result = pool.apply_async(
                    _execute_task,
                    args=(task, self.check_existing),
                )
                async_results.append(async_result)

            # Collect results
            self.results = []
            for i, async_result in enumerate(async_results):
                try:
                    result = async_result.get(timeout=self.timeout_seconds)
                    self.results.append(result)

                except mp.TimeoutError:
                    logger.error(f"Task {i} timed out")
                    self.results.append(
                        ExecutionResult(
                            task_name=self.tasks[i].name,
                            status=TaskStatus.FAILED,
                            error="Task timeout",
                        )
                    )
                except Exception as e:
                    logger.error(f"Error collecting result {i}: {e}")
                    self.results.append(
                        ExecutionResult(
                            task_name=self.tasks[i].name,
                            status=TaskStatus.FAILED,
                            error=str(e),
                        )
                    )

        total_time = time.time() - start_time
        self._log_execution_summary(total_time)

        return self.results

    def _log_execution_summary(self, total_time: float) -> None:
        """Log execution summary statistics."""
        completed = sum(1 for r in self.results if r.status == TaskStatus.COMPLETED)
        failed = sum(1 for r in self.results if r.status == TaskStatus.FAILED)
        skipped = sum(1 for r in self.results if r.status == TaskStatus.SKIPPED)

        logger.info(
            f"Execution complete: {completed} completed, {skipped} skipped, "
            f"{failed} failed in {total_time:.2f}s"
        )

        for result in self.results:
            if result.status == TaskStatus.COMPLETED:
                logger.info(f"  ✓ {result.task_name} ({result.duration:.2f}s)")
            elif result.status == TaskStatus.FAILED:
                logger.warning(f"  ✗ {result.task_name}: {result.error}")
            elif result.status == TaskStatus.SKIPPED:
                logger.info(f"  ⊘ {result.task_name} (skipped)")

    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary.

        Returns:
            Dictionary with execution statistics
        """
        total_duration = sum(r.duration for r in self.results if r.duration) or 0

        return {
            "total_tasks": len(self.tasks),
            "completed": sum(
                1 for r in self.results if r.status == TaskStatus.COMPLETED
            ),
            "failed": sum(1 for r in self.results if r.status == TaskStatus.FAILED),
            "skipped": sum(1 for r in self.results if r.status == TaskStatus.SKIPPED),
            "total_duration": total_duration,
            "workers_used": self.max_workers,
        }

    def clear(self) -> None:
        """Clear tasks and results."""
        self.tasks.clear()
        self.results.clear()
        logger.debug("Executor cleared")
