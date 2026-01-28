"""Performance monitoring."""

import time
import cProfile
import pstats
import io
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime

from logger_config import get_logger

logger = get_logger(__name__)


class PerformanceMonitor:
    """Performance monitoring and timing system."""

    def __init__(self, enable_profiling: bool = False):
        """Initialize performance monitor.

        Args:
            enable_profiling: Enable detailed profiling
        """
        self.enable_profiling = enable_profiling
        self.timings: Dict[str, List[float]] = {}
        self.counters: Dict[str, int] = {}
        self.profiler: Optional[cProfile.Profile] = None

        if enable_profiling:
            self.profiler = cProfile.Profile()

        logger.info(f"PerformanceMonitor initialized (profiling: {enable_profiling})")

    def start_timing(self, name: str) -> float:
        """Start timing an operation.

        Args:
            name: Operation name

        Returns:
            Start timestamp
        """
        if name not in self.timings:
            self.timings[name] = []

        start_time = time.time()
        return start_time

    def end_timing(self, name: str, start_time: float) -> float:
        """End timing an operation and record duration.

        Args:
            name: Operation name
            start_time: Start timestamp from start_timing()

        Returns:
            Duration in seconds
        """
        duration = time.time() - start_time
        self.timings[name].append(duration)

        return duration

    def record_timing(self, name: str, duration: float) -> None:
        """Record a timing directly.

        Args:
            name: Operation name
            duration: Duration in seconds
        """
        if name not in self.timings:
            self.timings[name] = []

        self.timings[name].append(duration)

    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a counter.

        Args:
            name: Counter name
            value: Increment value
        """
        if name not in self.counters:
            self.counters[name] = 0

        self.counters[name] += value

    def get_timing_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a timed operation.

        Args:
            name: Operation name

        Returns:
            Dictionary with min, max, mean, total, count
        """
        if name not in self.timings or not self.timings[name]:
            return {}

        timings = self.timings[name]

        return {
            "count": len(timings),
            "total": sum(timings),
            "mean": sum(timings) / len(timings),
            "min": min(timings),
            "max": max(timings),
        }

    def get_all_stats(self) -> Dict[str, Any]:
        """Get all timing and counter statistics.

        Returns:
            Dictionary with all statistics
        """
        stats = {
            "timings": {},
            "counters": self.counters.copy(),
        }

        for name in self.timings:
            stats["timings"][name] = self.get_timing_stats(name)

        return stats

    def print_summary(self) -> None:
        """Print performance summary to logger."""
        logger.info("=== Performance Summary ===")

        # Timings
        if self.timings:
            logger.info("Timings:")
            for name, stats in sorted(
                ((n, self.get_timing_stats(n)) for n in self.timings),
                key=lambda x: x[1].get("total", 0),
                reverse=True,
            ):
                if stats:
                    logger.info(
                        f"  {name}: {stats['count']} calls, "
                        f"{stats['total']:.2f}s total, "
                        f"{stats['mean']:.3f}s avg, "
                        f"{stats['min']:.3f}s min, "
                        f"{stats['max']:.3f}s max"
                    )

        # Counters
        if self.counters:
            logger.info("Counters:")
            for name, count in sorted(
                self.counters.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                logger.info(f"  {name}: {count}")

    def start_profiling(self) -> None:
        """Start profiling."""
        if not self.enable_profiling:
            logger.warning("Profiling not enabled")
            return

        if self.profiler is None:
            self.profiler = cProfile.Profile()

        self.profiler.enable()
        logger.debug("Profiling started")

    def stop_profiling(self) -> None:
        """Stop profiling."""
        if self.profiler is not None:
            self.profiler.disable()
            logger.debug("Profiling stopped")

    def get_profile_stats(self, sort_by: str = "cumulative", limit: int = 20) -> str:
        """Get profiling statistics.

        Args:
            sort_by: Sort criterion (cumulative, time, calls)
            limit: Number of lines to show

        Returns:
            Formatted profiling statistics
        """
        if self.profiler is None:
            return "Profiling not available"

        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.strip_dirs()
        ps.sort_stats(sort_by)
        ps.print_stats(limit)

        return s.getvalue()

    def save_profile(self, filepath: str) -> None:
        """Save profiling data to file.

        Args:
            filepath: Path to save profile data
        """
        if self.profiler is None:
            logger.warning("No profiling data to save")
            return

        self.profiler.dump_stats(filepath)
        logger.info(f"Profile saved to {filepath}")

    @contextmanager
    def measure(self, name: str, log: bool = True):
        """Context manager for measuring execution time.

        Args:
            name: Operation name
            log: Log the timing

        Example:
            with monitor.measure("data_loading"):
                data = load_data()
        """
        start = self.start_timing(name)

        try:
            yield
        finally:
            duration = self.end_timing(name, start)
            if log:
                logger.debug(f"{name}: {duration:.3f}s")

    def timed(self, name: Optional[str] = None):
        """Timing decorator.

        Args:
            name: Optional custom name

        Example:
            @monitor.timed()
            def process():
                pass
        """

        def decorator(func: Callable) -> Callable:
            operation_name = name or func.__name__

            def wrapper(*args, **kwargs):
                start = self.start_timing(operation_name)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = self.end_timing(operation_name, start)
                    logger.debug(f"{operation_name}: {duration:.3f}s")

            return wrapper

        return decorator

    def reset(self) -> None:
        """Reset all statistics."""
        self.timings.clear()
        self.counters.clear()

        if self.profiler is not None:
            self.profiler = cProfile.Profile()

        logger.debug("Performance monitor reset")

    def export_report(self, filepath: str) -> None:
        """Export performance report to file.

        Args:
            filepath: Path to save report
        """
        report_path = Path(filepath)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Performance Report\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")

            # Timings
            f.write("TIMINGS\n")
            f.write("-" * 80 + "\n")

            for name in sorted(self.timings.keys()):
                stats = self.get_timing_stats(name)
                if stats:
                    f.write(f"\n{name}:\n")
                    f.write(f"  Calls:  {stats['count']}\n")
                    f.write(f"  Total:  {stats['total']:.3f}s\n")
                    f.write(f"  Mean:   {stats['mean']:.3f}s\n")
                    f.write(f"  Min:    {stats['min']:.3f}s\n")
                    f.write(f"  Max:    {stats['max']:.3f}s\n")

            # Counters
            f.write("\n" + "=" * 80 + "\n")
            f.write("COUNTERS\n")
            f.write("-" * 80 + "\n\n")

            for name, count in sorted(self.counters.items()):
                f.write(f"{name}: {count}\n")

            # Profiling
            if self.profiler is not None:
                f.write("\n" + "=" * 80 + "\n")
                f.write("PROFILING\n")
                f.write("-" * 80 + "\n\n")
                f.write(self.get_profile_stats())

        logger.info(f"Performance report saved to {filepath}")


# Global monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_monitor() -> PerformanceMonitor:
    """Get global performance monitor.

    Returns:
        Global PerformanceMonitor instance
    """
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def measure(name: str):
    """Convenience function for timing context manager.

    Args:
        name: Operation name

    Returns:
        Context manager
    """
    return get_monitor().measure(name)


def timed(name: Optional[str] = None):
    """Convenience decorator for timing functions.

    Args:
        name: Optional operation name

    Returns:
        Decorator
    """
    return get_monitor().timed(name)
