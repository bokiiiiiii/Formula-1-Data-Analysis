"""Memory management utilities for optimizing resource usage."""

import gc
import psutil
import time
from typing import Optional, Dict, Any, Callable
from logger_config import get_logger

logger = get_logger(__name__)


class MemoryManager:
    """Memory management and monitoring system."""

    def __init__(
        self,
        threshold_mb: float = 2000,
        auto_gc: bool = True,
    ):
        """Initialize memory manager.

        Args:
            threshold_mb: Memory threshold in MB for warnings
            auto_gc: Enable automatic garbage collection
        """
        self.threshold_mb = threshold_mb
        self.auto_gc = auto_gc

        self.process = psutil.Process()
        self.initial_memory = self.get_memory_usage()

        logger.info(
            f"MemoryManager initialized (threshold: {threshold_mb}MB, "
            f"current: {self.initial_memory:.1f}MB)"
        )

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB.

        Returns:
            Memory usage in megabytes
        """
        return self.process.memory_info().rss / 1024 / 1024

    def get_memory_info(self) -> Dict[str, Any]:
        """Get detailed memory information.

        Returns:
            Dictionary with memory statistics
        """
        mem = self.process.memory_info()
        virtual_mem = psutil.virtual_memory()

        return {
            "rss_mb": mem.rss / 1024 / 1024,
            "vms_mb": mem.vms / 1024 / 1024,
            "percent": self.process.memory_percent(),
            "available_mb": virtual_mem.available / 1024 / 1024,
            "total_mb": virtual_mem.total / 1024 / 1024,
            "system_percent": virtual_mem.percent,
        }

    def check_memory(self) -> bool:
        """Check if memory usage is within threshold.

        Returns:
            True if within threshold, False otherwise
        """
        current_mb = self.get_memory_usage()

        if current_mb > self.threshold_mb:
            logger.warning(
                f"Memory usage high: {current_mb:.1f}MB "
                f"(threshold: {self.threshold_mb}MB)"
            )
            return False

        return True

    def force_gc(self) -> Dict[str, int]:
        """Force garbage collection.

        Returns:
            Dictionary with GC statistics
        """
        logger.debug("Forcing garbage collection")

        before_mb = self.get_memory_usage()

        # Collect all generations
        collected = {
            "gen0": gc.collect(0),
            "gen1": gc.collect(1),
            "gen2": gc.collect(2),
        }

        after_mb = self.get_memory_usage()
        freed_mb = before_mb - after_mb

        logger.info(
            f"GC completed: freed {freed_mb:.1f}MB "
            f"(collected: {sum(collected.values())} objects)"
        )

        return {
            **collected,
            "freed_mb": freed_mb,
            "before_mb": before_mb,
            "after_mb": after_mb,
        }

    def auto_manage(self) -> None:
        """Automatically manage memory based on usage.

        Performs garbage collection if memory exceeds threshold.
        """
        if not self.auto_gc:
            return

        if not self.check_memory():
            self.force_gc()

    def monitor_function(self, func: Callable) -> Callable:
        """Decorator to monitor memory usage of a function.

        Args:
            func: Function to monitor

        Returns:
            Wrapped function with memory monitoring

        Example:
            @memory_manager.monitor_function
            def process_data():
                # ... processing
        """

        def wrapper(*args, **kwargs):
            start_mb = self.get_memory_usage()
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_mb = self.get_memory_usage()
                duration = time.time() - start_time
                delta_mb = end_mb - start_mb

                logger.info(
                    f"{func.__name__}: {duration:.1f}s, "
                    f"memory: {start_mb:.1f}MB → {end_mb:.1f}MB "
                    f"({delta_mb:+.1f}MB)"
                )

                # Auto-manage if enabled
                self.auto_manage()

        return wrapper

    def get_top_objects(self, limit: int = 10) -> Dict[str, int]:
        """Get most common object types in memory.

        Args:
            limit: Number of types to return

        Returns:
            Dictionary mapping type name to count
        """
        import sys
        from collections import Counter

        type_counts = Counter(type(obj).__name__ for obj in gc.get_objects())

        return dict(type_counts.most_common(limit))

    def get_summary(self) -> str:
        """Get memory usage summary.

        Returns:
            Formatted summary string
        """
        info = self.get_memory_info()

        return (
            f"Memory Usage:\n"
            f"  Process: {info['rss_mb']:.1f}MB ({info['percent']:.1f}%)\n"
            f"  System: {info['system_percent']:.1f}% "
            f"({info['available_mb']:.0f}MB available)\n"
            f"  Initial: {self.initial_memory:.1f}MB\n"
            f"  Delta: {info['rss_mb'] - self.initial_memory:+.1f}MB"
        )


class MemoryContext:
    """Context manager for monitoring memory usage in a block.

    Example:
        with MemoryContext("data processing") as mem:
            # ... memory-intensive operations
        # Automatically reports memory usage
    """

    def __init__(
        self,
        name: str,
        auto_gc: bool = False,
        log_level: str = "INFO",
    ):
        """Initialize memory context.

        Args:
            name: Name for this memory context
            auto_gc: Run garbage collection on exit
            log_level: Logging level for reports
        """
        self.name = name
        self.auto_gc = auto_gc
        self.log_level = log_level

        self.manager = MemoryManager()
        self.start_mb = 0
        self.start_time = 0

    def __enter__(self):
        """Enter context."""
        self.start_mb = self.manager.get_memory_usage()
        self.start_time = time.time()

        logger.debug(f"[{self.name}] Starting (memory: {self.start_mb:.1f}MB)")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and report."""
        end_mb = self.manager.get_memory_usage()
        duration = time.time() - self.start_time
        delta_mb = end_mb - self.start_mb

        message = (
            f"[{self.name}] Completed in {duration:.1f}s: "
            f"{self.start_mb:.1f}MB → {end_mb:.1f}MB ({delta_mb:+.1f}MB)"
        )

        if self.log_level == "INFO":
            logger.info(message)
        elif self.log_level == "DEBUG":
            logger.debug(message)
        elif self.log_level == "WARNING" and delta_mb > 100:
            logger.warning(message + " - High memory increase!")

        if self.auto_gc and delta_mb > 50:
            logger.debug(f"[{self.name}] Running garbage collection")
            self.manager.force_gc()

        return False


def get_memory_info_string() -> str:
    """Get current memory info as formatted string.

    Returns:
        Formatted memory information
    """
    manager = MemoryManager()
    return manager.get_summary()


def optimize_memory() -> Dict[str, Any]:
    """Optimize memory usage by running garbage collection.

    Returns:
        Dictionary with optimization results
    """
    manager = MemoryManager()
    return manager.force_gc()
