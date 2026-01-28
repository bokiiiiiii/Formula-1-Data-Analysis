"""Progress tracking."""

import time
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import threading

from logger_config import get_logger

logger = get_logger(__name__)


class ProgressPhase(Enum):
    """Pipeline progress states."""

    INITIALIZATION = "Initializing..."
    LOADING_SCHEDULE = "Loading event schedule..."
    LOADING_SESSION = "Loading session data..."
    DOWNLOADING_TELEMETRY = "Downloading telemetry..."
    PROCESSING_DATA = "Processing data..."
    GENERATING_PLOTS = "Generating plots..."
    UPLOADING_INSTAGRAM = "Uploading to Instagram..."
    COMPLETED = "Completed"


@dataclass
class ProgressUpdate:
    """Represents a progress update."""

    phase: ProgressPhase
    current: int = 0  # Current item count
    total: int = 0  # Total items
    percentage: float = field(default=0.0, init=False)  # Calculated percentage
    message: str = ""  # Custom message
    details: str = ""  # Additional details
    elapsed_time: float = 0.0  # Elapsed time in seconds
    estimated_remaining: Optional[float] = None  # Estimated time remaining

    def __post_init__(self):
        """Calculate percentage."""
        if self.total > 0:
            self.percentage = (self.current / self.total) * 100.0
        else:
            self.percentage = 0.0


class ProgressTracker:
    """Tracks progress of operations."""

    def __init__(self, total_items: int = 0):
        """Initialize ProgressTracker.

        Args:
            total_items: Total number of items to process
        """
        self.total_items = total_items
        self.current_item = 0

        self.phase = ProgressPhase.INITIALIZATION
        self.message = ""
        self.details = ""

        self.start_time = time.time()
        self.last_update_time = self.start_time

        self._lock = threading.RLock()
        self._callbacks = []

        logger.debug(f"ProgressTracker initialized (total={total_items})")

    def set_phase(self, phase: ProgressPhase, message: str = "") -> None:
        """Set current progress state.

        Args:
            phase: Progress state enum value
            message: Optional custom message
        """
        with self._lock:
            self.phase = phase
            if message:
                self.message = message

            logger.info(f"Progress: {phase.value}")
            self._notify_callbacks()

    def update(
        self,
        current: int,
        total: Optional[int] = None,
        message: str = "",
        details: str = "",
    ) -> None:
        """Update progress.

        Args:
            current: Current item count
            total: Total items (optional, uses initial value if not provided)
            message: Progress message
            details: Additional details
        """
        with self._lock:
            self.current_item = current
            if total is not None:
                self.total_items = total

            if message:
                self.message = message
            if details:
                self.details = details

            self.last_update_time = time.time()
            self._notify_callbacks()

    def increment(self, message: str = "", details: str = "") -> None:
        """Increment progress by 1.

        Args:
            message: Progress message
            details: Additional details
        """
        with self._lock:
            self.current_item += 1
            if message:
                self.message = message
            if details:
                self.details = details

            self.last_update_time = time.time()
            self._notify_callbacks()

    def add_callback(self, callback: Callable[[ProgressUpdate], None]) -> None:
        """Register a callback for progress updates.

        Args:
            callback: Function that receives ProgressUpdate
        """
        with self._lock:
            self._callbacks.append(callback)

    def get_update(self) -> ProgressUpdate:
        """Get current progress as ProgressUpdate.

        Returns:
            ProgressUpdate object
        """
        with self._lock:
            elapsed = time.time() - self.start_time

            # Estimate remaining time
            estimated_remaining = None
            if (
                self.current_item > 0
                and self.total_items > 0
                and self.current_item < self.total_items
            ):

                rate = self.current_item / elapsed
                remaining_items = self.total_items - self.current_item
                estimated_remaining = remaining_items / rate if rate > 0 else None

            return ProgressUpdate(
                phase=self.phase,
                current=self.current_item,
                total=self.total_items,
                message=self.message,
                details=self.details,
                elapsed_time=elapsed,
                estimated_remaining=estimated_remaining,
            )

    def _notify_callbacks(self) -> None:
        """Notify all registered callbacks."""
        update = self.get_update()
        for callback in self._callbacks:
            try:
                callback(update)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")

    def finish(self) -> ProgressUpdate:
        """Mark progress as finished.

        Returns:
            Final ProgressUpdate
        """
        with self._lock:
            self.phase = ProgressPhase.COMPLETED
            self.current_item = self.total_items
            return self.get_update()

    def __repr__(self) -> str:
        """String representation."""
        update = self.get_update()
        return (
            f"Progress({self.phase.value}, "
            f"{update.current}/{update.total} ({update.percentage:.1f}%))"
        )


class ProgressLogger:
    """Logs progress to console."""

    @staticmethod
    def log(update: ProgressUpdate) -> None:
        """Log progress update.

        Args:
            update: ProgressUpdate object
        """
        percentage = f"{update.percentage:.0f}%" if update.total > 0 else "N/A"

        # Build progress bar
        bar_length = 30
        if update.total > 0:
            filled = int((update.current / update.total) * bar_length)
            bar = "█" * filled + "░" * (bar_length - filled)
        else:
            bar = "░" * bar_length

        # Build time estimate
        time_str = f"{update.elapsed_time:.0f}s"
        if update.estimated_remaining:
            time_str += f" (~{update.estimated_remaining:.0f}s remaining)"

        # Build message
        msg = (
            f"{update.phase.value} [{bar}] "
            f"{percentage} ({update.current}/{update.total}) - {time_str}"
        )

        if update.message:
            msg += f" - {update.message}"

        logger.info(msg)


class SimpleProgressBar:
    """Simple progress bar for CLI applications."""

    def __init__(self, total: int, description: str = ""):
        """Initialize progress bar.

        Args:
            total: Total items
            description: Bar description
        """
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = time.time()

    def update(self, current: int) -> str:
        """Update progress and return formatted bar.

        Args:
            current: Current item count

        Returns:
            Formatted progress bar string
        """
        self.current = current

        # Calculate progress
        percent = (current / self.total) * 100
        filled = int((current / self.total) * 30)
        bar = "█" * filled + "░" * (30 - filled)

        # Calculate time
        elapsed = time.time() - self.start_time
        if current > 0:
            per_item = elapsed / current
            remaining = (self.total - current) * per_item
        else:
            remaining = 0

        # Format output
        output = (
            f"\r{self.description} [{bar}] "
            f"{percent:.0f}% ({current}/{self.total}) "
            f"Elapsed: {elapsed:.0f}s, Remaining: {remaining:.0f}s"
        )

        return output

    def finish(self) -> str:
        """Finish progress bar.

        Returns:
            Final progress bar string
        """
        elapsed = time.time() - self.start_time
        output = (
            f"\r{self.description} [{'█'*30}] "
            f"100% ({self.total}/{self.total}) "
            f"Completed in {elapsed:.0f}s\n"
        )
        return output
