"""Retry utilities."""

import time
import functools
from typing import Callable, Any, Tuple, Type
from logger_config import get_logger

logger = get_logger(__name__)


def retry_on_failure(
    max_attempts: int = 3,
    delay: float = 5.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Callable[[int, Exception], None] = None,
):
    """Retry with exponential backoff on failure.

    Args:
        max_attempts: Max retries
        delay: Initial delay (seconds)
        backoff: Delay multiplier
        exceptions: Exception types to catch
        on_retry: Optional callback on retry
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            attempt = 1
            current_delay = delay
            last_exception = None

            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise

                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )

                    if on_retry:
                        try:
                            on_retry(attempt, e)
                        except Exception as callback_error:
                            logger.error(f"Retry callback error: {callback_error}")

                    time.sleep(current_delay)
                    current_delay *= backoff
                    attempt += 1

            # Should never reach here
            raise last_exception

        return wrapper

    return decorator


def retry_on_network_error(max_attempts: int = 3, delay: float = 5.0):
    """Specialized retry decorator for network-related errors.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries

    Returns:
        Decorated function with retry logic for network errors
    """
    import requests
    import urllib3

    network_exceptions = (
        requests.exceptions.RequestException,
        urllib3.exceptions.HTTPError,
        ConnectionError,
        TimeoutError,
    )

    return retry_on_failure(
        max_attempts=max_attempts,
        delay=delay,
        exceptions=network_exceptions,
    )


class RetryContext:
    """Context manager for retry logic.

    Example:
        with RetryContext(max_attempts=3) as retry:
            while retry.should_retry():
                try:
                    result = operation()
    """

    def __init__(
        self,
        max_attempts: int = 3,
        delay: float = 5.0,
        backoff: float = 2.0,
    ):
        """Initialize retry context.

        Args:
            max_attempts: Maximum number of attempts
            delay: Initial delay between retries
            backoff: Delay multiplier after each retry
        """
        self.max_attempts = max_attempts
        self.delay = delay
        self.backoff = backoff

        self.attempt = 0
        self.current_delay = delay
        self.last_error = None
        self._success = False

    def __enter__(self):
        """Enter context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        if exc_type is not None and not self._success:
            logger.error(f"RetryContext failed after {self.attempt} attempts")
        return False

    def should_retry(self) -> bool:
        """Check if should attempt/retry.

        Returns:
            True if should attempt, False if max attempts reached
        """
        return self.attempt < self.max_attempts

    def failed(self, error: Exception) -> None:
        """Record a failed attempt.

        Args:
            error: The exception that occurred
        """
        self.attempt += 1
        self.last_error = error

        if self.attempt < self.max_attempts:
            logger.warning(
                f"Attempt {self.attempt}/{self.max_attempts} failed: {error}. "
                f"Retrying in {self.current_delay:.1f}s..."
            )
            time.sleep(self.current_delay)
            self.current_delay *= self.backoff
        else:
            logger.error(f"All {self.max_attempts} attempts failed")

    def success(self) -> None:
        """Mark the operation as successful."""
        self._success = True
        if self.attempt > 1:
            logger.info(f"Operation succeeded on attempt {self.attempt}")


def with_timeout(timeout_seconds: float):
    """Add timeout to function execution.

    Args:
        timeout_seconds: Max execution time
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError(
                    f"{func.__name__} timed out after {timeout_seconds}s"
                )

            # Set the signal handler and alarm
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds))

            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            return result

        return wrapper

    return decorator

    return decorator
