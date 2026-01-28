"""Data caching system for intermediate results."""

import pickle
import json
import hashlib
import time
from pathlib import Path
from typing import Any, Optional, Dict, Callable
from datetime import datetime, timedelta
import pandas as pd

from logger_config import get_logger

logger = get_logger(__name__)


class DataCache:
    """Data cache manager for storing intermediate results."""

    def __init__(self, cache_dir: str = "../cache/processed"):
        """Initialize data cache.

        Args:
            cache_dir: Directory for cache storage
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.stats = {
            "hits": 0,
            "misses": 0,
            "writes": 0,
        }

        logger.info(f"DataCache initialized: {self.cache_dir}")

    def get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            MD5 hash of arguments
        """
        # Create a string representation of arguments
        key_parts = []

        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            elif isinstance(arg, (list, tuple)):
                key_parts.append(str(sorted(str(x) for x in arg)))
            elif isinstance(arg, dict):
                key_parts.append(str(sorted(arg.items())))
            else:
                key_parts.append(str(type(arg).__name__))

        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")

        key_string = "_".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(
        self,
        key: str,
        max_age_seconds: Optional[float] = None,
    ) -> Optional[Any]:
        """Get cached data.

        Args:
            key: Cache key
            max_age_seconds: Maximum age of cache entry in seconds

        Returns:
            Cached data or None if not found or expired
        """
        cache_file = self.cache_dir / f"{key}.pkl"

        if not cache_file.exists():
            self.stats["misses"] += 1
            return None

        # Check age if specified
        if max_age_seconds is not None:
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age > max_age_seconds:
                logger.debug(f"Cache expired: {key} (age: {file_age:.0f}s)")
                self.stats["misses"] += 1
                return None

        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)

            self.stats["hits"] += 1
            logger.debug(f"Cache hit: {key}")
            return data

        except Exception as e:
            logger.error(f"Error loading cache {key}: {e}")
            self.stats["misses"] += 1
            return None

    def set(self, key: str, data: Any) -> bool:
        """Store data in cache.

        Args:
            key: Cache key
            data: Data to cache

        Returns:
            True if successful, False otherwise
        """
        cache_file = self.cache_dir / f"{key}.pkl"

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

            self.stats["writes"] += 1
            logger.debug(f"Cache write: {key}")
            return True

        except Exception as e:
            logger.error(f"Error writing cache {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete cached data.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        cache_file = self.cache_dir / f"{key}.pkl"

        if cache_file.exists():
            cache_file.unlink()
            logger.debug(f"Cache deleted: {key}")
            return True

        return False

    def clear(self, older_than_days: Optional[int] = None) -> int:
        """Clear cache entries.

        Args:
            older_than_days: Only delete files older than this many days

        Returns:
            Number of files deleted
        """
        deleted = 0
        cutoff_time = None

        if older_than_days is not None:
            cutoff_time = time.time() - (older_than_days * 86400)

        for cache_file in self.cache_dir.glob("*.pkl"):
            if cutoff_time is None or cache_file.stat().st_mtime < cutoff_time:
                cache_file.unlink()
                deleted += 1

        logger.info(f"Cache cleared: {deleted} files deleted")
        return deleted

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (
            (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        )

        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "writes": self.stats["writes"],
            "hit_rate": hit_rate,
            "total_files": len(cache_files),
            "total_size_mb": total_size / 1024 / 1024,
        }

    def cached(
        self,
        key_func: Optional[Callable] = None,
        max_age_seconds: Optional[float] = None,
    ):
        """Decorator for caching function results.

        Args:
            key_func: Optional function to generate cache key from args
            max_age_seconds: Maximum age of cached result

        Returns:
            Decorated function with caching

        Example:
            @cache.cached(max_age_seconds=3600)
            def expensive_function(year, event):
                # ... expensive computation
                return result
        """

        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs) -> Any:
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self.get_cache_key(func.__name__, *args, **kwargs)

                # Try to get from cache
                cached_data = self.get(cache_key, max_age_seconds)
                if cached_data is not None:
                    logger.info(f"Using cached result for {func.__name__}")
                    return cached_data

                # Compute and cache
                logger.info(f"Computing {func.__name__} (cache miss)")
                result = func(*args, **kwargs)
                self.set(cache_key, result)

                return result

            return wrapper

        return decorator


class SessionDataCache(DataCache):
    """Specialized cache for FastF1 session data."""

    def get_session_key(self, year: int, event: str, session: str) -> str:
        """Get cache key for session data.

        Args:
            year: Season year
            event: Event name
            session: Session identifier

        Returns:
            Cache key
        """
        return self.get_cache_key("session", year, event, session)

    def get_laps_key(
        self,
        year: int,
        event: str,
        session: str,
        driver: Optional[str] = None,
    ) -> str:
        """Get cache key for lap data.

        Args:
            year: Season year
            event: Event name
            session: Session identifier
            driver: Optional driver abbreviation

        Returns:
            Cache key
        """
        return self.get_cache_key("laps", year, event, session, driver or "all")

    def cache_dataframe(
        self,
        key: str,
        df: pd.DataFrame,
        compression: str = "gzip",
    ) -> bool:
        """Cache a pandas DataFrame efficiently.

        Args:
            key: Cache key
            df: DataFrame to cache
            compression: Compression method

        Returns:
            True if successful
        """
        cache_file = self.cache_dir / f"{key}.parquet"

        try:
            df.to_parquet(cache_file, compression=compression)
            self.stats["writes"] += 1
            logger.debug(f"DataFrame cached: {key}")
            return True
        except Exception as e:
            logger.error(f"Error caching DataFrame {key}: {e}")
            return False

    def get_dataframe(
        self,
        key: str,
        max_age_seconds: Optional[float] = None,
    ) -> Optional[pd.DataFrame]:
        """Get cached DataFrame.

        Args:
            key: Cache key
            max_age_seconds: Maximum age

        Returns:
            DataFrame or None
        """
        cache_file = self.cache_dir / f"{key}.parquet"

        if not cache_file.exists():
            self.stats["misses"] += 1
            return None

        if max_age_seconds is not None:
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age > max_age_seconds:
                self.stats["misses"] += 1
                return None

        try:
            df = pd.read_parquet(cache_file)
            self.stats["hits"] += 1
            logger.debug(f"DataFrame cache hit: {key}")
            return df
        except Exception as e:
            logger.error(f"Error loading DataFrame cache {key}: {e}")
            self.stats["misses"] += 1
            return None


# Global cache instance
_global_cache: Optional[DataCache] = None


def get_cache() -> DataCache:
    """Get global cache instance.

    Returns:
        Global DataCache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = DataCache()
    return _global_cache
