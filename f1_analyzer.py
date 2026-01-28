"""F1 Data Analyzer."""

import os
import threading
import hashlib
from typing import Optional, Dict, Tuple, List
from pathlib import Path
import pandas as pd
import fastf1

from logger_config import get_logger
from track_data import (
    get_safety_car_probability,
    get_fuel_consumption_factor,
    get_tire_degradation,
)

logger = get_logger(__name__)


class F1Analyzer:
    """Unified manager for F1 session data with caching.

    Handles:
    - Caching to avoid duplicate downloads
    - Thread-safe operations
    - Access to track-specific data
    """

    def __init__(
        self,
        year: int,
        event_name: str,
        session_name: str,
        cache_path: str = "../cache",
    ):
        """Initialize F1Analyzer.

        Args:
            year: F1 season year (e.g., 2025)
            event_name: Event name (e.g., "Abu Dhabi")
            session_name: Session code (FP1, FP2, FP3, Q, R, SS, S)
            cache_path: Path for FastF1 cache directory
        """
        self.year = year
        self.event_name = event_name
        self.session_name = session_name
        self.cache_path = Path(cache_path)

        # Ensure cache directory exists
        self.cache_path.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(self.cache_path))

        # Session state
        self.session: Optional[fastf1.Session] = None
        self.race: Optional[pd.DataFrame] = None
        self.is_loaded = False

        # Thread safety
        self._lock = threading.RLock()

        logger.info(f"F1Analyzer initialized: {year} {event_name} {session_name}")

    @property
    def cache_key(self) -> str:
        """Get unique cache key for this analyzer configuration."""
        key = f"{self.year}_{self.event_name}_{self.session_name}"
        return hashlib.md5(key.encode()).hexdigest()[:8]

    def load_session(self) -> bool:
        """Load F1 session data from FastF1.

        Returns:
            True if loaded successfully, False otherwise
        """
        with self._lock:
            if self.is_loaded:
                logger.debug("Session already loaded, skipping")
                return True

            try:
                logger.info(
                    f"Loading session: {self.year} {self.event_name} "
                    f"{self.session_name}"
                )

                # Load session from FastF1
                self.session = fastf1.get_session(
                    year=self.year,
                    gp=self.event_name,
                    identifier=self.session_name,
                )

                # Load telemetry and pit info
                self.session.load(
                    weather=True,
                    telemetry=True,
                    messages=True,
                )

                # Get race results
                self.race = self.session.results

                self.is_loaded = True
                logger.info("Session loaded successfully")
                return True

            except Exception as e:
                logger.error(f"Failed to load session: {e}", exc_info=True)
                return False

    def get_race_data(self) -> Optional[pd.DataFrame]:
        """Get race results data.

        Returns:
            Race results DataFrame or None if not loaded
        """
        if not self.is_loaded:
            self.load_session()
        return self.race

    def get_driver_laps(
        self,
        driver: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """Get lap data for a specific driver or all drivers.

        Args:
            driver: Driver abbreviation (e.g., "VER") or None for all

        Returns:
            Laps DataFrame or None if not loaded
        """
        if not self.is_loaded:
            self.load_session()

        if self.session is None:
            return None

        try:
            if driver:
                laps = self.session.laps.pick_driver(driver)
            else:
                laps = self.session.laps

            return laps if len(laps) > 0 else None
        except Exception as e:
            logger.error(f"Error getting laps: {e}", exc_info=True)
            return None

    def get_driver_abbreviations(self) -> List[str]:
        """Get list of driver abbreviations in this session.

        Returns:
            List of driver abbreviation codes
        """
        if not self.is_loaded:
            self.load_session()

        if self.race is None:
            return []

        try:
            # Get abbreviations from race results
            abbrs = self.race["Abbreviation"].dropna().unique().tolist()
            return sorted(abbrs)
        except Exception as e:
            logger.error(f"Error getting abbreviations: {e}", exc_info=True)
            return []

    def get_event_schedule(self) -> Optional[pd.DataFrame]:
        """Get event schedule for the season.

        Returns:
            Event schedule DataFrame or None if error
        """
        try:
            schedule = fastf1.get_event_schedule(self.year)
            return schedule
        except Exception as e:
            logger.error(f"Error getting event schedule: {e}", exc_info=True)
            return None

    def get_event_names(self) -> List[str]:
        """Get list of all event names for the season.

        Returns:
            List of event names
        """
        schedule = self.get_event_schedule()
        if schedule is None:
            logger.warning("Could not load event schedule")
            return []

        try:
            events = schedule["EventName"].dropna().unique().tolist()
            return sorted(events)
        except Exception as e:
            logger.error(f"Error getting event names: {e}", exc_info=True)
            return []

    # ========================================================================
    # Track-Specific Data Methods
    # ========================================================================

    def get_safety_car_probability(self) -> float:
        """Get Safety Car probability for current track.

        Returns:
            Probability value (0.0 to 1.0)
        """
        return get_safety_car_probability(self.event_name)

    def get_fuel_consumption_factor(self) -> float:
        """Get fuel consumption factor for current track.

        Returns:
            Fuel consumption factor
        """
        return get_fuel_consumption_factor(self.event_name)

    def get_tire_degradation_profile(self):
        """Get tire degradation profile for current track.

        Returns:
            TireDegradationProfile or None
        """
        return get_tire_degradation(self.event_name)

    def get_session_info(self) -> Dict:
        """Get comprehensive session information.

        Returns:
            Dictionary with session metadata
        """
        if not self.is_loaded:
            self.load_session()

        info = {
            "year": self.year,
            "event_name": self.event_name,
            "session_name": self.session_name,
            "is_loaded": self.is_loaded,
            "driver_count": (
                len(self.get_driver_abbreviations()) if self.is_loaded else 0
            ),
            "safety_car_probability": self.get_safety_car_probability(),
            "fuel_consumption_factor": self.get_fuel_consumption_factor(),
            "cache_key": self.cache_key,
        }

        if self.session is not None:
            try:
                info.update(
                    {
                        "date": str(self.session.date),
                        "session_status": self.session.status,
                    }
                )
            except Exception:
                pass

        return info

    def __repr__(self) -> str:
        """String representation."""
        status = "loaded" if self.is_loaded else "not loaded"
        return (
            f"F1Analyzer({self.year} {self.event_name} {self.session_name} "
            f"[{status}])"
        )

    def __str__(self) -> str:
        """Human-readable string."""
        return f"{self.year} {self.event_name} {self.session_name}"


# ============================================================================
# Global Cache for Analyzers
# ============================================================================

_analyzer_cache: Dict[str, F1Analyzer] = {}
_cache_lock = threading.Lock()


def get_or_create_analyzer(
    year: int,
    event_name: str,
    session_name: str,
    cache_path: str = "../cache",
) -> F1Analyzer:
    """Get or create an F1Analyzer instance with caching.

    Args:
        year: F1 season year
        event_name: Event name
        session_name: Session code
        cache_path: Cache directory path

    Returns:
        F1Analyzer instance
    """
    cache_key = f"{year}_{event_name}_{session_name}"

    with _cache_lock:
        if cache_key not in _analyzer_cache:
            analyzer = F1Analyzer(year, event_name, session_name, cache_path)
            _analyzer_cache[cache_key] = analyzer

        return _analyzer_cache[cache_key]


def clear_analyzer_cache():
    """Clear the analyzer cache."""
    global _analyzer_cache
    with _cache_lock:
        _analyzer_cache.clear()
        logger.info("Analyzer cache cleared")


def get_cached_analyzers() -> Dict[str, F1Analyzer]:
    """Get all cached analyzers.

    Returns:
        Dictionary of cached F1Analyzer instances
    """
    with _cache_lock:
        return _analyzer_cache.copy()
