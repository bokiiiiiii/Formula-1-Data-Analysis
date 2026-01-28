"""Simplified plot runner for easier execution and management."""

from typing import Dict, Any, Optional, Callable
from pathlib import Path
import fastf1
from matplotlib import pyplot as plt

from logger_config import get_logger
from config import Config
from plot_functions import utils

logger = get_logger(__name__)


class PlotRunner:
    """Simplified plot execution runner.

    Handles:
    - Session caching to avoid redundant loads
    - Standard plot setup and teardown
    - Error handling and logging
    - Result aggregation
    """

    def __init__(self, config: Config):
        """Initialize PlotRunner.

        Args:
            config: Configuration object
        """
        self.config = config
        self.loaded_sessions: Dict[tuple, fastf1.core.Session] = {}
        self.results: Dict[str, Dict[str, Any]] = {}

        Path(config.folder_path).mkdir(parents=True, exist_ok=True)

        utils.setup_matplotlib_style(
            utils.PlotConfig(
                dpi=config.figure_dpi,
                fig_width_inch=config.figure_width,
                fig_height_inch=config.figure_height,
            )
        )
        utils.enable_cache(config.cache_path)
        plt.ion()

    def get_or_load_session(
        self, year: int, event_name: str, session_type: str
    ) -> Optional[fastf1.core.Session]:
        """Get or load a FastF1 session with caching.

        Args:
            year: Season year
            event_name: Event name
            session_type: Session type (FP1, FP2, FP3, Q, SS, S, R)

        Returns:
            FastF1 session object or None if error
        """
        session_key = (year, event_name, session_type)

        if session_key in self.loaded_sessions:
            logger.debug(f"Using cached session: {event_name} - {session_type}")
            return self.loaded_sessions[session_key]

        try:
            logger.info(f"Loading session: {event_name} - {session_type}")
            session = fastf1.get_session(year, event_name, session_type)
            session.load()
            self.loaded_sessions[session_key] = session
            return session
        except Exception as e:
            logger.error(f"Failed to load session {event_name} {session_type}: {e}")
            return None

    def run_plot(
        self,
        plot_name: str,
        year: int,
        event_name: str,
        session_name: str,
        race: fastf1.core.Session,
        post: bool = False,
        plot_func: Optional[Callable] = None,
    ) -> bool:
        """Run a single plot function.

        Args:
            plot_name: Name of the plot function
            year: Season year
            event_name: Event name
            session_name: Session display name
            race: FastF1 session object
            post: Whether to post to Instagram
            plot_func: Plot function to execute

        Returns:
            True if successful, False otherwise
        """
        try:
            if plot_func is None:
                logger.warning(f"Plot function not provided: {plot_name}")
                return False

            logger.info(f"Running: {plot_name}")
            result = plot_func(year, event_name, session_name, race, post)

            if not isinstance(result, dict):
                logger.error(f"{plot_name} returned invalid result type")
                return False

            self.results[plot_name] = result
            logger.info(f"Completed: {plot_name}")
            return True

        except Exception as e:
            logger.error(f"Error running {plot_name}: {str(e)}", exc_info=True)
            self.results[plot_name] = {
                "filename": None,
                "caption": f"Error: {str(e)}",
                "post": False,
            }
            return False

    def run_all(
        self, year: int, event_name: str, session_name: str, post: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """Run all enabled plot functions.

        Args:
            year: Season year
            event_name: Event name
            session_name: Session identifier (e.g., "R", "Q")
            post: Whether to post to Instagram

        Returns:
            Dictionary of results
        """
        logger.info("=" * 60)
        logger.info(f"Starting F1 Data Analysis")
        logger.info(f"Year: {year}, Session: {session_name}")
        logger.info("=" * 60)

        available_plots = list(self.config.plot_functions.keys())

        enabled_plots = []
        for name in available_plots:
            plot_config = self.config.plot_functions.get(name)
            if plot_config is None:
                continue
            is_enabled = self.config.enable_all or plot_config.enabled
            matches_session = plot_config.session == session_name
            if is_enabled and matches_session:
                enabled_plots.append(name)

        logger.info(
            f"Processing {len(enabled_plots)} plot functions for session {session_name}"
        )

        import importlib

        plot_functions_map = {}

        for plot_name in enabled_plots:
            try:
                module = importlib.import_module(f"plot_functions.{plot_name}")
                plot_func = getattr(module, plot_name, None)
                if plot_func and callable(plot_func):
                    plot_functions_map[plot_name] = plot_func
                else:
                    logger.warning(
                        f"Function {plot_name} not found in module {plot_name}"
                    )
            except ImportError as e:
                logger.warning(f"Could not import module {plot_name}: {e}")

        for plot_name in enabled_plots:
            try:
                if plot_name not in plot_functions_map:
                    logger.warning(f"Plot function not found: {plot_name}")
                    continue

                plot_func = plot_functions_map[plot_name]
                session_config = self.config.plot_functions.get(plot_name)
                if session_config is None:
                    logger.warning(f"No config found for {plot_name}")
                    continue

                session_type = (
                    session_config.session if session_config else session_name
                )

                race = self.get_or_load_session(year, event_name, session_type)
                if race is None:
                    logger.warning(f"Skipping {plot_name} (session load failed)")
                    continue

                self.run_plot(
                    plot_name=plot_name,
                    year=year,
                    event_name=event_name,
                    session_name=session_type,
                    race=race,
                    post=post,
                    plot_func=plot_func,
                )

            except Exception as e:
                logger.error(f"Error processing {plot_name}: {e}", exc_info=True)

        logger.info("=" * 60)
        logger.info(f"Completed {len(self.results)} plots")
        logger.info("=" * 60)

        return self.results

    def cleanup(self) -> None:
        """Clean up resources."""
        plt.close("all")
        self.loaded_sessions.clear()
