"""Utility functions for F1 data visualization.

This module contains common functions used across multiple plotting modules
to avoid code duplication and improve maintainability.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import fastf1
import fastf1.plotting
from matplotlib.figure import Figure

# Configure logging
logger = logging.getLogger(__name__)

# Ignore specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="fastf1")
warnings.filterwarnings("ignore", category=UserWarning, module="fastf1")

# Compound colors - fallback for newer FastF1 versions that removed COMPOUND_COLORS
COMPOUND_COLORS = {
    "SOFT": "#FF3333",
    "MEDIUM": "#FFCC00",
    "HARD": "#CCCCCC",
    "INTERMEDIATE": "#39B54A",
    "WET": "#00AEEF",
    "UNKNOWN": "#000000",
    "TEST_UNKNOWN": "#808080",
}


def get_compound_color(compound: str) -> str:
    """Get color for tire compound.

    Args:
        compound: Tire compound name (SOFT, MEDIUM, HARD, etc.)

    Returns:
        Color hex string
    """
    # Try FastF1's get_compound_color first (for newer versions)
    if hasattr(fastf1.plotting, "get_compound_color"):
        try:
            return fastf1.plotting.get_compound_color(compound, session=None)
        except Exception:
            pass
    # Try legacy COMPOUND_COLORS
    if hasattr(fastf1.plotting, "COMPOUND_COLORS"):
        return fastf1.plotting.COMPOUND_COLORS.get(
            compound, COMPOUND_COLORS.get(compound, "#808080")
        )
    # Use our fallback
    return COMPOUND_COLORS.get(compound, "#808080")


def get_point_finishers_abbr(race: fastf1.core.Session, top_n: int = 10) -> List[str]:
    """Get abbreviations of drivers who finished in points positions.

    Args:
        race: FastF1 session object
        top_n: Number of top finishers to return (default 10 for points)

    Returns:
        List of driver abbreviations for point finishers
    """
    if race.results is None or race.results.empty:
        logger.warning("No results available")
        return []

    # Get finishers sorted by position
    results = race.results.copy()

    # Filter out drivers who didn't finish if ClassifiedPosition exists
    if "ClassifiedPosition" in results.columns:
        results = results[results["ClassifiedPosition"].notna()]

    # Get top N finishers
    if len(results) > top_n:
        results = results.head(top_n)

    return list(results["Abbreviation"])


class PlotConfig:
    """Configuration class for plot settings."""

    def __init__(
        self,
        dpi: int = 125,
        fig_width_inch: float = 8.64,
        fig_height_inch: float = 10.8,
        font_size_title: int = 18,
        font_size_subtitle: int = 15,
        font_size_label: int = 14,
    ):
        """Initialize plot configuration.

        Args:
            dpi: Dots per inch for figure resolution
            fig_width_inch: Figure width in inches
            fig_height_inch: Figure height in inches
            font_size_title: Main title font size
            font_size_subtitle: Subtitle font size
            font_size_label: Axis label font size
        """
        self.dpi = dpi
        self.fig_size = (fig_width_inch, fig_height_inch)
        self.font_size_title = font_size_title
        self.font_size_subtitle = font_size_subtitle
        self.font_size_label = font_size_label


def load_race_data(
    race: fastf1.core.Session,
    telemetry: bool = True,
    laps: bool = True,
    weather: bool = False,
    logger_obj: Optional[logging.Logger] = None,
) -> None:
    """Load race data with error handling.

    Args:
        race: FastF1 session object
        telemetry: Whether to load telemetry data
        laps: Whether to load lap data
        weather: Whether to load weather data
        logger_obj: Logger instance for logging

    Raises:
        RuntimeError: If data loading fails
    """
    log = logger_obj or logger
    try:
        race.load(telemetry=telemetry, laps=laps, weather=weather)
        event_name = race.event.get("EventName", "Unknown Event")
        session_name = race.session_info.get("Name", "Unknown Session")
        log.info(f"Successfully loaded {event_name} {session_name} data")
    except Exception as e:
        try:
            event_name = race.event.get("EventName", "Unknown Event")
        except:
            event_name = "Unknown Event"
        log.error(f"Error loading {event_name} data: {str(e)}")
        raise RuntimeError(f"Error loading race data: {e}") from e


def get_driver_abbreviations(
    race: fastf1.core.Session,
    num_drivers: int = 1,
    logger_obj: Optional[logging.Logger] = None,
) -> List[str]:
    """Get driver abbreviations from race results.

    Args:
        race: FastF1 session object
        num_drivers: Number of top drivers to return
        logger_obj: Logger instance for logging

    Returns:
        List of driver abbreviations
    """
    log = logger_obj or logger

    if race.results is None or race.results.empty:
        log.warning(f"No results available for {race.event_name}")
        return []

    available_drivers = len(race.results)
    requested_drivers = min(num_drivers, available_drivers)

    abbrs = list(race.results["Abbreviation"][:requested_drivers])
    log.debug(f"Retrieved {len(abbrs)} driver(s): {abbrs}")
    return abbrs


def setup_matplotlib_style(config: PlotConfig) -> None:
    """Setup matplotlib style and parameters.

    Args:
        config: PlotConfig instance
    """
    fastf1.plotting.setup_mpl(
        mpl_timedelta_support=False,
        color_scheme=None,
        misc_mpl_mods=False,
    )

    # Set rcParams
    plt.rcParams["figure.dpi"] = config.dpi
    plt.rcParams["savefig.dpi"] = config.dpi
    plt.rcParams["figure.autolayout"] = False
    plt.rcParams["figure.constrained_layout.use"] = False
    plt.rcParams["savefig.bbox"] = None


def create_figure_and_axis(
    config: PlotConfig,
) -> Tuple[Figure, matplotlib.axes.Axes]:
    """Create a figure and axis with proper configuration.

    Args:
        config: PlotConfig instance

    Returns:
        Tuple of (figure, axis)
    """
    fig, ax = plt.subplots(figsize=config.fig_size, dpi=config.dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    return fig, ax


def save_plot(
    fig: Figure,
    filename: str,
    config: PlotConfig,
    folder_path: str = "../pic",
    logger_obj: Optional[logging.Logger] = None,
) -> str:
    """Save plot to file.

    Args:
        fig: Matplotlib figure object
        filename: Name for the saved file (without .png extension)
        config: PlotConfig instance
        folder_path: Directory to save the plot
        logger_obj: Logger instance for logging

    Returns:
        Full path to saved file
    """
    log = logger_obj or logger

    # Create folder if it doesn't exist
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    # Clean filename
    safe_filename = (
        filename.replace(" ", "_").replace(":", "").replace("/", "_").replace("\\", "_")
    )

    filepath = os.path.join(folder_path, f"{safe_filename}.png")

    try:
        fig.savefig(filepath, dpi=config.dpi, bbox_inches=None)
        log.info(f"Plot saved to {filepath}")
        return filepath
    except Exception as e:
        log.error(f"Failed to save plot: {str(e)}")
        raise


def set_axis_labels(
    ax: matplotlib.axes.Axes,
    xlabel: str,
    ylabel: str,
    config: PlotConfig,
) -> None:
    """Set axis labels with consistent styling.

    Args:
        ax: Matplotlib axis
        xlabel: X-axis label
        ylabel: Y-axis label
        config: PlotConfig instance
    """
    ax.set_xlabel(xlabel, fontsize=config.font_size_label, color="black")
    ax.set_ylabel(ylabel, fontsize=config.font_size_label, color="black")
    ax.tick_params(axis="both", colors="black", labelsize=12)


def set_title(
    fig: Figure,
    main_title: str,
    subtitle_upper: str = "",
    subtitle_lower: str = "",
    config: Optional[PlotConfig] = None,
) -> None:
    """Set figure titles with consistent styling.

    Args:
        fig: Matplotlib figure
        main_title: Main title text
        subtitle_upper: Upper subtitle text
        subtitle_lower: Lower subtitle text
        config: PlotConfig instance (uses defaults if None)
    """
    cfg = config or PlotConfig()

    plt.suptitle(main_title, fontsize=cfg.font_size_title, color="black", weight="bold")

    if subtitle_upper:
        plt.figtext(
            0.5,
            0.94,
            subtitle_upper,
            ha="center",
            fontsize=cfg.font_size_subtitle,
            color="black",
        )

    if subtitle_lower:
        plt.figtext(
            0.5,
            0.915,
            subtitle_lower,
            ha="center",
            fontsize=cfg.font_size_subtitle - 2,
            color="black",
        )


def get_driver_laps_cleaned(
    race: fastf1.core.Session,
    driver_abbr: str,
    quicklap_threshold: float = 1.05,
    logger_obj: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Get cleaned lap data for a specific driver.

    Args:
        race: FastF1 session object
        driver_abbr: Driver abbreviation
        quicklap_threshold: Threshold for picking quick laps
        logger_obj: Logger instance for logging

    Returns:
        DataFrame with cleaned lap data
    """
    log = logger_obj or logger

    try:
        driver_number = race.get_driver(driver_abbr)["DriverNumber"]
    except Exception as e:
        log.warning(f"Could not get driver number for {driver_abbr}: {str(e)}")
        return pd.DataFrame()

    try:
        laps = race.laps.pick_drivers(driver_number).pick_quicklaps(quicklap_threshold)
    except Exception as e:
        log.warning(f"Error picking quick laps for {driver_abbr}: {str(e)}")
        laps = race.laps.pick_drivers(driver_number)

    if laps.empty:
        log.debug(f"No laps found for driver {driver_abbr}")
        return pd.DataFrame()

    # Add lap time in seconds if not present
    if "LapTime(s)" not in laps.columns:
        laps = laps.copy()
        laps.loc[:, "LapTime(s)"] = laps["LapTime"].dt.total_seconds()

    # Standardize compound names
    if "Compound" in laps.columns:
        laps = laps.copy()
        laps.loc[:, "Compound"] = laps["Compound"].fillna("UNKNOWN")
        laps.loc[:, "Compound"] = (
            laps["Compound"]
            .astype(str)
            .replace(
                {
                    "UNKOWN": "UNKNOWN",
                    "nan": "UNKNOWN",
                    "None": "UNKNOWN",
                    "": "UNKNOWN",
                }
            )
        )
        valid_compounds = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN"]
        laps.loc[~laps["Compound"].isin(valid_compounds), "Compound"] = "UNKNOWN"
    else:
        laps = laps.copy()
        laps.loc[:, "Compound"] = "UNKNOWN"

    # Add stint lap number if not present
    if "StintLapNumber" not in laps.columns:
        laps["StintLapNumber"] = (
            laps.groupby("Stint")["LapNumber"]
            .rank(method="first", ascending=True)
            .astype(int)
        )

    return laps


def get_stints_info_for_driver(
    race: fastf1.core.Session,
    driver_abbr: str,
    logger_obj: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Get stint information for a driver.

    Args:
        race: FastF1 session object
        driver_abbr: Driver abbreviation
        logger_obj: Logger instance for logging

    Returns:
        DataFrame with stint information
    """
    log = logger_obj or logger

    try:
        driver_number = race.get_driver(driver_abbr)["DriverNumber"]
    except Exception as e:
        log.warning(f"Could not get driver number for {driver_abbr}: {str(e)}")
        return pd.DataFrame()

    stints_df = race.laps.pick_drivers(driver_number)[
        ["Stint", "LapNumber", "Compound"]
    ]

    stint_summary = (
        stints_df.groupby("Stint")
        .agg(
            StintEndLap=("LapNumber", "max"),
            Compound=("Compound", "first"),
        )
        .reset_index()
    )

    return stint_summary


def format_lap_time(
    lap_time_seconds: float,
) -> str:
    """Format lap time from seconds to MM:SS.mmm format.

    Args:
        lap_time_seconds: Lap time in seconds

    Returns:
        Formatted lap time string
    """
    if pd.isna(lap_time_seconds) or lap_time_seconds is None:
        return "N/A"

    minutes = int(lap_time_seconds // 60)
    seconds = lap_time_seconds % 60
    return f"{minutes}:{seconds:06.3f}"


def validate_race_results(
    race: fastf1.core.Session,
    min_drivers: int = 1,
    logger_obj: Optional[logging.Logger] = None,
) -> bool:
    """Validate race results.

    Args:
        race: FastF1 session object
        min_drivers: Minimum number of drivers required
        logger_obj: Logger instance for logging

    Returns:
        True if results are valid, False otherwise
    """
    log = logger_obj or logger

    if race.results is None or race.results.empty:
        log.warning(f"No results available for {race.event_name}")
        return False

    if len(race.results) < min_drivers:
        log.warning(
            f"Not enough drivers in results ({len(race.results)} < {min_drivers})"
        )
        return False

    return True


def create_plot_return_dict(
    filename: Optional[str],
    caption: str,
    post: bool,
    success: bool = True,
) -> Dict[str, any]:
    """Create standardized return dictionary for plot functions.

    Args:
        filename: Path to saved plot file
        caption: Instagram caption text
        post: Whether to post to Instagram
        success: Whether the plot was created successfully

    Returns:
        Dictionary with plot information
    """
    return {
        "filename": filename,
        "caption": caption,
        "post": post and success,
        "success": success,
    }


def safe_close_figures() -> None:
    """Safely close all matplotlib figures."""
    try:
        plt.close("all")
    except Exception as e:
        logger.warning(f"Error closing figures: {str(e)}")


def enable_cache(cache_path: str = "../cache") -> None:
    """Enable FastF1 caching.

    Args:
        cache_path: Path to cache directory
    """
    try:
        Path(cache_path).mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(cache_path)
        logger.info(f"Cache enabled at {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to enable cache: {str(e)}")


def create_styled_figure(
    figsize: Tuple[float, float] = (8.64, 10.8),
    dpi: int = 125,
    facecolor: str = "white",
) -> Tuple[Figure, matplotlib.axes.Axes]:
    """Create a matplotlib figure with scienceplots style.

    Args:
        figsize: Figure size in inches (width, height)
        dpi: Dots per inch
        facecolor: Background color

    Returns:
        Tuple of (figure, axis)
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)
    return fig, ax


def apply_scienceplots_style():
    """Apply scienceplots 'science' and 'bright' styles as context manager.

    Returns:
        Context manager for plt.style.context
    """
    return plt.style.context(["science", "bright"])


def configure_plot_params(dpi: int = 125):
    """Configure common matplotlib rcParams for plots.

    Args:
        dpi: Dots per inch for figure and savefig
    """
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = dpi
    plt.rcParams["figure.autolayout"] = False
    plt.rcParams["figure.constrained_layout.use"] = False
    plt.rcParams["savefig.bbox"] = None


# Common constants
DEFAULT_DPI = 125
DEFAULT_FIG_SIZE = (8.64, 10.8)


def setup_fastf1_plotting():
    """Setup FastF1 plotting with standard configuration."""
    fastf1.plotting.setup_mpl(
        mpl_timedelta_support=False,
        color_scheme=None,
        misc_mpl_mods=False,
    )


def load_session_data(race: fastf1.core.Session):
    """Setup matplotlib for session data visualization.

    Note: Session data is already loaded by PlotRunner.
    This function only sets up matplotlib configuration.

    Args:
        race: FastF1 session object
    """
    fastf1.plotting.setup_mpl(
        mpl_timedelta_support=False,
        color_scheme=None,
        misc_mpl_mods=False,
    )
    # Session is already loaded by PlotRunner - no need to load again


def save_plot_to_file(fig, title: str, dpi: int = DEFAULT_DPI) -> str:
    """Save plot to file with standardized naming.

    Args:
        fig: Matplotlib figure object
        title: Plot title (will be sanitized for filename)
        dpi: DPI for saving

    Returns:
        Path to saved file
    """
    filename = (
        f"../pic/{title.replace(' ', '_').replace(':', '').replace('/', '_')}.png"
    )
    fig.savefig(filename, dpi=dpi, bbox_inches=None)
    return filename


def create_instagram_caption(
    year: int, event_name: str, plot_title: str, description: str, hashtags: str = ""
) -> str:
    """Create standardized Instagram caption.

    Args:
        year: Season year
        event_name: Event name (e.g., "Abu Dhabi")
        plot_title: Title of the plot
        description: Main description text
        hashtags: Additional hashtags (optional)

    Returns:
        Formatted caption string
    """
    base_hashtags = f"#F1 #Formula1 #{event_name.replace(' ', '')}GP"
    if hashtags:
        base_hashtags = f"{base_hashtags} {hashtags}"

    return textwrap.dedent(
        f"""\
    🏎️
    « {year} {event_name} Grand Prix »

    • {plot_title}

    {description}

    {base_hashtags}"""
    )


def get_top_n_finishers(race, n: int = 10) -> list[str]:
    """Get abbreviations of top N finishers.

    Args:
        race: FastF1 session object
        n: Number of finishers to return

    Returns:
        List of driver abbreviations
    """
    if race.results is None or race.results.empty:
        logger.warning("Race results not loaded or empty")
        return []
    return list(race.results.loc[race.results["Position"] <= n, "Abbreviation"])


def get_winner(race) -> list[str]:
    """Get winner abbreviation.

    Args:
        race: FastF1 session object

    Returns:
        List containing winner abbreviation
    """
    return get_top_n_finishers(race, n=1)


def get_podium_finishers(race) -> list[str]:
    """Get podium finishers abbreviations.

    Args:
        race: FastF1 session object

    Returns:
        List of top 3 driver abbreviations
    """
    return get_top_n_finishers(race, n=3)
