"""Utility functions for F1 data visualization.

This module contains common functions used across multiple plotting modules
to avoid code duplication and improve maintainability.
"""

from openf1_data import HistoricalSession, quick_laps
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import textwrap
from contextvars import ContextVar
from contextlib import contextmanager

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401 - registers the science, bright and no-latex styles
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


_output_dir = ContextVar("plot_output_dir", default=Path("../pic"))


@contextmanager
def plot_directory(folder):
    token = _output_dir.set(Path(folder))
    try:
        yield
    finally:
        _output_dir.reset(token)


def plot_path(title, folder=None):
    directory = Path(folder) if folder is not None else _output_dir.get()
    directory.mkdir(parents=True, exist_ok=True)
    name = title.replace(" ", "_").replace(":", "").replace("/", "_").replace(chr(92), "_")
    return str(directory / f"{name}.png")


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
    return COMPOUND_COLORS.get(compound, "#808080")


def get_point_finishers_abbr(race: HistoricalSession, top_n: int = 10) -> List[str]:
    """Get abbreviations of drivers who finished in points positions.

    Args:
        race: historical session
        top_n: Number of top finishers to return (default 10 for points)

    Returns:
        List of driver abbreviations for point finishers
    """
    if race.results is None or race.results.empty:
        logger.warning("No results available")
        return []

    results = race.results.copy()

    if "ClassifiedPosition" in results.columns:
        results = results[results["ClassifiedPosition"].notna()]

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


def get_driver_abbreviations(
    race: HistoricalSession,
    num_drivers: int = 1,
    logger_obj: Optional[logging.Logger] = None,
) -> List[str]:
    """Get driver abbreviations from race results.

    Args:
        race: historical session
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
    folder_path: Optional[str] = None,
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

    filepath = plot_path(filename, folder_path)

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
    race: HistoricalSession,
    driver_abbr: str,
    quicklap_threshold: float = 1.05,
    logger_obj: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Get cleaned lap data for a specific driver.

    Args:
        race: historical session
        driver_abbr: Driver abbreviation
        quicklap_threshold: Threshold for picking quick laps
        logger_obj: Logger instance for logging

    Returns:
        DataFrame with cleaned lap data
    """
    log = logger_obj or logger

    laps = quick_laps(race.driver_laps(driver_abbr), quicklap_threshold)

    if laps.empty:
        log.debug(f"No laps found for driver {driver_abbr}")
        return pd.DataFrame()

    if "LapTime(s)" not in laps.columns:
        laps = laps.copy()
        laps.loc[:, "LapTime(s)"] = laps["LapTime"].dt.total_seconds()

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

    if "StintLapNumber" not in laps.columns:
        laps["StintLapNumber"] = (
            laps.groupby("Stint")["LapNumber"]
            .rank(method="first", ascending=True)
            .astype(int)
        )

    return laps


def get_stints_info_for_driver(
    race: HistoricalSession,
    driver_abbr: str,
    logger_obj: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Get stint information for a driver.

    Args:
        race: historical session
        driver_abbr: Driver abbreviation
        logger_obj: Logger instance for logging

    Returns:
        DataFrame with stint information
    """
    log = logger_obj or logger

    stints_df = race.driver_laps(driver_abbr)[["Stint", "LapNumber", "Compound"]]

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
    race: HistoricalSession,
    min_drivers: int = 1,
    logger_obj: Optional[logging.Logger] = None,
) -> bool:
    """Validate race results.

    Args:
        race: historical session
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
    return plt.style.context(["science", "bright", "no-latex"])


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


DEFAULT_DPI = 125
DEFAULT_FIG_SIZE = (8.64, 10.8)


def save_plot_to_file(fig, title: str, dpi: int = DEFAULT_DPI) -> str:
    """Save plot to file with standardized naming.

    Args:
        fig: Matplotlib figure object
        title: Plot title (will be sanitized for filename)
        dpi: DPI for saving

    Returns:
        Path to saved file
    """
    filename = plot_path(title)
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

    return textwrap.dedent(f"""\
    🏎️
    « {year} {event_name} Grand Prix »

    • {plot_title}

    {description}

    {base_hashtags}""")


def get_top_n_finishers(race, n: int = 10) -> list[str]:
    """Get abbreviations of top N finishers.

    Args:
        race: historical session
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
        race: historical session

    Returns:
        List containing winner abbreviation
    """
    return get_top_n_finishers(race, n=1)


def get_podium_finishers(race) -> list[str]:
    """Get podium finishers abbreviations.

    Args:
        race: historical session

    Returns:
        List of top 3 driver abbreviations
    """
    return get_top_n_finishers(race, n=3)
