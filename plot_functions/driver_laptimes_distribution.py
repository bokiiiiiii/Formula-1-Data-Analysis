import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import make_interp_spline
import pandas as pd
import fastf1
import fastf1.plotting
import textwrap
import scienceplots
import matplotlib
from . import utils
from .utils import get_point_finishers_abbr, get_compound_color, COMPOUND_COLORS

QUICKLAP_THRESHOLD = 1.05
BANDWIDTH = 0.17
B_SPLINE_DEG = 1


def load_race_data(race: fastf1.core.Session):
    # Session data should already be loaded by plot_runner
    # This is a no-op now, kept for compatibility
    pass


def get_driver_laps_for_distribution(
    race: fastf1.core.Session,
    driver_abbr_list: list[str],
    pick_quicklaps_threshold: float | None = None,
) -> pd.DataFrame:
    if not driver_abbr_list:
        return pd.DataFrame()

    all_drivers_info = {
        d_info["Abbreviation"]: d_info["DriverNumber"]
        for dn in race.drivers
        if (d_info := race.get_driver(dn)) is not None
    }

    driver_numbers_to_pick = [
        all_drivers_info[abbr] for abbr in driver_abbr_list if abbr in all_drivers_info
    ]

    if not driver_numbers_to_pick:
        print(f"No valid driver numbers found for abbreviations: {driver_abbr_list}")
        return pd.DataFrame()

    laps = race.laps.pick_drivers(driver_numbers_to_pick)

    if pick_quicklaps_threshold is not None:
        laps = laps.pick_quicklaps(pick_quicklaps_threshold)
        laps = laps.dropna(subset=["Compound", "LapTime"])
    else:
        laps = laps.copy()
        laps = laps.dropna(subset=["LapTime"])

    if laps.empty:
        return pd.DataFrame()

    laps["LapTime(s)"] = laps["LapTime"].dt.total_seconds()

    if "DriverNumber" in laps.columns and laps["DriverNumber"].notna().any():
        abbr_map = {num: abbr for abbr, num in all_drivers_info.items()}
        laps["Driver"] = laps["DriverNumber"].map(abbr_map)
    elif (
        "Driver" in laps.columns
        and not laps["Driver"].isin(all_drivers_info.keys()).all()
    ):
        try:
            numeric_drivers = pd.to_numeric(laps["Driver"], errors="coerce")
            if numeric_drivers.notna().any():
                abbr_map = {num: abbr for abbr, num in all_drivers_info.items()}
                laps["Driver"] = numeric_drivers.map(abbr_map)
        except Exception:
            pass

    return laps.dropna(subset=["Driver"])


def get_driver_statistics_laps(
    race: fastf1.core.Session, point_finishers_abbr_list: list[str]
) -> pd.DataFrame:
    """Get all laps (not just quick laps) for statistics calculation."""
    return get_driver_laps_for_distribution(
        race, point_finishers_abbr_list, pick_quicklaps_threshold=None
    )


def get_driver_color_palette(
    race: fastf1.core.Session, driver_abbr_list: list[str]
) -> dict:
    palette = {}
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, abbr in enumerate(driver_abbr_list):
        try:
            team_name = race.get_driver(abbr)["Team"]
            color = fastf1.plotting.get_team_color(team_name, race)
            palette[abbr] = color if color else default_colors[i % len(default_colors)]
        except Exception:
            palette[abbr] = default_colors[i % len(default_colors)]
    return palette


def plot_lap_time_distributions_styled(
    driver_laps_df: pd.DataFrame,
    finishing_order_list: list[str],
    color_palette: dict,
    ax: plt.Axes,
):
    sns.violinplot(
        data=driver_laps_df,
        x="Driver",
        y="LapTime(s)",
        hue="Driver",
        inner=None,
        density_norm="area",
        bw_method=BANDWIDTH,
        order=finishing_order_list,
        palette=color_palette,
        alpha=0.6,
        linewidth=1.0,
        ax=ax,
        legend=False,
    )


def plot_actual_laptimes_styled(
    driver_laps_df: pd.DataFrame, finishing_order_list: list[str], ax: plt.Axes
):
    laps_for_plot = driver_laps_df.copy()
    laps_for_plot["Compound"] = laps_for_plot["Compound"].fillna("UNKNOWN")

    for driver_abbr in finishing_order_list:
        driver_data = laps_for_plot[laps_for_plot["Driver"] == driver_abbr]
        for compound in driver_data["Compound"].unique():
            compound_data = driver_data[driver_data["Compound"] == compound]
            if not compound_data.empty:
                ax.plot(
                    [finishing_order_list.index(driver_abbr)] * len(compound_data),
                    compound_data["LapTime(s)"],
                    "o",
                    color=COMPOUND_COLORS.get(compound, "darkgrey"),
                    markersize=3,
                    alpha=1.0,
                    zorder=5,
                )


def plot_statistical_intervals_styled(
    driver_laps_quick: pd.DataFrame,
    driver_laps_all_stats: pd.DataFrame,
    finishing_order_list: list[str],
    ax: plt.Axes,
) -> tuple[list, list, float, float]:
    ylim_max_stat, ylim_min_stat = 0.0, float("inf")
    mean_laptime_values, xpos_values = [], []
    interval_line_color = "black"
    h_line_offset = 0.05

    for driver_abbr in finishing_order_list:
        driver_quick_laps_times = driver_laps_quick[
            driver_laps_quick["Driver"] == driver_abbr
        ]["LapTime(s)"]
        driver_all_laps_times = driver_laps_all_stats[
            driver_laps_all_stats["Driver"] == driver_abbr
        ]["LapTime(s)"]

        if driver_quick_laps_times.empty or driver_all_laps_times.empty:
            continue

        mean_val = driver_all_laps_times.mean()
        lower_68, upper_68 = np.percentile(driver_quick_laps_times, [16, 84])
        lower_95, upper_95 = np.percentile(driver_quick_laps_times, [2.5, 97.5])

        ylim_max_stat = max(upper_95, ylim_max_stat)
        ylim_min_stat = min(lower_95, ylim_min_stat)

        x_pos = finishing_order_list.index(driver_abbr)
        mean_laptime_values.append(mean_val)
        xpos_values.append(x_pos)

        common_line_args = {"colors": interval_line_color, "alpha": 0.6, "zorder": 6}

        ax.vlines(x_pos, lower_95, upper_95, linestyle=":", lw=0.6, **common_line_args)
        ax.hlines(
            lower_95,
            x_pos - h_line_offset,
            x_pos + h_line_offset,
            linestyle=":",
            lw=0.6,
            **common_line_args,
        )
        ax.hlines(
            upper_95,
            x_pos - h_line_offset,
            x_pos + h_line_offset,
            linestyle=":",
            lw=0.6,
            **common_line_args,
        )

        ax.vlines(x_pos, lower_68, upper_68, linestyle="-", lw=0.8, **common_line_args)
        ax.hlines(
            lower_68,
            x_pos - h_line_offset,
            x_pos + h_line_offset,
            linestyle="-",
            lw=0.8,
            **common_line_args,
        )
        ax.hlines(
            upper_68,
            x_pos - h_line_offset,
            x_pos + h_line_offset,
            linestyle="-",
            lw=0.8,
            **common_line_args,
        )

    if not xpos_values:
        ylim_min_stat, ylim_max_stat = ax.get_ylim()

    return mean_laptime_values, xpos_values, ylim_min_stat, ylim_max_stat


def plot_mean_laptime_styled(
    mean_values: list[float],
    x_positions: list[int],
    y_min_limit: float,
    y_max_limit: float,
    ax: plt.Axes,
) -> plt.Axes | None:
    if not x_positions or not mean_values:
        return None

    twin_ax = ax.twinx()
    mean_line_color = "dimgray"

    twin_ax.yaxis.label.set_color(mean_line_color)
    twin_ax.tick_params(axis="y", colors=mean_line_color)
    twin_ax.set_ylabel("Mean Lap Time (s)", fontsize=14, color=mean_line_color)

    if len(x_positions) > 1 and len(x_positions) == len(mean_values):
        sorted_indices = np.argsort(x_positions)
        sorted_x = np.array(x_positions)[sorted_indices]
        sorted_means = np.array(mean_values)[sorted_indices]

        if len(sorted_x) >= B_SPLINE_DEG + 1:
            x_new_smooth = np.linspace(sorted_x.min(), sorted_x.max(), 300)
            k_spline = min(B_SPLINE_DEG, len(sorted_x) - 1)
            if k_spline > 0:
                spl = make_interp_spline(sorted_x, sorted_means, k=k_spline)
                mean_smoothed_values = spl(x_new_smooth)
                twin_ax.plot(
                    x_new_smooth,
                    mean_smoothed_values,
                    "--",
                    color=mean_line_color,
                    lw=0.9,
                )

    twin_ax.plot(x_positions, mean_values, "o", color=mean_line_color, markersize=2)

    if mean_values:
        twin_ax.set_ylim([min(mean_values) - 0.5, max(mean_values) + 0.5])
    elif y_min_limit != float("inf") and y_max_limit != 0.0:
        twin_ax.set_ylim([y_min_limit - 1, y_max_limit + 1])

    return twin_ax


def add_styled_legends(fig: plt.Figure, ax: plt.Axes):
    legend_handles_stat = [
        Line2D([0], [0], color="black", linestyle="-", lw=0.8, label="68% Interval"),
        Line2D([0], [0], color="black", linestyle=":", lw=0.6, label="95% Interval"),
    ]
    compound_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=COMPOUND_COLORS.get(comp, "darkgrey"),
            label=comp.capitalize(),
            linestyle="None",
            markersize=5,
        )
        for comp in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN"]
    ]

    legend_common_args = {
        "fontsize": 10,
        "title_fontsize": 12,
        "labelcolor": "black",
        "framealpha": 0.5,
    }

    stat_legend = ax.legend(
        handles=legend_handles_stat,
        title=r"Pace Intervals (\%)",
        loc="lower right",
        facecolor=ax.get_facecolor(),
        edgecolor=ax.get_facecolor(),
        **legend_common_args,
    )
    if stat_legend.get_title():
        stat_legend.get_title().set_color("black")

    compound_legend = fig.legend(
        handles=compound_handles,
        title="Tyre Compounds",
        loc="upper left",
        bbox_to_anchor=(0.12, 0.88),
        ncol=1,
        facecolor=fig.get_facecolor(),
        edgecolor=fig.get_facecolor(),
        **legend_common_args,
    )
    if compound_legend.get_title():
        compound_legend.get_title().set_color("black")


def set_main_plot_labels_and_title(
    ax: plt.Axes, year: int, event_name: str, plot_suptitle: str, plot_subtitle: str
):
    ax.set_xlabel("Drivers", fontsize=14, color="black")
    ax.set_ylabel("Lap Time (s)", fontsize=14, color="black")
    ax.tick_params(axis="x", colors="black")
    ax.tick_params(axis="y", colors="black")
    plt.suptitle(plot_suptitle, fontsize=18, color="black")
    plt.figtext(0.5, 0.94, plot_subtitle, ha="center", fontsize=15, color="black")


def save_plot_final(fig: plt.Figure, filename_suptitle: str, dpi_val: int) -> str:
    filename = f"../pic/{filename_suptitle.replace(' ', '_').replace(':', '')}.png"
    fig.savefig(filename, dpi=dpi_val, bbox_inches=None)
    return filename


def create_styled_caption(year: int, event_name: str, caption_title_base: str) -> str:
    return textwrap.dedent(
        f"""\
    🏎️
    « {year} {event_name} Grand Prix »

    • {caption_title_base}
    (Top 10 Finishers)

    ‣ Violin plots show lap time density.
    ‣ Dots indicate actual lap times by tyre compound.
    ‣ Vertical lines show 68% and 95% pace intervals (based on quick laps).
    ‣ Dashed grey line shows mean lap time trend (all laps).

    #F1 #Formula1 #{event_name.replace(" ", "")}GP"""
    )


def driver_laptimes_distribution(
    year: int, event_name: str, session_name: str, race: fastf1.core.Session, post: bool
) -> dict:
    utils.setup_fastf1_plotting()

    DPI = utils.DEFAULT_DPI
    FIG_SIZE = (1080 / DPI, 1350 / DPI)

    try:
        load_race_data(race)
    except RuntimeError as e:
        print(e)
        return {"filename": None, "caption": "Error loading race data.", "post": False}

    finishing_order_abbr = get_point_finishers_abbr(race)
    if not finishing_order_abbr:
        print(
            f"Not enough data for {year} {event_name} (no point finishers). Skipping plot."
        )
        return {"filename": None, "caption": "Not enough data for plot.", "post": False}

    driver_laps_quick_df = get_driver_laps_for_distribution(
        race, finishing_order_abbr, QUICKLAP_THRESHOLD
    )
    driver_laps_all_stats_df = get_driver_statistics_laps(race, finishing_order_abbr)

    if driver_laps_quick_df.empty:
        print(f"No suitable quick laps found for {year} {event_name}. Skipping plot.")
        return {
            "filename": None,
            "caption": "No suitable quick laps for plot.",
            "post": False,
        }

    actual_drivers_in_plot = [
        abbr
        for abbr in finishing_order_abbr
        if abbr in driver_laps_quick_df["Driver"].unique()
    ]
    if not actual_drivers_in_plot:
        print(
            f"No drivers left after filtering for quick laps data in {year} {event_name}. Skipping plot."
        )
        return {
            "filename": None,
            "caption": "No drivers with quick laps data.",
            "post": False,
        }

    driver_color_palette = get_driver_color_palette(race, actual_drivers_in_plot)

    plot_suptitle = f"{year} {event_name} Grand Prix: Lap Time Distributions"
    plot_subtitle = "with Statistical Intervals and Tire Compound Labeled"
    filename_suptitle = f"{year} {event_name} Grand Prix Driver Lap Time Distributions"
    caption_title_base = f"{event_name} - Lap Time Distributions"

    with plt.style.context(["science", "bright"]):
        plt.rcParams.update(
            {
                "figure.dpi": DPI,
                "savefig.dpi": DPI,
                "figure.autolayout": False,
                "figure.constrained_layout.use": False,
                "savefig.bbox": None,
            }
        )

        fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        plot_lap_time_distributions_styled(
            driver_laps_quick_df, actual_drivers_in_plot, driver_color_palette, ax
        )
        plot_actual_laptimes_styled(driver_laps_quick_df, actual_drivers_in_plot, ax)

        mean_laps, x_pos, y_min, y_max = plot_statistical_intervals_styled(
            driver_laps_quick_df, driver_laps_all_stats_df, actual_drivers_in_plot, ax
        )

        if y_min != float("inf") and y_max != 0.0:
            ax.set_ylim([y_min - 2, y_max + 2])
        else:
            all_laptimes = driver_laps_quick_df["LapTime(s)"]
            if not all_laptimes.empty:
                ax.set_ylim([all_laptimes.min() - 2, all_laptimes.max() + 2])

        plot_mean_laptime_styled(mean_laps, x_pos, y_min, y_max, ax)

        ax.set_xticks(range(len(actual_drivers_in_plot)))
        ax.set_xticklabels(actual_drivers_in_plot, rotation=45, ha="right")

        set_main_plot_labels_and_title(
            ax, year, event_name, plot_suptitle, plot_subtitle
        )
        add_styled_legends(fig, ax)

        filename = save_plot_final(fig, filename_suptitle, DPI)
        plt.close(fig)

    caption = create_styled_caption(year, event_name, caption_title_base)
    return {"filename": filename, "caption": caption, "post": post}
