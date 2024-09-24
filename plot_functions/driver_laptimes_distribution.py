import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import make_interp_spline
import fastf1
import fastf1.plotting


# Parameters
QUICKLAP_THRESHOLD = 1.05
BANDWIDTH = 0.17
B_SPLINE_DEG = 2


def load_race_data(race):
    """Load race data."""
    try:
        race.load()
    except Exception as e:
        raise RuntimeError(f"Error loading race data: {e}")


def get_point_finishers(race):
    """Get the top 10 finishers."""
    return race.results[:10]["DriverNumber"]


def get_driver_laps(race, point_finishers):
    """Get the laps for the point finishers."""
    return race.laps.pick_drivers(point_finishers).pick_quicklaps(QUICKLAP_THRESHOLD)


def get_driver_statistics(race, point_finishers):
    """Get the lap statistics for the point finishers."""
    return race.laps.pick_drivers(point_finishers)


def get_finishing_order(race, point_finishers):
    """Get the finishing order of drivers."""
    return [race.get_driver(i)["Abbreviation"] for i in point_finishers]


def get_driver_colors():
    """Get the driver colors."""
    return {
        abv: fastf1.plotting.DRIVER_COLORS[driver]
        for abv, driver in fastf1.plotting.DRIVER_TRANSLATE.items()
    }


def plot_lap_time_distributions(driver_laps, finishing_order, driver_colors):
    """Plot the lap time distributions."""
    fig, ax = plt.subplots(figsize=(10.8, 10.8), dpi=100)
    sns.violinplot(
        data=driver_laps,
        x="Driver",
        y="LapTime(s)",
        hue="Driver",
        inner=None,
        density_norm="area",
        bw_method=BANDWIDTH,
        order=finishing_order,
        palette=driver_colors,
        alpha=0.5,
        ax=ax,
    )
    return fig, ax


def plot_actual_laptimes(driver_laps, finishing_order):
    """Plot the actual lap times."""
    sns.swarmplot(
        data=driver_laps,
        x="Driver",
        y="LapTime(s)",
        order=finishing_order,
        hue="Compound",
        palette=fastf1.plotting.COMPOUND_COLORS,
        hue_order=["SOFT", "MEDIUM", "HARD", "INTERMEDIATE"],
        linewidth=0,
        size=4.5,
    )


def plot_statistical_intervals(
    driver_laps, driver_laps_statistics, finishing_order, ax
):
    """Plot the statistical intervals."""
    ylim_max = 0
    ylim_min = 1000
    mean_laptime_array = []
    xpos_array = []

    for driver in finishing_order:
        driver_data_all = driver_laps_statistics[
            driver_laps_statistics["Driver"] == driver
        ]["LapTime(s)"]
        driver_data = driver_laps[driver_laps["Driver"] == driver]["LapTime(s)"]
        mean = driver_data_all.mean()
        lower_68, upper_68 = np.percentile(driver_data, [16, 84])  # 68% interval
        lower_95, upper_95 = np.percentile(driver_data, [2.5, 97.5])  # 95% interval

        ylim_max = max(mean, ylim_max)
        ylim_min = min(mean, ylim_min)

        xpos = finishing_order.index(driver)

        mean_laptime_array.append(mean)
        xpos_array.append(xpos)

        # Plot 95% interval
        ax.vlines(
            xpos,
            lower_95,
            upper_95,
            colors="white",
            linestyle=":",
            lw=1.4,
            label="95% Interval" if driver == finishing_order[0] else "",
        )
        ax.hlines(
            lower_95,
            xpos - 0.12,
            xpos + 0.12,
            colors="white",
            linestyle=":",
            lw=1.4,
            label="Median" if driver == finishing_order[0] else "",
        )
        ax.hlines(
            upper_95,
            xpos - 0.12,
            xpos + 0.12,
            colors="white",
            linestyle=":",
            lw=1.4,
            label="Median" if driver == finishing_order[0] else "",
        )

        # Plot 68% interval
        ax.vlines(
            xpos,
            lower_68,
            upper_68,
            colors="white",
            linestyle="-",
            lw=1.4,
            label="68% Interval" if driver == finishing_order[0] else "",
        )
        ax.hlines(
            lower_68,
            xpos - 0.12,
            xpos + 0.12,
            colors="white",
            linestyle="-",
            lw=1.4,
            label="Median" if driver == finishing_order[0] else "",
        )
        ax.hlines(
            upper_68,
            xpos - 0.12,
            xpos + 0.12,
            colors="white",
            linestyle="-",
            lw=1.4,
            label="Median" if driver == finishing_order[0] else "",
        )

    return mean_laptime_array, xpos_array, ylim_min, ylim_max


def plot_mean_laptime(mean_laptime_array, xpos_array, ylim_min, ylim_max, ax):
    """Plot the mean lap time."""
    twin = ax.twinx()
    twin.yaxis.label.set_color("gray")
    twin.tick_params(axis="y", colors="gray")
    twin.set_ylabel(r"$\mathbf{Mean\ Lap\ Time\ (s)}$", fontsize=14, color="gray")

    xnew = np.linspace(xpos_array[0], xpos_array[-1], 300)
    spl = make_interp_spline(
        xpos_array, mean_laptime_array, k=B_SPLINE_DEG
    )  # B-spline degree
    mean_smooth = spl(xnew)
    twin.plot(xnew, mean_smooth, "--", color="gray")
    twin.plot(
        xpos_array,
        mean_laptime_array,
        "o",
        color="gray",
        markersize=4.5,
        label="Mean Lap Time",
    )
    twin.set_ylim([ylim_min - 1, ylim_max + 1])

    twin.legend(loc="lower right")
    return twin


def add_legend(fig, ax):
    """Add the legend to the plot."""
    legend_handles = [
        Line2D([0], [0], color="white", linestyle="-", lw=1.4, label="68% Interval"),
        Line2D([0], [0], color="white", linestyle=":", lw=1.4, label="95% Interval"),
    ]
    fig.legend(
        title="Statistical Intervals",
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.08, 0.95),
    )


def save_plot(fig, year, event_name):
    """Save the plot to a file."""
    suptitle = f"{year} {event_name} Grand Prix Driver Lap Time Distributions"
    plt.suptitle(suptitle, fontweight="bold", fontsize=16)
    plt.figtext(
        0.5,
        0.935,
        "with Statistical Intervals and Tire Compound Labeled",
        ha="center",
        fontsize=14,
        bbox=dict(facecolor=fig.get_facecolor(), alpha=0.5, edgecolor="none"),
    )
    plt.tight_layout()

    filename = f"../pic/{suptitle.replace(' ', '_')}.png"
    plt.savefig(filename)

    return filename


def create_caption(year, event_name):
    """Create a caption for the plot."""
    titles_str = f"{event_name} Grand Prix"
    caption = f"🏎️\n« {year} {event_name} Grand Prix »\n\n• {titles_str}\n\n#F1 #Formula1 #{event_name.replace(' ', '')}GP"
    return caption


def driver_laptimes_distribution(
    year: int, event_name: str, session_name: str, race, post: bool
) -> dict:
    """Visualize lap time distributions of different drivers."""
    load_race_data(race)

    point_finishers = get_point_finishers(race)
    driver_laps = get_driver_laps(race, point_finishers).reset_index()
    driver_laps_statistics = get_driver_statistics(race, point_finishers)
    finishing_order = get_finishing_order(race, point_finishers)
    driver_colors = get_driver_colors()

    driver_colors["COL"] = "#005aff"  # WORKAROUND

    driver_laps["LapTime(s)"] = driver_laps["LapTime"].dt.total_seconds()
    driver_laps_statistics["LapTime(s)"] = driver_laps_statistics[
        "LapTime"
    ].dt.total_seconds()

    fig, ax = plot_lap_time_distributions(driver_laps, finishing_order, driver_colors)
    plot_actual_laptimes(driver_laps, finishing_order)

    mean_laptime_array, xpos_array, ylim_min, ylim_max = plot_statistical_intervals(
        driver_laps, driver_laps_statistics, finishing_order, ax
    )
    plot_mean_laptime(mean_laptime_array, xpos_array, ylim_min, ylim_max, ax)

    add_legend(fig, ax)
    filename = save_plot(fig, year, event_name)
    caption = create_caption(year, event_name)

    return {"filename": filename, "caption": caption, "post": post}
