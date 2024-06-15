import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import make_interp_spline

import fastf1
import fastf1.plotting

QUICKLAP_THRESHOLD = 1.25
BANDWIDTH = 0.17


# @brief driver_laptimes_distribution: Visualizae laptime distributions of different drivers
def driver_laptimes_distribution(
    Year: int, EventName: str, SessionName: str, race, post: bool
) -> dict:

    race.load()

    point_finishers = race.drivers[:10]
    driver_laps = race.laps.pick_drivers(point_finishers).pick_quicklaps(
        QUICKLAP_THRESHOLD
    )
    driver_laps_statistics = race.laps.pick_drivers(point_finishers)
    driver_laps = driver_laps.reset_index()

    finishing_order = [race.get_driver(i)["Abbreviation"] for i in point_finishers]

    driver_colors = {
        abv: fastf1.plotting.DRIVER_COLORS[driver]
        for abv, driver in fastf1.plotting.DRIVER_TRANSLATE.items()
    }

    fig, ax = plt.subplots(figsize=(10.8, 10.8), dpi=100)

    driver_laps["LapTime(s)"] = driver_laps["LapTime"].dt.total_seconds()
    driver_laps_statistics["LapTime(s)"] = driver_laps_statistics[
        "LapTime"
    ].dt.total_seconds()

    # Show the distributions
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
    )

    # Show the actual laptimes
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

    twin = ax.twinx()
    twin.spines["right"].set_color("gray")
    twin.yaxis.label.set_color("gray")
    twin.tick_params(axis="y", colors="gray")
    twin.set_ylabel(r"$\mathbf{Mean\ Lap\ Time\ (s)}$", fontsize=14, color="gray")

    ylim_max = 0
    ylim_min = 1000
    mean_laptime_array = []
    xpos_array = []

    # Calculate and plot statistics
    for driver in finishing_order:
        driver_data_all = driver_laps_statistics[
            driver_laps_statistics["Driver"] == driver
        ]["LapTime(s)"]
        driver_data = driver_laps[driver_laps["Driver"] == driver]["LapTime(s)"]
        mean = driver_data_all.mean()
        median = driver_data_all.median()
        lower_68, upper_68 = np.percentile(driver_data, [16, 84])  # 68% interval
        lower_95, upper_95 = np.percentile(driver_data, [2.5, 97.5])  # 95% interval

        ylim_max = max(mean.max(), ylim_max)
        ylim_min = min(mean.min(), ylim_min)

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

    ax.set_xlabel("Driver", fontweight="bold", fontsize=14)
    ax.set_ylabel("Lap Time (s)", fontweight="bold", fontsize=14)

    xnew = np.linspace(xpos_array[0], xpos_array[-1], 300)
    spl = make_interp_spline(xpos_array, mean_laptime_array, k=3)  # B-spline degree 3
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
    suptitle = f"{Year} {EventName} Grand Prix Driver Lap Time Distributions"

    plt.suptitle(
        suptitle,
        fontweight="bold",
        fontsize=16,
    )
    sns.despine(left=True, bottom=True)

    subtitle = "with Statistical Intervals and Tire Compound Labeled"
    bg_color = ax.get_facecolor()
    plt.figtext(
        0.5,
        0.935,
        subtitle,
        ha="center",
        fontsize=14,
        bbox=dict(facecolor=bg_color, alpha=0.5, edgecolor="none"),
    )

    plt.tight_layout()

    filename = "../pic/" + suptitle.replace(" ", "_") + ".png"
    plt.savefig(filename)

    titles_str = (
        suptitle.replace(f"{Year} ", "")
        .replace(f"{EventName} ", "")
        .replace("Grand Prix ", "")
    )

    caption = f"""\
🏎️
« {Year} {EventName} Grand Prix »

• {titles_str}

#F1 #Formula1 #{EventName.replace(" ", "")}GP"""

    return {"filename": filename, "caption": caption, "post": post}
