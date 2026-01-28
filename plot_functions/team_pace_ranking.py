import seaborn as sns
from matplotlib import pyplot as plt
import fastf1
import textwrap
import fastf1.plotting
import numpy as np
import scienceplots
import matplotlib
from scipy.interpolate import make_interp_spline
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

QUICKLAP_THRESHOLD = 1.05
BOXPLOT_WIDTH = 0.5
B_SPLINE_DEG = 2


def load_and_process_race_data(race):
    race.load()
    laps = race.laps.pick_quicklaps(QUICKLAP_THRESHOLD)
    transformed_laps = laps.copy()
    transformed_laps.loc[:, "LapTime (s)"] = laps["LapTime"].dt.total_seconds()
    return transformed_laps


def compute_team_order_by_median(transformed_laps):
    return (
        transformed_laps[["Team", "LapTime (s)"]]
        .groupby("Team")
        .median()["LapTime (s)"]
        .sort_values()
        .index
    )


def generate_team_color_palette(team_order, race):
    return {team: fastf1.plotting.get_team_color(team, race) for team in team_order}


def plot_team_pace_ranking_styled(ax, transformed_laps, team_order, team_palette):
    boxplot_line_color = "lightgray"
    median_line_color = "dimgray"

    sns.boxplot(
        data=transformed_laps,
        x="Team",
        y="LapTime (s)",
        hue="Team",
        order=team_order,
        palette=team_palette,
        whiskerprops=dict(color=boxplot_line_color, linestyle="--", linewidth=1),
        boxprops=dict(edgecolor=boxplot_line_color, alpha=0.5, linewidth=1),
        medianprops=dict(color=median_line_color, linewidth=1.2),
        capprops=dict(color=boxplot_line_color, linewidth=1),
        showfliers=False,
        linewidth=1,
        width=BOXPLOT_WIDTH,
        ax=ax,
        legend=False,
    )

    sns.swarmplot(
        data=transformed_laps,
        x="Team",
        y="LapTime (s)",
        hue="Team",
        order=team_order,
        palette=team_palette,
        alpha=0.5,
        dodge=False,
        size=4,
        ax=ax,
        legend=False,
    )

    median_lap_times = (
        transformed_laps[["Team", "LapTime (s)"]]
        .groupby("Team")
        .median()["LapTime (s)"]
        .reindex(team_order)
    )

    x_indices = np.arange(len(median_lap_times))
    y_values = median_lap_times.values
    median_trend_line_color = "dimgray"

    if len(x_indices) >= B_SPLINE_DEG + 1 and len(x_indices) > 1:
        spline_func = make_interp_spline(
            x_indices, y_values, k=min(B_SPLINE_DEG, len(x_indices) - 1)
        )
        x_smooth_indices = np.linspace(x_indices.min(), x_indices.max(), 300)
        y_smooth_values = spline_func(x_smooth_indices)
        ax.plot(
            x_smooth_indices,
            y_smooth_values,
            color=median_trend_line_color,
            linestyle="--",
            linewidth=1.2,
        )
    elif len(x_indices) > 0:
        ax.plot(
            x_indices,
            y_values,
            marker="o",
            color=median_trend_line_color,
            linestyle="--",
            markersize=4,
            linewidth=1.2,
        )

    legend_elements = [
        Patch(
            facecolor="grey", edgecolor=boxplot_line_color, alpha=0.5, label="IQR Box"
        ),
        Line2D(
            [0],
            [0],
            color=boxplot_line_color,
            linestyle="--",
            label="1.5 * IQR",
        ),
        Line2D(
            [0],
            [0],
            color=median_line_color,
            linestyle="-",
            linewidth=1.5,
            label="Median Lap Time",
        ),
        Line2D(
            [0],
            [0],
            color=median_trend_line_color,
            linestyle="--",
            linewidth=1.2,
            label="Median Trend",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="grey",
            alpha=0.7,
            markersize=4,
            linestyle="None",
            label="Individual Lap Times",
        ),
    ]

    leg = ax.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=10,
        title_fontsize=12,
        facecolor=ax.get_facecolor(),
        edgecolor=ax.get_facecolor(),
        framealpha=0.7,
    )
    if leg:
        plt.setp(leg.get_texts(), color="black")
        if leg.get_title():
            leg.get_title().set_color("black")

    if not transformed_laps.empty:
        max_lap_time = transformed_laps["LapTime (s)"].max()
        min_lap_time = transformed_laps["LapTime (s)"].min()
        ax.set_ylim(min_lap_time * 0.99, max_lap_time * 1.01)

    ax.set_xlabel("Team", fontsize=14, color="black")
    ax.set_ylabel("Lap Time (s)", fontsize=14, color="black")
    ax.tick_params(axis="x", colors="black", rotation=45, labelsize=10)
    ax.tick_params(axis="y", colors="black", labelsize=10)


def save_plot_and_get_filename(fig, suptitle_text, dpi):
    filename = f"../pic/{suptitle_text.replace(' ', '_').replace(':', '')}.png"
    fig.savefig(filename, dpi=dpi, bbox_inches=None)
    return filename


def generate_styled_caption(year, event_name, suptitle_display_text):
    base_title_for_caption = suptitle_display_text.replace(f"{year} ", "").replace(
        f"{event_name} Grand Prix ", ""
    )

    caption = textwrap.dedent(
        f"""\
    🏎️
    « {year} {event_name} Grand Prix »

    • {base_title_for_caption}
    (Based on quick laps, threshold: {QUICKLAP_THRESHOLD})

    ‣ Box plots show distribution of lap times per team.
    ‣ Swarm plots show individual quick lap times.
    ‣ Dashed line indicates the trend of median lap times across teams.

    #F1 #Formula1 #{event_name.replace(" ", "")}GP #TeamPace"""
    )
    return caption


def team_pace_ranking(
    year: int, event_name: str, session_name: str, race_data, post: bool
) -> dict:
    fastf1.plotting.setup_mpl(
        mpl_timedelta_support=False, color_scheme=None, misc_mpl_mods=False
    )

    DPI = 125
    FIG_SIZE = (1080 / DPI, 1350 / DPI)

    transformed_laps = load_and_process_race_data(race_data)
    if transformed_laps.empty or transformed_laps["Team"].nunique() == 0:
        print(
            f"No valid lap data or teams found for {year} {event_name}. Skipping plot."
        )
        return {
            "filename": None,
            "caption": "Not enough data for team pace ranking.",
            "post": False,
        }

    team_order = compute_team_order_by_median(transformed_laps)
    team_palette = generate_team_color_palette(team_order, race_data)

    with plt.style.context(["science", "bright"]):
        plt.rcParams["figure.dpi"] = DPI
        plt.rcParams["savefig.dpi"] = DPI
        plt.rcParams["figure.autolayout"] = False
        plt.rcParams["figure.constrained_layout.use"] = False
        plt.rcParams["savefig.bbox"] = None

        fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        plot_team_pace_ranking_styled(ax, transformed_laps, team_order, team_palette)

        suptitle_display_text = f"{year} {event_name} Grand Prix: Team Pace Ranking"
        plt.suptitle(suptitle_display_text, fontsize=18, color="black")

        ax.grid(True, linestyle=":", alpha=0.5, color="lightgrey", axis="y")

        suptitle_for_filename = f"{year} {event_name} Grand Prix Team Pace Ranking"
        filename = save_plot_and_get_filename(fig, suptitle_for_filename, DPI)
        plt.close(fig)

    caption = generate_styled_caption(year, event_name, suptitle_display_text)

    return {"filename": filename, "caption": caption, "post": post}
