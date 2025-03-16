import seaborn as sns
from matplotlib import pyplot as plt
import fastf1
import textwrap
import fastf1.plotting
import numpy as np
from scipy.interpolate import make_interp_spline
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# Parameters
QUICKLAP_THRESHOLD = 1.3
WIDTH = 0.5
B_SPLINE_DEG = 2


# Helper function to load and process race data
def load_race_data(race):
    race.load()
    laps = race.laps.pick_quicklaps(QUICKLAP_THRESHOLD)
    transformed_laps = laps.copy()
    transformed_laps["LapTime (s)"] = laps["LapTime"].dt.total_seconds()
    return transformed_laps


# Helper function to compute team order based on median lap times
def compute_team_order(transformed_laps):
    return (
        transformed_laps[["Team", "LapTime (s)"]]
        .groupby("Team")
        .median()["LapTime (s)"]
        .sort_values()
        .index
    )


# Helper function to generate team color palette
def generate_team_palette(team_order, race):
    return {team: fastf1.plotting.get_team_color(team, race) for team in team_order}


# Helper function to plot team pace ranking
def plot_team_pace_ranking(ax, transformed_laps, team_order, team_palette):
    sns.boxplot(
        data=transformed_laps,
        x="Team",
        y="LapTime (s)",
        hue="Team",
        order=team_order,
        palette=team_palette,
        whiskerprops=dict(color="grey", linestyle="--"),
        boxprops=dict(edgecolor="grey", alpha=0.3),
        medianprops=dict(color="white"),
        capprops=dict(color="grey"),
        linewidth=1,
        width=WIDTH,
        ax=ax,
    )

    sns.swarmplot(
        data=transformed_laps,
        x="Team",
        y="LapTime (s)",
        hue="Team",
        order=team_order,
        palette=team_palette,
        alpha=0.75,
        dodge=False,
        ax=ax,
    )

    # Calculate median lap times
    median_lap_times = (
        transformed_laps[["Team", "LapTime (s)"]]
        .groupby("Team")
        .median()["LapTime (s)"]
        .reindex(team_order)
    )

    # Interpolate for smooth line
    x = np.arange(len(median_lap_times))
    y = median_lap_times.values
    spline = make_interp_spline(x, y, k=B_SPLINE_DEG)
    x_smooth = np.linspace(x.min(), x.max(), 300)
    y_smooth = spline(x_smooth)

    ax.plot(
        x_smooth,
        y_smooth,
        color="white",
        linestyle="--",
        linewidth=1,
        label="Median Lap Time",
    )

    # Custom legend
    legend_elements = [
        Patch(
            facecolor="grey",
            edgecolor="grey",
            alpha=0.3,
            label="Interquartile Range (IQR)",
        ),
        Line2D([0], [0], color="white", linestyle="-", linewidth=1, label="Median"),
        Line2D(
            [0],
            [0],
            color="white",
            linestyle="--",
            linewidth=1,
            label="Median Variation",
        ),
        Line2D([0], [0], color="grey", linestyle="-", label="Boundary"),
        Line2D(
            [0],
            [0],
            color="white",
            marker="o",
            linestyle="",
            alpha=0.75,
            label="Lap Times",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=12)

    # Set y-axis limit
    max_lap_time = transformed_laps["LapTime (s)"].max()
    min_lap_time = transformed_laps["LapTime (s)"].min()
    ax.set_ylim(min_lap_time * 0.995, max_lap_time * 1.01)

    ax.set_xlabel("Team", fontweight="bold", fontsize=14)
    ax.set_ylabel("Lap Time (s)", fontweight="bold", fontsize=14)


# Helper function to save the plot
def save_plot(fig, filename):
    plt.tight_layout()
    plt.savefig(filename)


# Helper function to generate the caption for the post
def generate_caption(year, event_name, titles_str):
    return textwrap.dedent(
        f"""\
🏎️
« {year} {event_name} Grand Prix »

• {titles_str}

#F1 #Formula1 #{event_name.replace(" ", "")}GP"""
    )


# Main function to plot team pace ranking and generate post data
def team_pace_ranking(
    year: int, event_name: str, session_name: str, race, post: bool
) -> dict:
    transformed_laps = load_race_data(race)
    team_order = compute_team_order(transformed_laps)
    team_palette = generate_team_palette(team_order, race)

    fig, ax = plt.subplots(figsize=(10.8, 10.8), dpi=100)
    plot_team_pace_ranking(ax, transformed_laps, team_order, team_palette)

    suptitle = f"{year} {event_name} Grand Prix Team Pace Ranking"
    plt.suptitle(suptitle, fontweight="bold", fontsize=16)
    plt.grid(visible=False)

    filename = f"../pic/{suptitle.replace(' ', '_')}.png"
    save_plot(fig, filename)

    titles_str = (
        suptitle.replace(f"{year} ", "")
        .replace(f"{event_name} ", "")
        .replace("Grand Prix ", "")
    )
    caption = generate_caption(year, event_name, titles_str)

    return {"filename": filename, "caption": caption, "post": post}
