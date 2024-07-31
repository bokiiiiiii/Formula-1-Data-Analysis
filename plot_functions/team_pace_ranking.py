import seaborn as sns
from matplotlib import pyplot as plt
import fastf1
import textwrap
import fastf1.plotting


# Helper function to load and process race data
def load_race_data(race):
    race.load()
    laps = race.laps.pick_quicklaps()
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
def generate_team_palette(team_order):
    return {team: fastf1.plotting.team_color(team) for team in team_order}


# Helper function to plot team pace ranking
def plot_team_pace_ranking(ax, transformed_laps, team_order, team_palette):
    sns.boxplot(
        data=transformed_laps,
        x="Team",
        y="LapTime (s)",
        hue="Team",
        order=team_order,
        palette=team_palette,
        whiskerprops=dict(color="white"),
        boxprops=dict(edgecolor="grey", alpha=0.7),
        medianprops=dict(color="white"),
        capprops=dict(color="white"),
        linewidth=1,
        width=0.5,
        ax=ax,
    )

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
    team_palette = generate_team_palette(team_order)

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
