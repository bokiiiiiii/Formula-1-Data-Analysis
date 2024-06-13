import seaborn as sns
from matplotlib import pyplot as plt

import fastf1, textwrap
import fastf1.plotting


# @brief team_pace_ranking: Rank race pace of each team
def team_pace_ranking(Year: int, EventName: str, SessionName: str, race, post: bool) -> dict:

    race.load()
    laps = race.laps.pick_quicklaps()

    transformed_laps = laps.copy()
    transformed_laps.loc[:, "LapTime (s)"] = laps["LapTime"].dt.total_seconds()

    team_order = (
        transformed_laps[["Team", "LapTime (s)"]]
        .groupby("Team")
        .median()["LapTime (s)"]
        .sort_values()
        .index
    )

    team_palette = {team: fastf1.plotting.team_color(team) for team in team_order}

    fig, ax = plt.subplots(figsize=(10.8, 10.8), dpi=100)

    sns.boxplot(
        data=transformed_laps,
        x="Team",
        y="LapTime (s)",
        hue="Team",
        order=team_order,
        palette=team_palette,
        whiskerprops=dict(color="white"),
        boxprops=dict(edgecolor="white"),
        medianprops=dict(color="white"),
        capprops=dict(color="white"),
        linewidth=1.7,
    )

    ax.set_xlabel("Team", fontweight="bold", fontsize=14)
    ax.set_ylabel("Lap Time (s)", fontweight="bold", fontsize=14)

    suptitle = f"{Year} {EventName} Grand Prix Team Pace Ranking"

    plt.suptitle(
        suptitle,
        fontweight="bold",
        fontsize=16,
    )
    plt.grid(visible=False)

    plt.tight_layout()

    filename = "../pic/" + suptitle.replace(" ", "_") + ".png"
    plt.savefig(filename)
    
    titles_str = (
        suptitle.replace(f"{Year} ", "")
        .replace(f"{EventName} ", "")
        .replace("Grand Prix ", "")
    )
 
    caption = textwrap.dedent(
    f"""\
🏎️
« {Year} {EventName} Grand Prix »

• {titles_str}

#formula1 #{EventName.replace(" ", "")}"""
)   
    
    return {"filename": filename, "caption": caption, "post": post}