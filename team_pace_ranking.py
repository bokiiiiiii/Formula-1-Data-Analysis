import seaborn as sns
from matplotlib import pyplot as plt

import fastf1
import fastf1.plotting

# @brief team_pace_ranking: Rank race pace of each team
def team_pace_ranking(Year: int, EventName: str, SessionName: str, race): 

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

    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.boxplot(
        data=transformed_laps,
        x="Team",
        y="LapTime (s)",
        hue="Team",
        order=team_order,
        palette=team_palette,
        whiskerprops=dict(color="white"),
        boxprops=dict(edgecolor="white"),
        medianprops=dict(color="grey"),
        capprops=dict(color="white"),
    )


    ax.set_xlabel("Team", fontweight='bold')
    ax.set_ylabel("Lap Time (s)", fontweight='bold')
    
    plt.suptitle(f"{Year} {EventName} Grand Prix Team Pace Ranking", fontweight='bold')
    plt.grid(visible=False)
    
    plt.tight_layout()
    