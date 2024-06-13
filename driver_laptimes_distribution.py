import seaborn as sns
from matplotlib import pyplot as plt

import fastf1,textwrap
import fastf1.plotting


# @brief driver_laptimes_distribution: Visualizae laptime distributions of different drivers
def driver_laptimes_distribution(Year: int, EventName: str, SessionName: str, race, post: bool) -> dict:

    race.load()

    point_finishers = race.drivers[:10]
    driver_laps = race.laps.pick_drivers(point_finishers).pick_quicklaps()
    driver_laps = driver_laps.reset_index()

    finishing_order = [race.get_driver(i)["Abbreviation"] for i in point_finishers]

    driver_colors = {
        abv: fastf1.plotting.DRIVER_COLORS[driver]
        for abv, driver in fastf1.plotting.DRIVER_TRANSLATE.items()
    }

    fig, ax = plt.subplots(figsize=(10.8, 10.8), dpi=100)

    driver_laps["LapTime(s)"] = driver_laps["LapTime"].dt.total_seconds()

    # Show the distributions
    sns.violinplot(
        data=driver_laps,
        x="Driver",
        y="LapTime(s)",
        hue="Driver",
        inner=None,
        density_norm="area",
        order=finishing_order,
        palette=driver_colors,
        alpha=0.4,
    )

    # Show the actual laptimes
    sns.swarmplot(
        data=driver_laps,
        x="Driver",
        y="LapTime(s)",
        order=finishing_order,
        hue="Compound",
        palette=fastf1.plotting.COMPOUND_COLORS,
        hue_order=["SOFT", "MEDIUM", "HARD"],
        linewidth=0,
        size=4,
    )

    ax.set_xlabel("Driver", fontweight="bold", fontsize=14)
    ax.set_ylabel("Lap Time (s)", fontweight="bold", fontsize=14)

    suptitle = f"{Year} {EventName} Grand Prix Driver Lap Time Distributions"

    plt.suptitle(
        suptitle,
        fontweight="bold",
        fontsize=16,
    )
    sns.despine(left=True, bottom=True)

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
