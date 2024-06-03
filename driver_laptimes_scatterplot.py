import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import fastf1
import fastf1.plotting

# @brief driver_laptimes_scatterplot: Plot driver lap times variation with pit lap annotations
def driver_laptimes_scatterplot(Year: int, EventName: str, SessionName: str, race):

    race.load()

    podium_finishers = race.drivers[:2]

    fig, ax = plt.subplots(figsize=(10.8, 10.8), dpi=100)

    markers = ['.', '*']
    lines = ['--', ':']
    
    stints_laps = race.laps
    stints = stints_laps[["Driver", "Stint", "LapNumber"]]
    stints = stints.groupby(["Driver", "Stint"])
    stints = stints.count().reset_index()
    print(stints)

    legend_elements = []
    for i, driver in enumerate(podium_finishers):

        driver_laps = race.laps.pick_drivers(driver).pick_quicklaps()
        
        driver_laps["LapTime(s)"] = driver_laps["LapTime"].dt.total_seconds()
       
        driver_laps = driver_laps.reset_index()
        driver_lap = driver_laps.pick_fastest()
        
        sns.scatterplot(data=driver_laps,
                        x="LapNumber",
                        y="LapTime(s)",
                        ax=ax,
                        hue="Compound",
                        palette=fastf1.plotting.COMPOUND_COLORS,
                        marker=markers[i],
                        s=80,
                        linewidth=0)       
        
        
        driver_abbr = race.get_driver(driver)["Abbreviation"]
        driver_name = fastf1.plotting.DRIVER_TRANSLATE[driver_abbr]
        driver_color = fastf1.plotting.DRIVER_COLORS[driver_name]   
        legend_elements.append(Line2D([0], [0], marker=markers[i], color=driver_color, markerfacecolor=driver_color, label=driver_abbr, markersize=10, linestyle=''))
        
        stints_stints = stints.loc[stints["Driver"] == driver_abbr]
        
        pit_lap = 0
        pit_lap_array = []
        for idx, row in stints_stints.iterrows():
            pit_lap += row["LapNumber"]
            pit_lap_array.append(pit_lap)
               
        for pit_lap in pit_lap_array:
            ax.axvline(x=pit_lap, color=driver_color, linestyle=lines[i]) 
            ax.text(pit_lap, driver_laps["LapTime(s)"].max()-0.1, 'Pit Lap', rotation=90, color='grey', verticalalignment='top', horizontalalignment='right')


    ax.set_xlabel("Lap Number", fontweight="bold", fontsize=14)
    ax.set_ylabel("Lap Time (s)", fontweight="bold", fontsize=14)

    plt.suptitle(
        f"{Year} {EventName} Grand Prix Driver Lap Time Variation",
        fontweight="bold",
        fontsize=16,
    )
    
    sns.despine(left=True, bottom=True)

    plt.tight_layout()
    
    ax.legend(title='Compound', loc='upper right')
    fig.legend(title='Drivers', handles=legend_elements, loc='upper left', bbox_to_anchor=(0.06, 0.95))
