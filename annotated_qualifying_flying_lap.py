import seaborn as sns
from matplotlib import pyplot as plt

import fastf1
import fastf1.plotting

# @brief annotated_qualifying_flying_lap: Plot the speed of a qualifying flying lap and add annotations to mark corners
def annotated_qualifying_flying_lap(Year: int, EventName: str, SessionName: str, race):

    race.load()
    fastest_laps = race.laps.pick_quicklaps().groupby("Driver")["LapTime"].nsmallest(1).reset_index(level=1, drop=True).nsmallest(2)
    
    fig, ax = plt.subplots(figsize=(10.8, 10.8), dpi=100)
    
    v_min = float('inf')
    v_max = float('-inf')

    drivers = list(fastest_laps.index)
    teams = [race.laps[race.laps['Driver'] == driver]['Team'].iloc[0] for driver in drivers]
    same_team = teams[0] == teams[1]

    for i, (driver, lap_time) in enumerate(fastest_laps.items()):
        lap = race.laps[race.laps['Driver'] == driver].pick_fastest()
        car_data = lap.get_car_data().add_distance()
        team_color = fastf1.plotting.team_color(lap['Team'])

        linestyle = '-' if i == 0 or not same_team else '--'
        ax.plot(car_data['Distance'], car_data['Speed'], color=team_color, label=lap['Driver'], linestyle=linestyle)
        
        v_min = min(v_min, car_data['Speed'].min())
        v_max = max(v_max, car_data['Speed'].max())
    
    circuit_info = race.get_circuit_info()
    
    ax.vlines(x=circuit_info.corners['Distance'], ymin=v_min-20, ymax=v_max+20, linestyles='dotted', colors='grey')

    for _, corner in circuit_info.corners.iterrows():
        txt = f"{corner['Number']}{corner['Letter']}"
        ax.text(corner['Distance'], v_min-30, txt, va='center_baseline', ha='center', size='small')

    ax.set_xlabel('Distance (m)', fontweight="bold", fontsize=14)
    ax.set_ylabel('Speed (km/h)', fontweight="bold", fontsize=14)
    ax.legend(title="Drivers")

    ax.set_ylim([v_min - 40, v_max + 20])

    plt.suptitle(f"{Year} {EventName} Grand Prix Qualifying Flying Lap Comparison of Front Row", fontweight="bold", fontsize=16)
    plt.tight_layout()
    plt.show()
