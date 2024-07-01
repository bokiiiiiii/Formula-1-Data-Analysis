import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib, textwrap
import fastf1
import fastf1.plotting
import fastf1.utils


# @brief race_fatest_lap_telemetry_data: Plot the telemetry data of the race fastest laps
def race_fatest_lap_telemetry_data(
    Year: int, EventName: str, SessionName: str, race, post: bool
) -> dict:

    race.load()

    race_results = race.results
    front_row = race_results[:2]

    fig, ax = plt.subplots(4, figsize=(10.8, 10.8), dpi=100)

    drivers = list(front_row["Abbreviation"])
    teams = [
        race.laps[race.laps["Driver"] == driver]["Team"].iloc[0] for driver in drivers
    ]
    same_team = teams[0] == teams[1]

    top_speeds = {}
    compared_laps = []
    team_colors = []
    drivers_abbr = []
    lap_time_array = []
    lap_time_str_array = []

    v_min = float("inf")
    v_max = float("-inf")
    d_max = float("-inf")

    for i, driver in enumerate(drivers):
        lap = race.laps[race.laps["Driver"] == driver].pick_fastest()
        compared_laps.append(lap)
        car_data = lap.get_car_data().add_distance()
        team_color = fastf1.plotting.team_color(lap["Team"])
        team_colors.append(team_color)

        lap_time = lap["LapTime"]
        lap_time_str = (
            f"{lap_time.total_seconds() // 60:.0f}:{lap_time.total_seconds() % 60:.3f}"
        )
        lap_time_array.append(lap_time.total_seconds())
        lap_time_str_array.append(lap_time_str)

        linestyle = "-" if i == 0 or not same_team else "--"
        ax[0].plot(
            car_data["Distance"],
            car_data["Speed"],
            color=team_color,
            label=f"{lap['Driver']}: {lap_time_str} (min:s.ms)",
            linestyle=linestyle,
        )

        ax[1].plot(
            car_data["Distance"],
            car_data["Throttle"],
            color=team_color,
            linestyle=linestyle,
        )

        ax[2].plot(
            car_data["Distance"],
            car_data["Brake"],
            color=team_color,
            linestyle=linestyle,
        )

        # Highlight the first brake point in each segment
        brake_segments = car_data[car_data["Brake"] > 0]
        segment_indices = brake_segments.index.to_series().diff().gt(1).cumsum()
        first_brake_points = brake_segments.groupby(segment_indices).first()

        ax[2].scatter(
            first_brake_points["Distance"],
            first_brake_points["Brake"],
            color=team_color,
            s=50,  # size of the marker
            zorder=5,  # put the marker on top
            edgecolor="black",
            label=f"{lap['Driver']} Brake Points",
        )

        top_speed = car_data["Speed"].max()
        top_speed_distance = car_data[car_data["Speed"] == top_speed]["Distance"].iloc[
            0
        ]
        top_speeds[driver] = top_speed

        driver_abbr = race.get_driver(driver)["Abbreviation"]
        drivers_abbr.append(driver_abbr)

        v_min = min(v_min, car_data["Speed"].min())
        v_max = max(v_max, car_data["Speed"].max())
        d_max = max(d_max, car_data["Distance"].max())

    speed_diff = abs(top_speeds[drivers[0]] - top_speeds[drivers[1]])
    laptime_diff = abs(lap_time_array[0] - lap_time_array[1])
    laptime_diff_str = f"{laptime_diff % 60:.3f}"

    for a in ax.flat:
        a.label_outer()
        a.grid(True, alpha=0.3)
        a.set_xlim([0, d_max])

    # ax.set_xlabel("Distance (m)", fontweight="bold", fontsize=14)

    ax[0].set_ylim([v_min - 40, v_max + 40])
    ax[0].set_ylabel("Speed (km/h)", fontweight="bold", fontsize=12)
    ax[0].legend(title="Drivers", loc="upper right")

    ax[1].set_ylabel("Throttle (%)", fontweight="bold", fontsize=12)

    ax[2].set_ylabel("Brakes (%)", fontweight="bold", fontsize=12)
    ax[2].set_yticklabels([str(i) for i in range(-20, 101, 20)])

    delta_time, ref_tel, _ = fastf1.utils.delta_time(compared_laps[1], compared_laps[0])

    ax[3].plot(ref_tel["Distance"], delta_time, color=team_colors[1], label=drivers[1])

    ax[3].set_ylabel("Delta Lap Time (s)", fontweight="bold", fontsize=12)
    ax[3].set_xlabel("Distance (m)", fontweight="bold", fontsize=12)
    ax[3].legend(title="Driver", loc="upper right")

    suptitle = f"{Year} {EventName} Grand Prix Driver Race Fatest Lap Telemetry"

    plt.suptitle(
        suptitle,
        fontweight="bold",
        fontsize=16,
    )

    subtitle = f"{drivers_abbr[0]} vs {drivers_abbr[1]}"
    plt.figtext(0.5, 0.937, subtitle, ha="center", fontsize=12, fontweight="bold")

    plt.tight_layout()

    filename = "../pic/" + suptitle.replace(" ", "_") + ".png"
    plt.savefig(filename)

    titles_str = (
        suptitle.replace(f"{Year} ", "")
        .replace(f"{EventName} ", "")
        .replace("Grand Prix ", "")
    )

    plt.savefig(filename)

    caption = textwrap.dedent(
        f"""\
🏎️
« {Year} {EventName} Grand Prix »

• {titles_str} {subtitle}

‣ Top Speed
\t◦ {drivers_abbr[0]}
\t{top_speeds[drivers[0]]:.1f}  (km/h)
\t◦ {drivers_abbr[1]}
\t{top_speeds[drivers[1]]:.1f}  (km/h)
‣‣ Top Speed Gap: {speed_diff:.1f}  (km/h)

‣ Lap Time
\t◦ {drivers_abbr[0]}
\t{lap_time_str_array[0]} (min:s.ms)
\t◦ {drivers_abbr[1]}
\t{lap_time_str_array[1]} (min:s.ms)
‣‣ Delta Lap Time: {laptime_diff_str} (s)  

#F1 #Formula1 #{EventName.replace(" ", "")}GP"""
    )

    return {"filename": filename, "caption": caption, "post": post}
