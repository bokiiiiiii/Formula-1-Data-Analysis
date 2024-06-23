import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib, textwrap
import fastf1
import fastf1.plotting
import fastf1.utils


# @brief annotated_race_fatest_lap: Plot the speed of a race fastest lap and add annotations to mark corners
def annotated_race_fatest_lap(
    Year: int, EventName: str, SessionName: str, race, post: bool
) -> dict:

    race.load()

    race_results = race.results
    front_row = race_results[:2]

    fig, ax = plt.subplots(figsize=(10.8, 10.8), dpi=100)

    v_min = float("inf")
    v_max = float("-inf")
    d_max = float("-inf")

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
    bg_color = ax.get_facecolor()

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
        ax.plot(
            car_data["Distance"],
            car_data["Speed"],
            color=team_color,
            label=f"{lap['Driver']}: {lap_time_str} (min:s.ms)",
            linestyle=linestyle,
        )

        v_min = min(v_min, car_data["Speed"].min())
        v_max = max(v_max, car_data["Speed"].max())
        d_max = max(d_max, car_data["Distance"].max())

        top_speed = car_data["Speed"].max()
        top_speed_distance = car_data[car_data["Speed"] == top_speed]["Distance"].iloc[
            0
        ]
        top_speeds[driver] = top_speed

        if top_speed > top_speeds[drivers[0]]:
            ypos = top_speed + 5 * i
        else:
            ypos = top_speed - 5 * i
        ax.annotate(
            f"Top Speed: {top_speed:.1f} km/h",
            xy=(top_speed_distance, ypos),
            xytext=(top_speed_distance + 100, ypos),
            fontsize=9,
            fontweight="bold",
            color=team_color,
            bbox=dict(facecolor=bg_color, alpha=0.5, edgecolor="none"),
        )

        driver_abbr = race.get_driver(driver)["Abbreviation"]
        drivers_abbr.append(driver_abbr)

    speed_diff = abs(top_speeds[drivers[0]] - top_speeds[drivers[1]])
    laptime_diff = abs(lap_time_array[0] - lap_time_array[1])
    laptime_diff_str = f"{laptime_diff % 60:.3f}"

    circuit_info = race.get_circuit_info()

    ax.vlines(
        x=circuit_info.corners["Distance"],
        ymin=v_min - 20,
        ymax=v_max + 20,
        linestyles="dotted",
        colors="grey",
    )

    for _, corner in circuit_info.corners.iterrows():
        txt = f"{corner['Number']}{corner['Letter']}"
        ax.text(
            corner["Distance"],
            v_min - 30,
            txt,
            va="center_baseline",
            ha="center",
            size=7,
        )

    ax.set_xlabel("Distance (m)", fontweight="bold", fontsize=14)
    ax.set_ylabel("Speed (km/h)", fontweight="bold", fontsize=14)
    ax.legend(title="Drivers", loc="upper right")

    ax.set_ylim([v_min - 40, v_max + 40])
    ax.set_xlim([0, d_max])

    delta_time, ref_tel, _ = fastf1.utils.delta_time(compared_laps[0], compared_laps[1])
    twin = ax.twinx()
    twin.plot(ref_tel["Distance"], delta_time, "--", color="white")
    twin.set_ylabel(r"$\mathbf{Delta\ Lap\ Time\ (s)}$", fontsize=14)
    twin.yaxis.set_major_formatter(lambda x, pos: f"+{x:.1f}" if x > 0 else f"{x:.1f}")
    plt.text(
        d_max * 1.075,
        0,
        f"←  {drivers[1]}  ahead"
        f"                                                                                                                         "
        f"{drivers[0]}  ahead  →",
        fontsize=12,
        rotation=90,
        ha="center",
        va="center",
    )

    twin_ylim = max(abs(delta_time.min()), abs(delta_time.max())) + 0.1
    twin.set_ylim([-twin_ylim, twin_ylim])

    suptitle = f"{Year} {EventName} Grand Prix Driver Race Fatest Lap"

    plt.suptitle(
        suptitle,
        fontweight="bold",
        fontsize=16,
    )

    subtitle = "with Track Corners Annotated"
    subtitle_lower = f"{drivers_abbr[0]} vs {drivers_abbr[1]}"
    pltbg_color = fig.get_facecolor()
    plt.figtext(
        0.5,
        0.937,
        subtitle,
        ha="center",
        fontsize=14,
        bbox=dict(facecolor=pltbg_color, alpha=0.5, edgecolor="none"),
    )
    axbg_color = ax.get_facecolor()
    plt.figtext(0.5, 0.912, subtitle_lower, ha="center", fontsize=12, fontweight="bold")

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

• {titles_str} {subtitle_lower}

‣ Top Speed
\t◦ {drivers_abbr[0]}
\t{top_speeds[drivers[0]]:.1f} (km/h)
\t◦ {drivers_abbr[1]}
\t{top_speeds[drivers[1]]:.1f} (km/h)
‣‣ Top Speed Gap: {speed_diff:.1f} (km/h)

‣ Lap Time
\t◦ {drivers_abbr[0]}
\t{lap_time_str_array[0]} (min:s.ms)
\t◦ {drivers_abbr[1]}
\t{lap_time_str_array[1]} (min:s.ms)
‣‣ Delta Lap Time: {laptime_diff_str} (s)  

#F1 #Formula1 #{EventName.replace(" ", "")}GP"""
    )

    return {"filename": filename, "caption": caption, "post": post}
