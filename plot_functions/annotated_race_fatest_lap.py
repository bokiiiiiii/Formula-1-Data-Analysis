import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib, textwrap
import fastf1
import fastf1.plotting
import fastf1.utils


def annotated_race_fatest_lap(
    year: int, event_name: str, session_name: str, race, post: bool
) -> dict:
    """Plot the speed of a race fastest lap and add annotations to mark corners."""

    def get_driver_team_info(driver):
        return race.laps[race.laps["Driver"] == driver]["Team"].iloc[0]

    def get_lap_time_str(lap_time):
        return f"{lap_time.total_seconds() // 60:.0f}:{lap_time.total_seconds() % 60:06.3f}"

    def plot_lap_speed(ax, car_data, lap_time_str, team_color, driver, linestyle):
        ax.plot(
            car_data["Distance"],
            car_data["Speed"],
            color=team_color,
            label=f"{driver}: {lap_time_str} (min:s.ms)",
            linestyle=linestyle,
        )

    def annotate_top_speed(
        ax, car_data, top_speed, driver, bg_color, team_color, index
    ):
        top_speed_distance = car_data[car_data["Speed"] == top_speed]["Distance"].iloc[
            0
        ]
        ypos = (
            top_speed + 5 * index
            if top_speed > driver_data["top_speeds"][drivers[0]]
            else top_speed - 5 * index
        )
        ax.annotate(
            f"Top Speed: {top_speed:.1f} km/h",
            xy=(top_speed_distance, ypos),
            xytext=(top_speed_distance + 100, ypos),
            fontsize=9,
            fontweight="bold",
            color=team_color,
            bbox=dict(facecolor=bg_color, alpha=0.5, edgecolor="none"),
        )

    def plot_corners(ax, circuit_info, v_min, v_max):
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

    def plot_delta_time(ax, compared_laps, d_max):
        delta_time, ref_tel, _ = fastf1.utils.delta_time(
            compared_laps[0], compared_laps[1]
        )
        twin = ax.twinx()
        twin.plot(ref_tel["Distance"], delta_time, "--", color="white")
        twin.set_ylabel(r"$\mathbf{Delta\ Lap\ Time\ (s)}$", fontsize=14)
        twin.yaxis.set_major_formatter(
            lambda x, pos: f"+{x:.1f}" if x > 0 else f"{x:.1f}"
        )
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

    def save_plot(fig, filename):
        plt.tight_layout()
        plt.savefig(filename)

    def generate_caption():
        titles_str = (
            suptitle.replace(f"{year} ", "")
            .replace(f"{event_name} ", "")
            .replace("Grand Prix ", "")
        )
        return textwrap.dedent(
            f"""\
🏎️
« {year} {event_name} Grand Prix »

• {titles_str} {subtitle_lower}

‣ Top Speed
\t◦ {driver_data['abbreviations'][0]}
\t{driver_data['top_speeds'][drivers[0]]:.1f} (km/h)
\t◦ {driver_data['abbreviations'][1]}
\t{driver_data['top_speeds'][drivers[1]]:.1f} (km/h)
‣‣ Top Speed Gap: {speed_diff:.1f} (km/h)

‣ Lap Time
\t◦ {driver_data['abbreviations'][0]}
\t{driver_data['lap_time_str'][0]} (min:s.ms)
\t◦ {driver_data['abbreviations'][1]}
\t{driver_data['lap_time_str'][1]} (min:s.ms)
‣‣ Delta Lap Time: {laptime_diff_str} (s)  

#F1 #Formula1 #{event_name.replace(" ", "")}GP"""
        )

    def process_driver_lap_data(driver, index, same_team, v_min, v_max, d_max):
        lap = race.laps[race.laps["Driver"] == driver].pick_fastest()
        driver_data["compared_laps"].append(lap)
        car_data = lap.get_car_data().add_distance()
        team_color = fastf1.plotting.get_team_color(lap["Team"], race)
        driver_data["team_colors"].append(team_color)

        lap_time = lap["LapTime"]
        lap_time_str = get_lap_time_str(lap_time)
        driver_data["lap_time_array"].append(lap_time.total_seconds())
        driver_data["lap_time_str"].append(lap_time_str)

        linestyle = "-" if index == 0 or not same_team else "--"
        plot_lap_speed(ax, car_data, lap_time_str, team_color, lap["Driver"], linestyle)

        v_min = min(v_min, car_data["Speed"].min())
        v_max = max(v_max, car_data["Speed"].max())
        d_max = max(d_max, car_data["Distance"].max())

        top_speed = car_data["Speed"].max()
        driver_data["top_speeds"][driver] = top_speed

        annotate_top_speed(ax, car_data, top_speed, driver, bg_color, team_color, index)
        driver_abbr = race.get_driver(driver)["Abbreviation"]
        driver_data["abbreviations"].append(driver_abbr)

        return v_min, v_max, d_max

    race.load()
    race_results = race.results
    front_row = race_results[:2]

    fig, ax = plt.subplots(figsize=(10.8, 10.8), dpi=100)
    v_min, v_max, d_max = float("inf"), float("-inf"), float("-inf")

    drivers = list(front_row["Abbreviation"])
    teams = [get_driver_team_info(driver) for driver in drivers]
    same_team = teams[0] == teams[1]

    driver_data = {
        "top_speeds": {},
        "compared_laps": [],
        "team_colors": [],
        "abbreviations": [],
        "lap_time_array": [],
        "lap_time_str": [],
    }
    bg_color = ax.get_facecolor()

    for i, driver in enumerate(drivers):
        v_min, v_max, d_max = process_driver_lap_data(
            driver, i, same_team, v_min, v_max, d_max
        )

    speed_diff = abs(
        driver_data["top_speeds"][drivers[0]] - driver_data["top_speeds"][drivers[1]]
    )
    laptime_diff = abs(
        driver_data["lap_time_array"][0] - driver_data["lap_time_array"][1]
    )
    laptime_diff_str = f"{laptime_diff % 60:.3f}"

    circuit_info = race.get_circuit_info()
    plot_corners(ax, circuit_info, v_min, v_max)

    ax.set_xlabel("Distance (m)", fontweight="bold", fontsize=14)
    ax.set_ylabel("Speed (km/h)", fontweight="bold", fontsize=14)
    ax.legend(title="Drivers", loc="upper right")

    ax.set_ylim([v_min - 40, v_max + 40])
    ax.set_xlim([0, d_max])

    plot_delta_time(ax, driver_data["compared_laps"], d_max)

    suptitle = f"{year} {event_name} Grand Prix Driver Race Fastest Lap"
    plt.suptitle(suptitle, fontweight="bold", fontsize=16)

    subtitle = "with Track Corners Annotated"
    subtitle_lower = (
        f"{driver_data['abbreviations'][0]} vs {driver_data['abbreviations'][1]}"
    )
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

    filename = f"../pic/{suptitle.replace(' ', '_')}.png"
    save_plot(fig, filename)

    caption = generate_caption()

    return {"filename": filename, "caption": caption, "post": post}
