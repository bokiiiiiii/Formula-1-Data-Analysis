import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib, textwrap
import fastf1
import fastf1.plotting
import fastf1.utils


def race_fatest_lap_telemetry_data(year: int, event_name: str, session_name: str, race, post: bool) -> dict:
    """Plot the telemetry data of the race fastest laps."""

    def get_driver_team_info(driver):
        return race.laps[race.laps["Driver"] == driver]["Team"].iloc[0]

    def get_lap_time_str(lap_time):
        return f"{lap_time.total_seconds() // 60:.0f}:{lap_time.total_seconds() % 60:.3f}"

    def plot_telemetry_data(ax, car_data, team_color, driver, linestyle, data_label):
        ax.plot(car_data["Distance"], car_data[data_label], color=team_color, linestyle=linestyle)

    def plot_speed_data(ax, car_data, team_color, driver, linestyle, lap_time_str):
        ax.plot(
            car_data["Distance"],
            car_data["Speed"],
            color=team_color,
            label=f"{driver}: {lap_time_str} (min:s.ms)",
            linestyle=linestyle,
        )

    def annotate_brake_points(ax, car_data, team_color, driver):
        brake_segments = car_data[car_data["Brake"] > 0]
        segment_indices = brake_segments.index.to_series().diff().gt(1).cumsum()
        first_brake_points = brake_segments.groupby(segment_indices).first()

        ax.scatter(
            first_brake_points["Distance"],
            first_brake_points["Brake"],
            color=team_color,
            s=50,
            zorder=5,
            edgecolor="black",
            label=f"{driver} Brake Points",
        )

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

• {titles_str} {subtitle}

‣ Top Speed
\t◦ {driver_data["abbreviations"][0]}
\t{driver_data["top_speeds"][drivers[0]]:.1f} (km/h)
\t◦ {driver_data["abbreviations"][1]}
\t{driver_data["top_speeds"][drivers[1]]:.1f} (km/h)
‣‣ Top Speed Gap: {speed_diff:.1f} (km/h)

‣ Lap Time
\t◦ {driver_data["abbreviations"][0]}
\t{driver_data["lap_time_array"][0]} (min:s.ms)
\t◦ {driver_data["abbreviations"][1]}
\t{driver_data["lap_time_array"][1]} (min:s.ms)
‣‣ Delta Lap Time: {laptime_diff_str} (s)  

#F1 #Formula1 #{event_name.replace(" ", "")}GP"""
        )

    race.load()
    race_results = race.results
    front_row = race_results[:2]

    fig, ax = plt.subplots(4, figsize=(10.8, 10.8), dpi=100)

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

    v_min = float("inf")
    v_max = float("-inf")
    d_max = float("-inf")

    for i, driver in enumerate(drivers):
        lap = race.laps[race.laps["Driver"] == driver].pick_fastest()
        driver_data["compared_laps"].append(lap)
        car_data = lap.get_car_data().add_distance()
        team_color = fastf1.plotting.team_color(lap["Team"])
        driver_data["team_colors"].append(team_color)

        lap_time = lap["LapTime"]
        lap_time_str = get_lap_time_str(lap_time)
        driver_data["lap_time_array"].append(lap_time.total_seconds())
        driver_data["lap_time_str"].append(lap_time_str)

        linestyle = "-" if i == 0 or not same_team else "--"
        plot_speed_data(ax[0], car_data, team_color, driver, linestyle, lap_time_str)
        plot_telemetry_data(ax[1], car_data, team_color, driver, linestyle, "Throttle")
        plot_telemetry_data(ax[2], car_data, team_color, driver, linestyle, "Brake")

        annotate_brake_points(ax[2], car_data, team_color, driver)

        top_speed = car_data["Speed"].max()
        driver_data["top_speeds"][driver] = top_speed

        driver_abbr = race.get_driver(driver)["Abbreviation"]
        driver_data["abbreviations"].append(driver_abbr)

        v_min = min(v_min, car_data["Speed"].min())
        v_max = max(v_max, car_data["Speed"].max())
        d_max = max(d_max, car_data["Distance"].max())

    speed_diff = abs(driver_data["top_speeds"][drivers[0]] - driver_data["top_speeds"][drivers[1]])
    laptime_diff = abs(driver_data["lap_time_array"][0] - driver_data["lap_time_array"][1])
    laptime_diff_str = f"{laptime_diff % 60:.3f}"

    for a in ax.flat:
        a.label_outer()
        a.grid(True, alpha=0.3)
        a.set_xlim([0, d_max])

    ax[0].set_ylim([v_min - 40, v_max + 40])
    ax[0].set_ylabel("Speed (km/h)", fontweight="bold", fontsize=12)
    ax[0].legend(title="Drivers", loc="upper right")

    ax[1].set_ylabel("Throttle (%)", fontweight="bold", fontsize=12)

    ax[2].set_ylabel("Brakes (%)", fontweight="bold", fontsize=12)
    ax[2].set_yticklabels([str(i) for i in range(-20, 101, 20)])

    delta_time, ref_tel, _ = fastf1.utils.delta_time(driver_data["compared_laps"][1], driver_data["compared_laps"][0])
    ax[3].plot(ref_tel["Distance"], delta_time, color=driver_data["team_colors"][1], label=drivers[1])

    ax[3].set_ylabel("Delta Lap Time (s)", fontweight="bold", fontsize=12)
    ax[3].set_xlabel("Distance (m)", fontweight="bold", fontsize=12)
    ax[3].legend(title="Driver", loc="upper right")

    suptitle = f"{year} {event_name} Grand Prix Driver Race Fastest Lap Telemetry"
    plt.suptitle(suptitle, fontweight="bold", fontsize=16)

    subtitle = f"{driver_data['abbreviations'][0]} vs {driver_data['abbreviations'][1]}"
    plt.figtext(0.5, 0.937, subtitle, ha="center", fontsize=12, fontweight="bold")

    filename = f"../pic/{suptitle.replace(' ', '_')}.png"
    save_plot(fig, filename)

    caption = generate_caption()
    return {"filename": filename, "caption": caption, "post": post}
