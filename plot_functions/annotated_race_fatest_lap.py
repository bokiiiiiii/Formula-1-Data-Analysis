from matplotlib import pyplot as plt
import matplotlib.cm as cm
import textwrap
import scienceplots
import matplotlib
import fastf1
import fastf1.plotting
import fastf1.utils


def annotated_race_fatest_lap(
    year: int, event_name: str, session_name: str, race, post: bool
) -> dict:
    """Plot the speed of a race fastest lap and add annotations to mark corners."""

    def get_driver_team_info(driver):
        # This function seems to be defined but not used in the original annotated_race_fatest_lap.py
        # Keeping it in case it's intended for future use or was overlooked.
        return race.laps[race.laps["Driver"] == driver]["Team"].iloc[0]

    def get_lap_time_str(lap_time):
        return f"{lap_time.total_seconds() // 60:.0f}:{lap_time.total_seconds() % 60:06.3f}"

    def plot_lap_speed(
        ax,
        car_data,
        lap_time_str,
        team_color,
        driver_abbr,
        linestyle,
    ):
        plot_kwargs = {
            "label": f"{driver_abbr}: {lap_time_str}",
            "linestyle": linestyle,
        }
        # The explicit color setting is removed to allow scienceplots 'bright' style to define colors.
        # The team_color parameter is still passed but not used for explicit color setting here.
        # If specific team colors are desired over scienceplots styles,
        # the logic to use 'team_color' in 'plot_kwargs' would need to be re-enabled.

        line = ax.plot(car_data["Distance"], car_data["Speed"], **plot_kwargs)
        return line[0]

    def annotate_top_speed(
        ax,
        car_data,
        top_speed,
        driver_abbr,
        bg_color,
        line_color_for_annotation,
        index,
        first_driver_abbr,
    ):
        top_speed_distance = car_data[car_data["Speed"] == top_speed]["Distance"].iloc[
            0
        ]
        ypos = top_speed
        if index == 1:
            offset_val = 8
            # Need access to the first driver's top speed for comparison
            # This assumes driver_data['top_speeds'] is accessible here or passed
            offset_direction = (
                1 if top_speed > driver_data["top_speeds"][first_driver_abbr] else -1
            )
            ypos += offset_direction * offset_val
        ax.annotate(
            f"Top Speed: {top_speed:.1f} km/h",
            xy=(top_speed_distance, ypos),
            xytext=(top_speed_distance + 100, ypos),
            fontsize=12,
            color=line_color_for_annotation,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

    def plot_corners(ax, circuit_info, v_min, v_max):
        ax.vlines(
            x=circuit_info.corners["Distance"],
            ymin=v_min - 20,
            ymax=v_max + 20,
            linestyles="dotted",
            colors="grey",
        )
        for i, (
            original_index,
            corner_data,
        ) in enumerate(circuit_info.corners.iterrows()):
            txt = f"{corner_data['Number']}{corner_data['Letter']}"
            y_base = v_min - 25
            y_offset = 0 if i % 2 == 0 else -7
            ax.text(
                corner_data["Distance"],
                y_base + y_offset,
                txt,
                va="center_baseline",
                ha="center",
                size=7,
                color="black",
            )

    def plot_delta_time(
        ax, compared_laps, d_max, driver_line_colors_param, drivers_abbr_list_param
    ):
        delta_time, ref_tel, _ = fastf1.utils.delta_time(
            compared_laps[0], compared_laps[1]
        )
        twin = ax.twinx()
        cmap = cm.get_cmap("tab20c")
        delta_line_color = cmap(8)

        twin.plot(
            ref_tel["Distance"],
            delta_time,
            "--",
            color=delta_line_color,
        )

        twin.fill_between(
            ref_tel["Distance"],
            delta_time,
            0,
            where=delta_time > 0,
            color=driver_line_colors_param[0],
            alpha=0.1,
            interpolate=True,
        )
        twin.fill_between(
            ref_tel["Distance"],
            delta_time,
            0,
            where=delta_time < 0,
            color=driver_line_colors_param[1],
            alpha=0.1,
            interpolate=True,
        )

        twin.set_ylabel(
            r"$\Delta \mathrm{Lap\ Time\ (s)}$",
            fontsize=14,
            color=delta_line_color,
        )
        twin.tick_params(axis="y", colors=delta_line_color)
        twin.yaxis.set_major_formatter(
            lambda x, pos: f"+{x:.1f}" if x > 0 else f"{x:.1f}"
        )

        current_twin_ylim_val = max(abs(delta_time.min()), abs(delta_time.max())) + 0.1
        twin.set_ylim([-current_twin_ylim_val, current_twin_ylim_val])

        final_twin_ylim = twin.get_ylim()
        x_text_position = d_max * 1.08

        y_offset_val = 0.05 * (final_twin_ylim[1] - final_twin_ylim[0])
        plt.text(
            x_text_position,
            final_twin_ylim[0] + y_offset_val,
            f"← {drivers_abbr_list_param[1]} ahead",
            fontsize=13,
            rotation=90,
            ha="center",
            va="center",
            color=driver_line_colors_param[1],
        )

        plt.text(
            x_text_position,
            final_twin_ylim[1] - y_offset_val,
            f"{drivers_abbr_list_param[0]} ahead →",
            fontsize=13,
            rotation=90,
            ha="center",
            va="center",
            color=driver_line_colors_param[0],
        )

    def save_plot(fig, filename):
        fig.savefig(filename, dpi=DPI, bbox_inches=None)

    def generate_caption():
        titles_str = (
            suptitle_text.replace(f"{year} ", "")
            .replace(f"{event_name} ", "")
            .replace("Grand Prix ", "")
        )
        return textwrap.dedent(
            f"""\
🏎️
« {year} {event_name} Grand Prix »

• {titles_str} {subtitle_lower_text}

‣ Top Speed
\t◦ {driver_data['abbreviations'][0]}
\t{driver_data['top_speeds'][drivers_abbr_list[0]]:.1f} (km/h)
\t◦ {driver_data['abbreviations'][1]}
\t{driver_data['top_speeds'][drivers_abbr_list[1]]:.1f} (km/h)
‣‣ Top Speed Gap: {speed_diff:.1f} (km/h)

‣ Lap Time
\t◦ {driver_data['abbreviations'][0]}
\t{driver_data['lap_time_str'][0]} (min:s.ms)
\t◦ {driver_data['abbreviations'][1]}
\t{driver_data['lap_time_str'][1]} (min:s.ms)
‣‣ Delta Lap Time: {laptime_diff_str} (s)  

#F1 #Formula1 #{event_name.replace(" ", "")}GP"""
        )

    def process_driver_lap_data(
        driver_abbr_param,
        index,
        same_team,
        v_min,
        v_max,
        d_max,
    ):
        # In race context, we pick fastest lap directly, not from Q3 sessions
        lap = race.laps[race.laps["Driver"] == driver_abbr_param].pick_fastest()
        driver_data["compared_laps"].append(lap)
        car_data = lap.get_car_data().add_distance()

        team_color_for_plot = fastf1.plotting.get_team_color(lap["Team"], race)

        lap_time = lap["LapTime"]
        lap_time_str = get_lap_time_str(lap_time)
        driver_data["lap_time_array"].append(lap_time.total_seconds())
        driver_data["lap_time_str"].append(lap_time_str)

        linestyle = "-" if index == 0 or not same_team else "--"

        line_obj = plot_lap_speed(
            ax,
            car_data,
            lap_time_str,
            team_color_for_plot,
            driver_abbr_param,
            linestyle,
        )
        actual_line_color = line_obj.get_color()
        driver_data["driver_line_colors"].append(actual_line_color)

        v_min = min(v_min, car_data["Speed"].min())
        v_max = max(v_max, car_data["Speed"].max())
        d_max = max(d_max, car_data["Distance"].max())

        top_speed = car_data["Speed"].max()
        driver_data["top_speeds"][driver_abbr_param] = top_speed

        annotate_top_speed(
            ax,
            car_data,
            top_speed,
            driver_abbr_param,
            bg_color,
            actual_line_color,
            index,
            drivers_abbr_list[0],
        )
        driver_data["abbreviations"].append(driver_abbr_param)

        return v_min, v_max, d_max

    fastf1.plotting.setup_mpl(
        mpl_timedelta_support=True, color_scheme=None, misc_mpl_mods=False
    )

    DPI = 125
    FIG_SIZE = (1080 / DPI, 1350 / DPI)

    race.load()
    race_results = race.results
    # Consider taking top 2 based on fastest lap, not just first two in results if they DNF'd early
    # For now, sticking to original logic of top 2 from results.
    front_row_drivers = race_results[:2]
    drivers_abbr_list = list(front_row_drivers["Abbreviation"])

    if len(drivers_abbr_list) < 2:
        print(
            f"Not enough drivers for comparison in {year} {event_name} {session_name}. Skipping plot."
        )
        return {"filename": None, "caption": "Not enough data for plot.", "post": False}

    with plt.style.context(["science", "bright"]):
        plt.rcParams["figure.dpi"] = DPI
        plt.rcParams["savefig.dpi"] = DPI
        plt.rcParams["figure.autolayout"] = False
        plt.rcParams["figure.constrained_layout.use"] = False
        plt.rcParams["savefig.bbox"] = None

        fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        v_min, v_max, d_max = float("inf"), float("-inf"), float("-inf")

        # Get team info for same_team logic
        # We need the team name string for comparison.
        teams_names = [
            race.laps[race.laps["Driver"] == driver_abbr]["Team"].iloc[0]
            for driver_abbr in drivers_abbr_list[:2]
        ]
        same_team = teams_names[0] == teams_names[1] if len(teams_names) == 2 else False

        driver_data = {
            "top_speeds": {},
            "compared_laps": [],
            "driver_line_colors": [],
            "abbreviations": [],
            "lap_time_array": [],
            "lap_time_str": [],
        }
        bg_color = ax.get_facecolor()

        for i, driver_abbr_item in enumerate(drivers_abbr_list[:2]):
            v_min, v_max, d_max = process_driver_lap_data(
                driver_abbr_item,
                i,
                same_team,
                v_min,
                v_max,
                d_max,
            )

        if len(driver_data["abbreviations"]) < 2:
            print(
                f"Could not process data for two drivers in {year} {event_name} {session_name}. Skipping plot."
            )
            plt.close(fig)
            return {
                "filename": None,
                "caption": "Failed to process data for two drivers.",
                "post": False,
            }

        speed_diff = abs(
            driver_data["top_speeds"][drivers_abbr_list[0]]
            - driver_data["top_speeds"][drivers_abbr_list[1]]
        )
        laptime_diff = abs(
            driver_data["lap_time_array"][0] - driver_data["lap_time_array"][1]
        )
        laptime_diff_str = f"{laptime_diff % 60:.3f}"

        circuit_info = race.get_circuit_info()
        plot_corners(ax, circuit_info, v_min, v_max)

        ax.set_xlabel("Distance (m)", fontsize=14, color="black")
        ax.set_ylabel("Speed (km/h)", fontsize=14, color="black")

        legend = ax.legend(
            title="Drivers and Lap Times (min:s.ms)",
            loc="upper right",
            fontsize=12,
            title_fontsize=13,
            frameon=True,
            facecolor="white",
            edgecolor="white",
            framealpha=0.7,
        )
        if legend:
            plt.setp(legend.get_texts(), color="black")
            if legend.get_title():
                legend.get_title().set_color("black")

        ax.tick_params(axis="x", colors="black")
        ax.tick_params(axis="y", colors="black")

        ax.set_ylim([v_min - 40, v_max + 40])
        ax.set_xlim([0, d_max])

        plot_delta_time(
            ax,
            driver_data["compared_laps"],
            d_max,
            driver_data["driver_line_colors"],
            drivers_abbr_list,
        )

        # Define titles for plot display and filename generation
        plot_display_suptitle = (
            f"{year} {event_name} Grand Prix Driver Fastest Laps Comparison"
        )
        plt.suptitle(plot_display_suptitle, fontsize=18, color="black")

        suptitle_for_filename = (
            f"{year} {event_name} Grand Prix Driver Race Fastest Lap"
        )
        # suptitle_text is used by generate_caption
        suptitle_text = plot_display_suptitle

        subtitle_text = "with Track Corners Annotated"
        subtitle_lower_text = (
            f"{driver_data['abbreviations'][0]} vs {driver_data['abbreviations'][1]}"
        )

        plt.figtext(
            0.5,
            0.94,
            subtitle_text,
            ha="center",
            fontsize=15,
            color="black",
        )
        plt.figtext(
            0.5,
            0.915,
            subtitle_lower_text,
            ha="center",
            fontsize=13,
            color="black",
        )

        filename = f"../pic/{suptitle_for_filename.replace(' ', '_')}.png"
        save_plot(fig, filename)
        plt.close(fig)

    caption = generate_caption()

    return {"filename": filename, "caption": caption, "post": post}
