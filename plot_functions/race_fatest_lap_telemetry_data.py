from matplotlib import pyplot as plt
import matplotlib.cm as cm
import textwrap
import fastf1
import fastf1.plotting
import fastf1.utils
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import numpy as np
import scienceplots
import matplotlib

# Global variables for caption generation consistency
suptitle_text_global = ""
subtitle_lower_text_global = ""


def race_fatest_lap_telemetry_data(
    year: int, event_name: str, session_name: str, race, post: bool
) -> dict:
    """Plot the telemetry data of the race fastest laps with styled appearance."""
    global suptitle_text_global, subtitle_lower_text_global

    fastf1.plotting.setup_mpl(
        mpl_timedelta_support=True, color_scheme=None, misc_mpl_mods=False
    )

    # Plotting constants for consistent sizing
    DPI = 125
    FIG_SIZE = (1080 / DPI, 1350 / DPI)  # Target 1080x1350 pixels

    race.load()
    race_results = race.results

    # Ensure there are at least two drivers in results
    if len(race_results) < 2:
        print(
            f"Not enough drivers in results for {year} {event_name} {session_name}. Skipping plot."
        )
        return {
            "filename": None,
            "caption": "Not enough data for comparison.",
            "post": False,
        }

    front_row_drivers_abbr = list(race_results["Abbreviation"][:2])

    def get_lap_time_str_formatted(lap_time_obj):
        return f"{lap_time_obj.total_seconds() // 60:.0f}:{lap_time_obj.total_seconds() % 60:06.3f}"  # Ensure 3 decimal places for ms

    # Modified plotting functions to accept assigned_color from scienceplots cycle
    def plot_telemetry_data_styled(
        ax_param,
        car_data_df,
        assigned_color,
        driver_abbr_label,
        linestyle_val,
        data_key_label,
    ):
        data_to_plot = car_data_df[data_key_label]
        if data_key_label == "Brake":
            data_to_plot = data_to_plot * 100
        ax_param.plot(
            car_data_df["Distance"],
            data_to_plot,
            color=assigned_color,
            linestyle=linestyle_val,
        )

    def plot_speed_data_styled(
        ax_param,
        car_data_df,
        assigned_color,
        driver_abbr_label,
        linestyle_val,
        lap_time_str_val,
    ):
        ax_param.plot(
            car_data_df["Distance"],
            car_data_df["Speed"],
            color=assigned_color,
            label=f"{driver_abbr_label}: {lap_time_str_val}",  # Lap time format is fixed in get_lap_time_str_formatted
            linestyle=linestyle_val,
        )

    def annotate_brake_points_styled(
        ax_param, car_data_df, assigned_color, driver_abbr_label
    ):
        brake_segments_df = car_data_df[
            car_data_df["Brake"] > 0.1
        ]  # Use a threshold for brake
        if brake_segments_df.empty:
            return

        segment_indices_val = (
            brake_segments_df.index.to_series().diff().fillna(1.0).gt(1).cumsum()
        )  # fillna for first element
        first_brake_points_df = brake_segments_df.groupby(segment_indices_val).first()

        ax_param.scatter(
            first_brake_points_df["Distance"],
            first_brake_points_df["Brake"] * 100,  # Plot against Brake value for y-pos
            color=assigned_color,
            s=7,
            zorder=5,
            edgecolor="None",
            linewidth=0.5,
            label=f"{driver_abbr_label}",
        )

    def save_plot_final(fig_param, suptitle_text_param, dpi_val):
        filename = (
            f"../pic/{suptitle_text_param.replace(' ', '_').replace(':', '')}.png"
        )
        fig_param.savefig(filename, dpi=dpi_val, bbox_inches=None)
        return filename

    def generate_styled_caption():  # Uses global variables for consistency
        # Construct caption title from global suptitle
        base_title_for_caption = suptitle_text_global.replace(f"{year} ", "").replace(
            f"{event_name} Grand Prix ", ""
        )

        return textwrap.dedent(
            f"""\
🏎️
« {year} {event_name} Grand Prix »

• {base_title_for_caption}
• Comparison: {subtitle_lower_text_global}

‣ Top Speed
\t◦ {driver_data_dict["abbreviations"][0]}
\t{driver_data_dict["top_speeds"][front_row_drivers_abbr[0]]:.1f} (km/h)
\t◦ {driver_data_dict["abbreviations"][1]}
\t{driver_data_dict["top_speeds"][front_row_drivers_abbr[1]]:.1f} (km/h)
‣‣ Top Speed Gap: {speed_diff_val:.1f} (km/h)

‣ Lap Time
\t◦ {driver_data_dict["abbreviations"][0]}
\t{driver_data_dict["lap_time_str_list"][0]} (min:s.ms)
\t◦ {driver_data_dict["abbreviations"][1]}
\t{driver_data_dict["lap_time_str_list"][1]} (min:s.ms)
‣‣ Delta Lap Time: {laptime_diff_str_val} (s)  

#F1 #Formula1 #{event_name.replace(" ", "")}GP"""
        )

    with plt.style.context(["science", "bright"]):
        # Attempt to override scienceplots' potential dimension-altering rcParams
        plt.rcParams["figure.dpi"] = DPI
        plt.rcParams["savefig.dpi"] = DPI
        plt.rcParams["figure.autolayout"] = False
        plt.rcParams["figure.constrained_layout.use"] = False
        plt.rcParams["savefig.bbox"] = None

        fig, axes = plt.subplots(5, figsize=FIG_SIZE, dpi=DPI, sharex=True)
        fig.patch.set_facecolor("white")
        for ax_item in axes:
            ax_item.set_facecolor("white")

        # Get scienceplots color cycle
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        color_cycle = prop_cycle.by_key()["color"]

        driver_data_dict = {
            "top_speeds": {},
            "compared_laps_objects": [],
            "driver_line_colors": [],
            "abbreviations": [],
            "lap_time_seconds_list": [],
            "lap_time_str_list": [],
        }

        v_min_speed, v_max_speed, d_max_dist = (
            float("inf"),
            float("-inf"),
            float("-inf"),
        )

        processed_driver_count = 0
        for i, driver_abbr_val in enumerate(front_row_drivers_abbr):
            try:
                lap_obj = race.laps.pick_driver(driver_abbr_val).pick_fastest()
                if lap_obj is None or pd.isna(lap_obj.LapTime):
                    print(
                        f"No valid fastest lap for {driver_abbr_val}. Skipping this driver."
                    )
                    continue
            except Exception as e:
                print(
                    f"Error picking fastest lap for {driver_abbr_val}: {e}. Skipping this driver."
                )
                continue

            driver_data_dict["compared_laps_objects"].append(lap_obj)
            car_data_df = lap_obj.get_car_data().add_distance()

            # If car_data is empty, skip this driver
            if car_data_df.empty:
                print(
                    f"No car data for {driver_abbr_val}'s fastest lap. Skipping this driver."
                )
                # Remove already added lap_obj if car_data is empty
                if (
                    driver_data_dict["compared_laps_objects"]
                    and driver_data_dict["compared_laps_objects"][-1] is lap_obj
                ):
                    driver_data_dict["compared_laps_objects"].pop()
                continue

            assigned_plot_color = color_cycle[i % len(color_cycle)]
            driver_data_dict["driver_line_colors"].append(assigned_plot_color)

            lap_time_obj = lap_obj["LapTime"]
            lap_time_str_val = get_lap_time_str_formatted(lap_time_obj)
            driver_data_dict["lap_time_seconds_list"].append(
                lap_time_obj.total_seconds()
            )
            driver_data_dict["lap_time_str_list"].append(lap_time_str_val)

            linestyle_val = "-"

            plot_speed_data_styled(
                axes[0],
                car_data_df,
                assigned_plot_color,
                driver_abbr_val,
                linestyle_val,
                lap_time_str_val,
            )
            plot_telemetry_data_styled(
                axes[1],
                car_data_df,
                assigned_plot_color,
                driver_abbr_val,
                linestyle_val,
                "Throttle",
            )
            plot_telemetry_data_styled(
                axes[2],
                car_data_df,
                assigned_plot_color,
                driver_abbr_val,
                linestyle_val,
                "Brake",
            )

            # Acceleration
            car_data_df.loc[:, "Acceleration"] = gaussian_filter1d(
                car_data_df["Speed"].diff().fillna(0)
                / car_data_df["Time"].dt.total_seconds().diff().fillna(method="bfill"),
                sigma=2,
            )
            axes[3].plot(
                car_data_df["Distance"],
                car_data_df["Acceleration"],
                color=assigned_plot_color,
                linestyle=linestyle_val,
                label=f"{driver_abbr_val}",
            )

            annotate_brake_points_styled(
                axes[2], car_data_df, assigned_plot_color, driver_abbr_val
            )

            driver_data_dict["top_speeds"][driver_abbr_val] = car_data_df["Speed"].max()
            driver_data_dict["abbreviations"].append(driver_abbr_val)

            v_min_speed = min(v_min_speed, car_data_df["Speed"].min())
            v_max_speed = max(v_max_speed, car_data_df["Speed"].max())
            d_max_dist = max(d_max_dist, car_data_df["Distance"].max())
            processed_driver_count += 1

        if processed_driver_count < 2:
            print(
                f"Not enough driver data processed for comparison in {year} {event_name}. Skipping plot."
            )
            plt.close(fig)
            return {
                "filename": None,
                "caption": "Failed to process data for two drivers.",
                "post": False,
            }

        speed_diff_val = abs(
            driver_data_dict["top_speeds"][front_row_drivers_abbr[0]]
            - driver_data_dict["top_speeds"][front_row_drivers_abbr[1]]
        )
        laptime_diff_val = abs(
            driver_data_dict["lap_time_seconds_list"][0]
            - driver_data_dict["lap_time_seconds_list"][1]
        )
        laptime_diff_str_val = f"{laptime_diff_val % 60:.3f}"

        # Speed difference plot (ax0_right)
        car_data1_df = (
            driver_data_dict["compared_laps_objects"][0].get_car_data().add_distance()
        )
        car_data2_df = (
            driver_data_dict["compared_laps_objects"][1].get_car_data().add_distance()
        )

        interp_speed2_func = interp1d(
            car_data2_df["Distance"],
            car_data2_df["Speed"],
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        aligned_speed2_vals = interp_speed2_func(car_data1_df["Distance"])
        speed_difference_vals = car_data1_df["Speed"] - aligned_speed2_vals
        max_abs_diff = (
            np.nanmax(np.abs(speed_difference_vals))
            if not np.all(np.isnan(speed_difference_vals))
            else 10
        )

        ax0_twin = axes[0].twinx()
        speed_diff_line_color = "green"
        ax0_twin.plot(
            car_data1_df["Distance"],
            speed_difference_vals,
            color=speed_diff_line_color,
            linestyle=":",
            label="Diff.",
        )
        # Fill between for speed difference
        ax0_twin.fill_between(
            car_data1_df["Distance"],
            speed_difference_vals,
            0,
            where=speed_difference_vals > 0,
            color=driver_data_dict["driver_line_colors"][0],
            alpha=0.2,
            interpolate=True,
        )
        ax0_twin.fill_between(
            car_data1_df["Distance"],
            speed_difference_vals,
            0,
            where=speed_difference_vals < 0,
            color=driver_data_dict["driver_line_colors"][1],
            alpha=0.2,
            interpolate=True,
        )
        ax0_twin.set_ylabel(
            r"$\Delta$ Speed (km/h)", fontsize=12, color=speed_diff_line_color
        )
        ax0_twin.tick_params(
            axis="y", labelcolor=speed_diff_line_color, colors=speed_diff_line_color
        )
        ax0_twin.set_ylim([-max_abs_diff * 1.1, max_abs_diff * 1.1])
        ax0_twin.grid(False)

        # Consolidate legends for axes[0]
        lines_main, labels_main = axes[0].get_legend_handles_labels()
        lines_twin, labels_twin = ax0_twin.get_legend_handles_labels()
        axes[0].legend(
            lines_main + lines_twin,
            labels_main + labels_twin,
            loc="lower center",  # Position legend anchor point at the bottom center of the legend box
            bbox_to_anchor=(
                0.5,
                1.02,
            ),  # Place legend box: x=0.5 (center), y=1.02 (slightly above the axes)
            ncol=len(labels_main + labels_twin),  # Arrange legend items horizontally
            fontsize=9,
            title_fontsize=10,
            facecolor="white",
            framealpha=0.7,
        ).get_title().set_color("black")
        plt.setp(axes[0].get_legend().get_texts(), color="black")

        # Delta time plot (axes[4]) - similar to annotated_qualifying_flying_lap
        delta_time_vals, ref_tel_df, _ = fastf1.utils.delta_time(
            driver_data_dict["compared_laps_objects"][0],
            driver_data_dict["compared_laps_objects"][1],
        )
        delta_line_color_val = cm.get_cmap("PiYG")(
            0.9 if delta_time_vals.mean() > 0 else 0.1
        )  # Color based on who is faster on avg
        axes[4].plot(
            ref_tel_df["Distance"], delta_time_vals, "--", color=delta_line_color_val
        )
        axes[4].fill_between(
            ref_tel_df["Distance"],
            delta_time_vals,
            0,
            where=delta_time_vals > 0,
            color=driver_data_dict["driver_line_colors"][0],
            alpha=0.2,
            interpolate=True,
        )
        axes[4].fill_between(
            ref_tel_df["Distance"],
            delta_time_vals,
            0,
            where=delta_time_vals < 0,
            color=driver_data_dict["driver_line_colors"][1],
            alpha=0.2,
            interpolate=True,
        )

        current_delta_ylim = (
            np.nanmax(np.abs(delta_time_vals)) + 0.1
            if not np.all(np.isnan(delta_time_vals))
            else 1.0
        )
        axes[4].set_ylim([-current_delta_ylim, current_delta_ylim])
        axes[4].yaxis.set_major_formatter(
            lambda x, pos: f"+{x:.1f}" if x > 0 else f"{x:.1f}"
        )

        # Styling for all axes
        for i, ax_item in enumerate(axes):
            ax_item.label_outer()  # Hide interior labels
            ax_item.grid(True, linestyle="--", alpha=0.7, color="lightgrey")
            ax_item.set_xlim([0, d_max_dist])
            ax_item.tick_params(axis="x", colors="black")
            ax_item.tick_params(axis="y", colors="black")
            if ax_item.get_legend() and i != 0:  # axes[0] legend handled above
                plt.setp(ax_item.get_legend().get_texts(), color="black")
                if ax_item.get_legend().get_title():
                    ax_item.get_legend().get_title().set_color("black")

        axes[0].set_ylim([v_min_speed - 20, v_max_speed + 20])
        axes[0].set_ylabel(r"Speed (km/h)", fontsize=12, color="black")

        axes[1].set_ylabel(r"Throttle (\%)", fontsize=12, color="black")
        axes[1].set_ylim([-5, 105])

        axes[2].set_ylabel(r"Brake (\%)", fontsize=12, color="black")
        axes[2].set_ylim([-5, 105])
        axes[2].set_yticks([0, 25, 50, 75, 100])

        axes[3].set_ylabel(r"Acceleration (m/s$^2$)", fontsize=12, color="black")

        axes[4].set_ylabel(
            r"$\Delta$ Lap Time (s)", fontsize=12, color=delta_line_color_val
        )
        axes[4].tick_params(
            axis="y", labelcolor=delta_line_color_val, colors=delta_line_color_val
        )
        axes[4].set_xlabel("Distance (m)", fontsize=12, color="black")

        suptitle_text_global = (
            f"{year} {event_name} Grand Prix: Fastest Lap Telemetry Comparison"
        )
        plt.suptitle(suptitle_text_global, fontsize=18, color="black")
        subtitle_lower_text_global = f"{driver_data_dict['abbreviations'][0]} vs {driver_data_dict['abbreviations'][1]}"
        plt.figtext(
            0.5,
            0.935,
            subtitle_lower_text_global,
            ha="center",
            fontsize=13,
            color="black",
        )

        # Define the suptitle string that should be used for generating the filename,
        # matching the original pattern.
        suptitle_for_filename = (
            f"{year} {event_name} Grand Prix Driver Race Fastest Lap Telemetry"
        )

        filename = save_plot_final(fig, suptitle_for_filename, DPI)
        plt.close(fig)

    # generate_styled_caption uses suptitle_text_global for display consistency
    caption = generate_styled_caption()
    return {"filename": filename, "caption": caption, "post": post}
