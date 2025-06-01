import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection
from sklearn.linear_model import LinearRegression
import fastf1
import fastf1.plotting
import pandas as pd
import textwrap
import numpy as np

# Parameters
QUICKLAP_THRESHOLD = 1.07
MARKERS = [".", "*"]
LINES = ["-", "--"]

# Global variables for caption generation consistency
suptitle_text_global = ""
subtitle_lower_text_global = ""


def load_race_data(race):
    """Load race data."""
    try:
        race.load()
    except Exception as e:
        raise RuntimeError(f"Error loading race data: {e}")


def get_podium_finishers_abbr(race):
    """Get the top 2 finishers' abbreviations."""
    results = race.results
    if results is None or results.empty or len(results) < 2:
        print("Warning: Not enough results to pick top 2 finishers.")
        return list(results["Abbreviation"][: len(results)])
    return list(results["Abbreviation"][:2])


def get_driver_laps_cleaned(race, driver_abbr):
    """Get the laps for a driver, with standardized Compound names."""
    try:
        driver_number = race.get_driver(driver_abbr)["DriverNumber"]
    except Exception as e:
        print(f"Could not get driver number for {driver_abbr}: {e}")
        return pd.DataFrame()

    laps_orig = race.laps.pick_drivers(driver_number).pick_quicklaps(QUICKLAP_THRESHOLD)
    laps = laps_orig.copy()

    if laps.empty:
        return pd.DataFrame()

    if "LapTime(s)" not in laps.columns:
        laps.loc[:, "LapTime(s)"] = laps["LapTime"].dt.total_seconds()

    if "Compound" in laps.columns:
        laps.loc[:, "Compound"] = laps["Compound"].fillna("UNKNOWN")
        laps.loc[:, "Compound"] = (
            laps["Compound"]
            .astype(str)
            .replace(
                {
                    "UNKOWN": "UNKNOWN",
                    "nan": "UNKNOWN",
                    "None": "UNKNOWN",
                    "": "UNKNOWN",
                }
            )
        )
        valid_compounds = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN"]
        laps.loc[~laps["Compound"].isin(valid_compounds), "Compound"] = "UNKNOWN"
    else:
        laps.loc[:, "Compound"] = "UNKNOWN"

    if "StintLapNumber" not in laps.columns:
        laps["StintLapNumber"] = (
            laps.groupby("Stint")["LapNumber"]
            .rank(method="first", ascending=True)
            .astype(int)
        )
    return laps


def get_stints_info_for_driver(race, driver_abbr):
    """Get the stints data for a specific driver."""
    try:
        driver_number = race.get_driver(driver_abbr)["DriverNumber"]
    except Exception:
        return pd.DataFrame()

    stints_df = race.laps.pick_drivers(driver_number)[
        ["Stint", "LapNumber", "Compound"]
    ]
    stint_summary = (
        stints_df.groupby("Stint")
        .agg(
            StintEndLap=("LapNumber", "max"),
            Compound=("Compound", "first"),
        )
        .reset_index()
    )
    return stint_summary


def plot_driver_laps_styled(
    ax,
    driver_laps_df,
    driver_idx,
    driver_abbr_label,
    assigned_driver_color,
):
    """Plot the driver laps with scienceplots style."""
    sns.scatterplot(
        data=driver_laps_df,
        x="LapNumber",
        y="LapTime(s)",
        ax=ax,
        hue="Compound",
        palette={
            **fastf1.plotting.COMPOUND_COLORS,
            "UNKNOWN": "black",
        },
        hue_order=[
            "SOFT",
            "MEDIUM",
            "HARD",
            "INTERMEDIATE",
            "WET",
            "UNKNOWN",
        ],
        marker=MARKERS[driver_idx % len(MARKERS)],
        s=60,
        linewidth=0.25,
        edgecolor=assigned_driver_color,
    )


def plot_pit_lap_lines_styled(ax, pit_lap_numbers, line_color, y_pos_for_text):
    """Plot vertical lines for pit laps and "Pit" text with styled color."""
    for pit_lap in pit_lap_numbers:
        ax.axvline(
            x=pit_lap, color=line_color, linestyle="--", linewidth=1.2, alpha=0.7
        )
        if y_pos_for_text is not None:
            ax.text(
                pit_lap,
                y_pos_for_text,
                "Pit",
                rotation=90,
                color=line_color,
                fontsize=9,
                va="top",
                ha="center",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=0.1),
            )


def plot_stint_trendlines_styled(
    ax,
    driver_laps_df,
    assigned_color,
    driver_idx,
    line_styles,
    global_y_anchor_top,
    global_plot_range_y,
):
    """Plot stint trendlines with styled color and text."""
    slope_str_list = []
    for stint_num, stint_data in driver_laps_df.groupby("Stint"):
        if len(stint_data) < 2:
            continue

        compound_for_stint = stint_data["Compound"].iloc[0]
        if compound_for_stint == "HARD":
            reg_color = "grey"
        else:
            reg_color = fastf1.plotting.COMPOUND_COLORS.get(
                compound_for_stint, assigned_color
            )

        collections_before = list(ax.collections)

        sns.regplot(
            x="LapNumber",
            y="LapTime(s)",
            data=stint_data,
            ax=ax,
            scatter=False,
            color=reg_color,
            line_kws={
                "linestyle": line_styles[driver_idx % len(line_styles)],
                "linewidth": 0.8,
                "alpha": 0.7,
            },
        )

        newly_added_collections = [
            c for c in ax.collections if c not in collections_before
        ]
        for collection in newly_added_collections:
            if isinstance(collection, PolyCollection):
                collection.set_alpha(0.1)

        X_reg = stint_data["LapNumber"].values.reshape(-1, 1)
        Y_reg = stint_data["LapTime(s)"].values.reshape(-1, 1)
        try:
            reg_model = LinearRegression().fit(X_reg, Y_reg)
            slope_val = reg_model.coef_[0][0]
        except ValueError:
            slope_val = np.nan

        mid_lap = stint_data["LapNumber"].mean()
        y_pos_for_slope_text = (
            global_y_anchor_top - (0.018 + driver_idx * 0.020) * global_plot_range_y
        )

        slope_text_value = f"{slope_val:+.3f} s/lap"
        slope_str_list.append(f"{slope_val:+.3f}")

        tire_color = fastf1.plotting.COMPOUND_COLORS.get(compound_for_stint, "grey")
        driver_marker_shape = MARKERS[driver_idx % len(MARKERS)]
        marker_center_x = mid_lap - 2.0
        text_start_x = mid_lap - 1.3

        ax.plot(
            marker_center_x,
            y_pos_for_slope_text + 0.008,
            marker=driver_marker_shape,
            markersize=5,
            color=tire_color,
            markeredgecolor=assigned_color,
            markeredgewidth=0.5,
            linestyle="None",
            transform=ax.transData,
            clip_on=False,
        )

        ax.text(
            text_start_x,
            y_pos_for_slope_text,
            slope_text_value,
            color=assigned_color,
            fontsize=9,
            ha="left",
            va="center",
            bbox=dict(
                facecolor="white",
                alpha=0.4,
                edgecolor="none",
                pad=0.1,
            ),
        )
    return slope_str_list


def set_plot_labels_styled(ax, race_obj, min_time_val, max_time_val):
    """Set plot labels and axis limits with new style."""
    ax.set_xlabel("Lap Number", fontsize=14, color="black")
    ax.set_ylabel("Lap Time (s)", fontsize=14, color="black")
    ax.tick_params(axis="x", colors="black")
    ax.tick_params(axis="y", colors="black")

    max_lap_num = race_obj.laps["LapNumber"].max()
    ax.set_xlim(0.5, max_lap_num + 0.5 if not pd.isna(max_lap_num) else 50)

    if (
        min_time_val is not None
        and max_time_val is not None
        and not (pd.isna(min_time_val) or pd.isna(max_time_val))
    ):
        padding = (max_time_val - min_time_val) * 0.05
        ax.set_ylim(min_time_val - padding, max_time_val + padding + 0.5)


def add_plot_titles_styled(fig, year_val, event_name_val, drivers_abbr_list):
    """Add plot titles and subtitles with new style."""
    global suptitle_text_global, subtitle_lower_text_global
    suptitle_text_global = f"{year_val} {event_name_val} Grand Prix: Lap Time Variation"
    plt.suptitle(suptitle_text_global, fontsize=18, color="black")

    subtitle_upper = "with Trendlines per Stint and Pit Lap Annotations"
    subtitle_lower_text_global = (
        f"{drivers_abbr_list[0]} vs {drivers_abbr_list[1]}"
        if len(drivers_abbr_list) >= 2
        else drivers_abbr_list[0]
    )

    plt.figtext(0.5, 0.94, subtitle_upper, ha="center", fontsize=15, color="black")
    plt.figtext(
        0.5, 0.915, subtitle_lower_text_global, ha="center", fontsize=13, color="black"
    )


def save_plot_and_get_filename(fig, suptitle_text_param, dpi_val):
    """Save the plot to a file and return filename."""
    filename = f"../pic/{suptitle_text_param.replace(' ', '_').replace(':', '')}.png"
    fig.savefig(filename, dpi=dpi_val, bbox_inches=None)
    return filename


def create_styled_caption(year_val, event_name_val, stored_data_dict):
    """Create a styled caption for the plot."""
    base_title_for_caption = suptitle_text_global.replace(f"{year_val} ", "").replace(
        " Grand Prix: ", " - "
    )

    caption_parts = [
        "🏎️",
        f"« {year_val} {event_name_val} Grand Prix »",
        "",
        f"• {base_title_for_caption}",
        f"• Comparison: {subtitle_lower_text_global}",
        "",
    ]

    caption_parts.append("‣ Pit Laps (End of Stint):")
    for i, driver_abbr in enumerate(stored_data_dict["drivers_abbr_list"]):
        pit_laps_str = (
            " → ".join(stored_data_dict["pit_lap_caption_arrays"][i])
            if stored_data_dict["pit_lap_caption_arrays"][i]
            else "N/A"
        )
        caption_parts.append(
            f"\t◦ {driver_abbr}: {pit_laps_str if pit_laps_str else 'No recorded pit stops'}"
        )
    caption_parts.append("")

    caption_parts.append("‣ Tyre Compounds per Stint:")
    for i, driver_abbr in enumerate(stored_data_dict["drivers_abbr_list"]):
        compounds_str = (
            " → ".join(stored_data_dict["tire_type_arrays"][i])
            if stored_data_dict["tire_type_arrays"][i]
            else "N/A"
        )
        caption_parts.append(f"\t◦ {driver_abbr}: {compounds_str}")
    caption_parts.append("")

    caption_parts.append("‣ Lap Time Variation Rate (s/lap per Stint):")
    for i, driver_abbr in enumerate(stored_data_dict["drivers_abbr_list"]):
        slopes_str = (
            " → ".join(stored_data_dict["slope_str_arrays"][i])
            if stored_data_dict["slope_str_arrays"][i]
            else "N/A"
        )
        caption_parts.append(f"\t◦ {driver_abbr}: {slopes_str}")
    caption_parts.append("")

    caption_parts.append(f"#F1 #Formula1 #{event_name_val.replace(' ', '')}GP")
    return textwrap.dedent("\n".join(caption_parts))


def initialize_driver_data_storage():
    """Initialize the dictionary to store data for each driver."""
    return {
        "all_pit_lap_numbers": [],
        "legend_elements": [],
        "drivers_abbr_list": [],
        "slope_str_arrays": [],
        "tire_type_arrays": [],
        "pit_lap_caption_arrays": [],
    }


def process_driver_data_single(
    ax,
    race_obj,
    driver_abbr_val,
    driver_idx,
    stored_data_ref,
    driver_laps_df,
    assigned_driver_color,
    line_styles_list,
    global_y_anchor_top,
    global_plot_range_y,
    y_pos_for_pit_text,
):
    """Process and plot data for a single driver."""
    if driver_laps_df.empty:
        if driver_abbr_val not in stored_data_ref["drivers_abbr_list"]:
            stored_data_ref["drivers_abbr_list"].append(driver_abbr_val)
            for key in [
                "slope_str_arrays",
                "tire_type_arrays",
                "pit_lap_caption_arrays",
            ]:
                stored_data_ref[key].append(["N/A"])
        return

    if driver_abbr_val not in stored_data_ref["drivers_abbr_list"]:
        stored_data_ref["drivers_abbr_list"].append(driver_abbr_val)

    plot_driver_laps_styled(
        ax, driver_laps_df, driver_idx, driver_abbr_val, assigned_driver_color
    )

    stints_summary = get_stints_info_for_driver(race_obj, driver_abbr_val)
    driver_pit_laps = []
    driver_compounds_per_stint = []
    if not stints_summary.empty:
        driver_pit_laps = list(stints_summary["StintEndLap"][:-1])
        driver_compounds_per_stint = list(stints_summary["Compound"])

    stored_data_ref["all_pit_lap_numbers"].extend(driver_pit_laps)
    caption_pit_laps = (
        [str(lap) for lap in stints_summary["StintEndLap"]]
        if not stints_summary.empty
        else ["N/A"]
    )
    stored_data_ref["pit_lap_caption_arrays"].append(caption_pit_laps)
    stored_data_ref["tire_type_arrays"].append(
        driver_compounds_per_stint if driver_compounds_per_stint else ["N/A"]
    )

    plot_pit_lap_lines_styled(
        ax, driver_pit_laps, assigned_driver_color, y_pos_for_pit_text
    )

    slopes = plot_stint_trendlines_styled(
        ax,
        driver_laps_df,
        assigned_driver_color,
        driver_idx,
        line_styles_list,
        global_y_anchor_top,
        global_plot_range_y,
    )
    stored_data_ref["slope_str_arrays"].append(slopes if slopes else ["N/A"])

    stored_data_ref["legend_elements"].append(
        Line2D(
            [0],
            [0],
            marker=MARKERS[driver_idx % len(MARKERS)],
            color=assigned_driver_color,
            label=driver_abbr_val,
            markersize=8,
            linestyle="None",
        )
    )


def driver_laptimes_scatterplot(
    year: int, event_name: str, session_name: str, race, post: bool
) -> dict:
    """Plot driver lap times variation with pit lap annotations, styled."""
    fastf1.plotting.setup_mpl(
        mpl_timedelta_support=False, color_scheme=None, misc_mpl_mods=False
    )

    DPI = 125
    FIG_SIZE = (1080 / DPI, 1350 / DPI)

    load_race_data(race)

    podium_finishers_abbrs = get_podium_finishers_abbr(race)
    if not podium_finishers_abbrs:
        print(f"Not enough podium finishers for {year} {event_name}. Skipping.")
        return {"filename": None, "caption": "Not enough data for plot.", "post": False}

    driver_laps_map = {
        abbr: get_driver_laps_cleaned(race, abbr) for abbr in podium_finishers_abbrs
    }

    valid_driver_laps_list = [
        laps_df
        for laps_df in driver_laps_map.values()
        if laps_df is not None and not laps_df.empty
    ]

    if not valid_driver_laps_list:
        print(
            f"No valid lap data for any podium finishers in {year} {event_name}. Skipping."
        )
        return {
            "filename": None,
            "caption": "No valid lap data for plot.",
            "post": False,
        }

    all_laps_for_scaling = pd.concat(valid_driver_laps_list, ignore_index=True)

    min_overall_time = (
        all_laps_for_scaling["LapTime(s)"].min()
        if not all_laps_for_scaling.empty
        else None
    )
    max_overall_time = (
        all_laps_for_scaling["LapTime(s)"].max()
        if not all_laps_for_scaling.empty
        else None
    )

    global_y_anchor_top_val = None
    global_plot_range_y_val = None
    y_pos_for_pit_text_val = None

    if (
        min_overall_time is not None
        and max_overall_time is not None
        and not (pd.isna(min_overall_time) or pd.isna(max_overall_time))
    ):
        padding = (max_overall_time - min_overall_time) * 0.05
        final_y_min = min_overall_time - padding
        final_y_max = (
            max_overall_time + padding + 0.5
        )  # Adjusted for the new y-axis limit
        global_y_anchor_top_val = final_y_max
        global_plot_range_y_val = final_y_max - final_y_min
        if global_plot_range_y_val == 0:
            global_plot_range_y_val = 1
        y_pos_for_pit_text_val = max_overall_time * 0.99
    else:
        global_plot_range_y_val = 1

    stored_data = initialize_driver_data_storage()

    with plt.style.context(["science", "bright"]):
        plt.rcParams["figure.dpi"] = DPI
        plt.rcParams["savefig.dpi"] = DPI
        plt.rcParams["figure.autolayout"] = False
        plt.rcParams["figure.constrained_layout.use"] = False
        plt.rcParams["savefig.bbox"] = None

        fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        prop_cycle = plt.rcParams["axes.prop_cycle"]
        driver_plot_colors = [
            prop_cycle.by_key()["color"][i % len(prop_cycle.by_key()["color"])]
            for i in range(len(podium_finishers_abbrs))
        ]

        if global_y_anchor_top_val is None:
            current_y_lim_pre = ax.get_ylim()
            global_y_anchor_top_val = current_y_lim_pre[1]
            if global_plot_range_y_val == 1 and (
                current_y_lim_pre[1] - current_y_lim_pre[0] != 0
            ):
                global_plot_range_y_val = current_y_lim_pre[1] - current_y_lim_pre[0]

        processed_drivers_count = 0
        for i, driver_abbr_item in enumerate(podium_finishers_abbrs):
            driver_laps_data = driver_laps_map.get(driver_abbr_item)

            if driver_laps_data is None or driver_laps_data.empty:
                if driver_abbr_item not in stored_data["drivers_abbr_list"]:
                    stored_data["drivers_abbr_list"].append(driver_abbr_item)
                    for key_to_update in [
                        "slope_str_arrays",
                        "tire_type_arrays",
                        "pit_lap_caption_arrays",
                    ]:
                        stored_data[key_to_update].append(["N/A"])
                continue

            process_driver_data_single(
                ax,
                race,
                driver_abbr_item,
                i,
                stored_data,
                driver_laps_data,
                driver_plot_colors[i],
                LINES,
                global_y_anchor_top_val,
                global_plot_range_y_val,
                y_pos_for_pit_text_val,
            )
            processed_drivers_count += 1

        if processed_drivers_count == 0:
            plt.close(fig)
            print(
                f"No driver data could be processed for plotting in {year} {event_name}."
            )
            return {
                "filename": None,
                "caption": "Failed to process driver data.",
                "post": False,
            }

        set_plot_labels_styled(ax, race, min_overall_time, max_overall_time)
        add_plot_titles_styled(fig, year, event_name, stored_data["drivers_abbr_list"])

        compound_legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color=fastf1.plotting.COMPOUND_COLORS.get(c, "black"),
                label=c.capitalize(),
                linestyle="None",
                markersize=6,
            )
            for c in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN"]
        ]
        ax.legend(
            handles=compound_legend_handles,
            title="Tyre Compound",
            loc="lower right",
            fontsize=9,
            title_fontsize=11,
            labelcolor="black",
            facecolor=ax.get_facecolor(),
            edgecolor=ax.get_facecolor(),
            framealpha=0.5,
        )
        if ax.get_legend() and ax.get_legend().get_title():
            ax.get_legend().get_title().set_color("black")

        if stored_data["legend_elements"]:
            fig.legend(
                handles=stored_data["legend_elements"],
                title="Drivers",
                loc="upper left",
                bbox_to_anchor=(0.12, 0.88),
                fontsize=9,
                title_fontsize=11,
                labelcolor="black",
                facecolor=fig.get_facecolor(),
                edgecolor=fig.get_facecolor(),
                framealpha=0.5,
            )
            if fig.legends and fig.legends[0].get_title():
                fig.legends[0].get_title().set_color("black")

        suptitle_for_filename = (
            f"{year} {event_name} Grand Prix Driver Lap Time Variation"
        )
        filename = save_plot_and_get_filename(fig, suptitle_for_filename, DPI)
        plt.close(fig)

    caption = create_styled_caption(year, event_name, stored_data)
    return {"filename": filename, "caption": caption, "post": post}
