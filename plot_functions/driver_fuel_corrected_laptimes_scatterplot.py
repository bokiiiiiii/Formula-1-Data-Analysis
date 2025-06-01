import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection
from sklearn.linear_model import LinearRegression
import fastf1
import fastf1.plotting
import pandas as pd
import scienceplots
import textwrap
import numpy as np

QUICKLAP_THRESHOLD = 1.07
CORRECTED_LAPTIME_PER_LAP = 0.05
MARKERS = [".", "*"]
LINES = ["-", "--"]

suptitle_text_global = ""
subtitle_lower_text_global = ""


def load_race_data(race):
    try:
        race.load()
    except Exception as e:
        raise RuntimeError(f"Error loading race data: {e}")


def get_podium_finishers_abbr(race):
    results = race.results
    if results is None or results.empty:
        print("Warning: Results are empty. Cannot pick finishers.")
        return []
    return list(results["Abbreviation"][: min(len(results), 2)])


def get_driver_laps_cleaned_fuel_corrected(race, driver_abbr):
    try:
        laps_orig = race.laps.pick_drivers(driver_abbr).pick_quicklaps(
            QUICKLAP_THRESHOLD
        )
    except Exception:
        try:
            driver_number = race.get_driver(driver_abbr)["DriverNumber"]
            laps_orig = race.laps.pick_drivers(driver_number).pick_quicklaps(
                QUICKLAP_THRESHOLD
            )
        except Exception as e_inner:
            print(f"Could not get laps for {driver_abbr}: {e_inner}")
            return pd.DataFrame()

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

    if "Stint" in laps.columns:
        laps["StintLapNumber"] = (
            laps.groupby("Stint")["LapNumber"]
            .rank(method="first", ascending=True)
            .astype(int)
        )
    else:
        print(
            f"Warning: No 'Stint' column for {driver_abbr}, cannot calculate StintLapNumber or fuel correction."
        )
        laps["StintLapNumber"] = laps["LapNumber"]
        laps["FuelCorrectedLapTime(s)"] = laps["LapTime(s)"]
        return laps

    laps["FuelCorrectedLapTime(s)"] = laps["LapTime(s)"] + CORRECTED_LAPTIME_PER_LAP * (
        laps["LapNumber"] - 1
    )
    return laps


def plot_driver_laps_styled_fuel_corrected(
    ax, driver_laps_df, driver_idx, assigned_driver_color
):
    sns.scatterplot(
        data=driver_laps_df,
        x="StintLapNumber",
        y="FuelCorrectedLapTime(s)",
        ax=ax,
        hue="Compound",
        palette={**fastf1.plotting.COMPOUND_COLORS, "UNKNOWN": "black"},
        hue_order=["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN"],
        marker=MARKERS[driver_idx % len(MARKERS)],
        s=60,
        linewidth=0.25,
        edgecolor=assigned_driver_color,
    )


def plot_stint_compound_trendlines_styled(
    ax,
    driver_laps_df,
    assigned_color,
    driver_idx,
    line_styles,
):
    slope_info_list = []
    driver_laps_compound_filled = driver_laps_df.copy()
    driver_laps_compound_filled["Compound"] = driver_laps_compound_filled[
        "Compound"
    ].fillna("UNKNOWN")
    compound_groups = driver_laps_compound_filled.groupby("Compound")

    for compound_name, group_data in compound_groups:
        group = group_data.dropna(subset=["StintLapNumber", "FuelCorrectedLapTime(s)"])
        if len(group) < 2:
            continue

        reg_color = (
            fastf1.plotting.COMPOUND_COLORS.get(compound_name, assigned_color)
            if compound_name != "HARD"
            else "grey"
        )

        collections_before = list(ax.collections)

        current_line_kws = {
            "linestyle": line_styles[driver_idx % len(line_styles)],
            "linewidth": 0.8,
            "alpha": 0.7,
        }
        sns.regplot(
            x="StintLapNumber",
            y="FuelCorrectedLapTime(s)",
            data=group,
            ax=ax,
            scatter=False,
            color=reg_color,
            line_kws=current_line_kws,
        )

        for collection_item in [
            c for c in ax.collections if c not in collections_before
        ]:
            if isinstance(collection_item, PolyCollection):
                collection_item.set_alpha(0.1)

        X_reg = group["StintLapNumber"].values.reshape(-1, 1)
        Y_reg = group["FuelCorrectedLapTime(s)"].values.reshape(-1, 1)
        try:
            reg_model = LinearRegression().fit(X_reg, Y_reg)
            slope_val = reg_model.coef_[0][0]
            slope_info_list.append(
                {"compound": compound_name, "slope": f"{slope_val:+.3f}"}
            )
        except ValueError:
            pass

    return slope_info_list


def set_plot_labels_styled_fuel_corrected(
    ax, min_time_val, max_time_val, max_stint_lap_num_val
):
    ax.set_xlabel("Stint Lap Number", fontsize=14, color="black")
    ax.set_ylabel("Fuel Corrected Lap Time (s)", fontsize=14, color="black")

    ax.tick_params(axis="x", colors="black")
    ax.tick_params(axis="y", colors="black")

    ax.set_xlim(
        0.5, max_stint_lap_num_val + 0.5 if not pd.isna(max_stint_lap_num_val) else 30
    )

    if (
        min_time_val is not None
        and max_time_val is not None
        and not (pd.isna(min_time_val) or pd.isna(max_time_val))
    ):
        padding = (max_time_val - min_time_val) * 0.05
        padding = 0.5 if padding == 0 else padding
        ax.set_ylim(min_time_val - padding, max_time_val + padding + 0.5)
    else:
        current_y_data = []
        for collection in ax.collections:
            if hasattr(collection, "get_offsets"):
                offsets = collection.get_offsets()
                if offsets.ndim == 2 and offsets.shape[1] >= 2:
                    current_y_data.extend(offsets[:, 1])
        for line in ax.get_lines():
            y_data = line.get_ydata()
            if isinstance(y_data, np.ndarray) and y_data.size > 0:
                current_y_data.extend(y_data)
        if current_y_data:
            min_y = np.min(current_y_data)
            max_y = np.max(current_y_data)
            padding = (max_y - min_y) * 0.05 if (max_y - min_y) > 0 else 0.5
            ax.set_ylim(min_y - padding, max_y + padding + 0.5)


def add_plot_titles_styled_fuel_corrected(
    fig, year_val, event_name_val, drivers_abbr_list
):
    global suptitle_text_global, subtitle_lower_text_global
    suptitle_text_global = (
        f"{year_val} {event_name_val} Grand Prix: Fuel-Corrected Lap Time Variation"
    )
    plt.suptitle(suptitle_text_global, fontsize=18, color="black")
    subtitle_upper = "with Trendlines per Compound"
    if drivers_abbr_list and len(drivers_abbr_list) >= 2:
        subtitle_lower_text_global = f"{drivers_abbr_list[0]} vs {drivers_abbr_list[1]}"
    elif drivers_abbr_list and len(drivers_abbr_list) == 1:
        subtitle_lower_text_global = f"{drivers_abbr_list[0]}"
    else:
        subtitle_lower_text_global = "Driver Comparison"
    plt.figtext(0.5, 0.94, subtitle_upper, ha="center", fontsize=15, color="black")
    plt.figtext(
        0.5, 0.915, subtitle_lower_text_global, ha="center", fontsize=13, color="black"
    )


def save_plot_and_get_filename(fig, suptitle_text_param, dpi_val):
    filename_safe_title = (
        suptitle_text_param.replace(" ", "_").replace(":", "").replace("/", "_")
    )
    filename = f"../pic/{filename_safe_title}.png"
    fig.savefig(filename, dpi=dpi_val)
    return filename


def create_styled_caption_fuel_corrected(year_val, event_name_val, stored_data_dict):
    base_title_for_caption = suptitle_text_global.replace(f"{year_val} ", "").replace(
        " Grand Prix: ", " - "
    )
    caption_parts = [
        f"🏎️",
        f"« {year_val} {event_name_val} Grand Prix »",
        "",
        f"• {base_title_for_caption}",
        f"• Comparison: {subtitle_lower_text_global}",
        "",
        "‣ Fuel-Corrected Lap Time Variation Rate by Compound (s/lap):",
    ]
    for i, driver_abbr in enumerate(stored_data_dict["drivers_abbr_list"]):
        caption_parts.append(f"\t◦ {driver_abbr}:")
        slopes_for_driver = stored_data_dict["compound_slope_arrays"][i]
        if slopes_for_driver:
            compound_slope_strs = [
                f"{s_info['compound']}: {s_info['slope']}"
                for s_info in slopes_for_driver
            ]
            caption_parts.append("\t  " + " | ".join(compound_slope_strs))
        else:
            caption_parts.append("\t  N/A (No trendline data)")
    caption_parts.append("")
    caption_parts.append(f"#F1 #Formula1 #{event_name_val.replace(' ', '')}GP")
    return textwrap.dedent("\n".join(caption_parts))


def initialize_driver_data_storage():
    return {
        "legend_elements": [],
        "drivers_abbr_list": [],
        "compound_slope_arrays": [],
        "all_laps_data_for_scaling": pd.DataFrame(),
        "min_overall_time_corrected": None,
        "max_overall_time_corrected": None,
        "max_stint_lap_overall": 0,
    }


def process_driver_data_single_fuel_corrected(
    ax,
    driver_abbr_val,
    driver_idx,
    stored_data_ref,
    driver_laps_df,
    assigned_driver_color,
    line_styles_list,
):
    if driver_abbr_val not in stored_data_ref["drivers_abbr_list"]:
        stored_data_ref["drivers_abbr_list"].append(driver_abbr_val)
        stored_data_ref["compound_slope_arrays"].append([])

    if driver_laps_df.empty or "FuelCorrectedLapTime(s)" not in driver_laps_df.columns:
        return False

    plot_driver_laps_styled_fuel_corrected(
        ax, driver_laps_df, driver_idx, assigned_driver_color
    )

    current_min_time = driver_laps_df["FuelCorrectedLapTime(s)"].min()
    current_max_time = driver_laps_df["FuelCorrectedLapTime(s)"].max()
    current_max_stint_lap = driver_laps_df["StintLapNumber"].max()

    if (
        stored_data_ref["min_overall_time_corrected"] is None
        or current_min_time < stored_data_ref["min_overall_time_corrected"]
    ):
        stored_data_ref["min_overall_time_corrected"] = current_min_time
    if (
        stored_data_ref["max_overall_time_corrected"] is None
        or current_max_time > stored_data_ref["max_overall_time_corrected"]
    ):
        stored_data_ref["max_overall_time_corrected"] = current_max_time
    if current_max_stint_lap > stored_data_ref["max_stint_lap_overall"]:
        stored_data_ref["max_stint_lap_overall"] = current_max_stint_lap

    columns_to_copy = ["StintLapNumber", "FuelCorrectedLapTime(s)", "Compound"]
    if "Compound" not in driver_laps_df.columns:
        columns_to_copy.remove("Compound")

    temp_df_for_concat = driver_laps_df[columns_to_copy].copy()

    if stored_data_ref["all_laps_data_for_scaling"].empty:
        stored_data_ref["all_laps_data_for_scaling"] = temp_df_for_concat
    else:
        s_cols = set(stored_data_ref["all_laps_data_for_scaling"].columns)
        t_cols = set(temp_df_for_concat.columns)

        if "Compound" in t_cols and "Compound" not in s_cols:
            stored_data_ref["all_laps_data_for_scaling"]["Compound"] = "UNKNOWN"
        elif "Compound" not in t_cols and "Compound" in s_cols:
            temp_df_for_concat["Compound"] = "UNKNOWN"

        stored_data_ref["all_laps_data_for_scaling"] = pd.concat(
            [stored_data_ref["all_laps_data_for_scaling"], temp_df_for_concat],
            ignore_index=True,
        ).drop_duplicates()

    slopes_info = plot_stint_compound_trendlines_styled(
        ax,
        driver_laps_df,
        assigned_driver_color,
        driver_idx,
        line_styles_list,
    )

    try:
        idx = stored_data_ref["drivers_abbr_list"].index(driver_abbr_val)
        stored_data_ref["compound_slope_arrays"][idx] = (
            slopes_info if slopes_info else []
        )
    except (ValueError, IndexError) as e:
        print(f"Error updating slopes for {driver_abbr_val}: {e}. Attempting recovery.")
        if driver_abbr_val in stored_data_ref["drivers_abbr_list"]:
            while len(stored_data_ref["compound_slope_arrays"]) < len(
                stored_data_ref["drivers_abbr_list"]
            ):
                stored_data_ref["compound_slope_arrays"].append([])
            try:
                idx = stored_data_ref["drivers_abbr_list"].index(driver_abbr_val)
                stored_data_ref["compound_slope_arrays"][idx] = (
                    slopes_info if slopes_info else []
                )
            except (ValueError, IndexError) as e_retry:
                print(f"Critical Error on retry for {driver_abbr_val}: {e_retry}")
        else:
            print(
                f"Critical Error: Driver {driver_abbr_val} not in list for slope update despite earlier add."
            )

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
    return True


def driver_fuel_corrected_laptimes_scatterplot(
    year: int, event_name: str, session_name: str, race, post: bool
) -> dict:
    global suptitle_text_global, subtitle_lower_text_global
    fastf1.plotting.setup_mpl(
        mpl_timedelta_support=False, color_scheme=None, misc_mpl_mods=False
    )
    DPI = 125
    FIG_SIZE = (1080 / DPI, 1350 / DPI)

    load_race_data(race)
    drivers_to_plot_abbrs = get_podium_finishers_abbr(race)
    if not drivers_to_plot_abbrs:
        return {"filename": None, "caption": "Not enough data for plot.", "post": False}

    driver_laps_map = {
        abbr: get_driver_laps_cleaned_fuel_corrected(race, abbr)
        for abbr in drivers_to_plot_abbrs
    }

    valid_drivers_abbrs = [
        abbr
        for abbr, laps_df in driver_laps_map.items()
        if laps_df is not None
        and not laps_df.empty
        and "FuelCorrectedLapTime(s)" in laps_df.columns
    ]
    if not valid_drivers_abbrs:
        return {
            "filename": None,
            "caption": "No valid fuel-corrected lap data for plot.",
            "post": False,
        }
    drivers_to_plot_abbrs = valid_drivers_abbrs

    stored_driver_data = initialize_driver_data_storage()

    temp_all_laps_for_scaling = []
    for abbr in drivers_to_plot_abbrs:
        laps_df = driver_laps_map.get(abbr)
        if (
            laps_df is not None
            and not laps_df.empty
            and "FuelCorrectedLapTime(s)" in laps_df.columns
        ):
            laps_df_copy = laps_df.copy()
            if "Compound" not in laps_df_copy.columns:
                laps_df_copy["Compound"] = "UNKNOWN"
            temp_all_laps_for_scaling.append(
                laps_df_copy[["FuelCorrectedLapTime(s)", "StintLapNumber", "Compound"]]
            )

    if not temp_all_laps_for_scaling:
        return {
            "filename": None,
            "caption": "No data for y-axis scaling.",
            "post": False,
        }

    concatenated_laps_for_scale_calc = pd.concat(
        temp_all_laps_for_scaling, ignore_index=True
    )
    min_overall_time_corr = concatenated_laps_for_scale_calc[
        "FuelCorrectedLapTime(s)"
    ].min()
    max_overall_time_corr = concatenated_laps_for_scale_calc[
        "FuelCorrectedLapTime(s)"
    ].max()

    global_y_anchor_top_val = None
    global_plot_range_y_val = 1.0

    if not (pd.isna(min_overall_time_corr) or pd.isna(max_overall_time_corr)):
        padding = (max_overall_time_corr - min_overall_time_corr) * 0.05
        padding = 0.5 if padding == 0 else padding
        final_y_min = min_overall_time_corr - padding
        final_y_max = max_overall_time_corr + padding + 0.5
        global_y_anchor_top_val = final_y_max
        global_plot_range_y_val = final_y_max - final_y_min
        if global_plot_range_y_val <= 0:
            global_plot_range_y_val = 1.0

    with plt.style.context(["science", "bright"]):
        plt.rcParams.update(
            {
                "figure.dpi": DPI,
                "savefig.dpi": DPI,
                "figure.autolayout": False,
                "figure.constrained_layout.use": False,
            }
        )
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        prop_cycle = plt.rcParams["axes.prop_cycle"]
        driver_plot_colors_map = {
            abbr: prop_cycle.by_key()["color"][i % len(prop_cycle.by_key()["color"])]
            for i, abbr in enumerate(drivers_to_plot_abbrs)
        }

        if global_y_anchor_top_val is None:  # Should be initialized if data was present
            current_y_lim_pre = ax.get_ylim()
            global_y_anchor_top_val = current_y_lim_pre[1]
            if global_plot_range_y_val == 1.0 and (
                current_y_lim_pre[1] - current_y_lim_pre[0] != 0
            ):
                global_plot_range_y_val = float(
                    current_y_lim_pre[1] - current_y_lim_pre[0]
                )

        processed_drivers_count = 0
        for i, driver_abbr_item in enumerate(drivers_to_plot_abbrs):
            driver_laps_data = driver_laps_map.get(driver_abbr_item)
            if driver_laps_data is None or driver_laps_data.empty:
                continue

            assigned_color = driver_plot_colors_map[driver_abbr_item]
            success = process_driver_data_single_fuel_corrected(
                ax,
                driver_abbr_item,
                i,
                stored_driver_data,
                driver_laps_data,
                assigned_color,
                LINES,
            )
            if success:
                processed_drivers_count += 1

        if processed_drivers_count == 0:
            plt.close(fig)
            return {
                "filename": None,
                "caption": "Failed to process driver data.",
                "post": False,
            }

        set_plot_labels_styled_fuel_corrected(
            ax,
            stored_driver_data["min_overall_time_corrected"],
            stored_driver_data["max_overall_time_corrected"],
            stored_driver_data["max_stint_lap_overall"],
        )
        add_plot_titles_styled_fuel_corrected(
            fig, year, event_name, stored_driver_data["drivers_abbr_list"]
        )

        compound_legend_handles = []
        if (
            not stored_driver_data["all_laps_data_for_scaling"].empty
            and "Compound" in stored_driver_data["all_laps_data_for_scaling"].columns
        ):
            unique_compounds = stored_driver_data["all_laps_data_for_scaling"][
                "Compound"
            ].unique()
            compound_order = [
                "SOFT",
                "MEDIUM",
                "HARD",
                "INTERMEDIATE",
                "WET",
                "UNKNOWN",
            ]
            ordered_unique_compounds = [
                c for c in compound_order if c in unique_compounds
            ]

            compound_legend_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    markersize=6,
                    color=fastf1.plotting.COMPOUND_COLORS.get(c, "black"),
                    label=c.capitalize(),
                    linestyle="None",
                )
                for c in ordered_unique_compounds
            ]
        if compound_legend_handles:
            compound_leg = ax.legend(
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
            if compound_leg.get_title():
                compound_leg.get_title().set_color("black")

        if stored_driver_data["legend_elements"]:
            driver_leg = fig.legend(
                handles=stored_driver_data["legend_elements"],
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
            if driver_leg.get_title():
                driver_leg.get_title().set_color("black")

        title_x_main = 0.98
        title_y_main = 0.98
        content_start_y = title_y_main - 0.030
        line_height_small = 0.028
        base_fontsize_val = 9
        indent_compound_text = 0.015

        ax.text(
            title_x_main,
            title_y_main,
            "Fuel-corrected Lap Time Variation (s/lap)",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=base_fontsize_val,
            color="black",
        )

        drivers_for_slope_legend = []
        for i, abbr_iter in enumerate(stored_driver_data["drivers_abbr_list"]):
            if (
                i < len(stored_driver_data["compound_slope_arrays"])
                and stored_driver_data["compound_slope_arrays"][i]
            ):
                drivers_for_slope_legend.append(abbr_iter)
            if len(drivers_for_slope_legend) == 2:
                break

        num_drivers_in_legend = len(drivers_for_slope_legend)
        column_x_starts_legend = [0.65, 0.82] if num_drivers_in_legend == 2 else [0.82]

        for col_idx, driver_abbr_leg in enumerate(drivers_for_slope_legend):
            current_x_col_base = column_x_starts_legend[col_idx]
            current_y_col = content_start_y

            original_driver_idx_in_storage = stored_driver_data[
                "drivers_abbr_list"
            ].index(driver_abbr_leg)
            driver_actual_color = driver_plot_colors_map.get(driver_abbr_leg, "black")

            slopes_for_this_driver = stored_driver_data["compound_slope_arrays"][
                original_driver_idx_in_storage
            ]

            ax.text(
                current_x_col_base,
                current_y_col,
                f"{driver_abbr_leg}:",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=base_fontsize_val,
                color=driver_actual_color,
            )
            current_y_col -= line_height_small

            for slope_info in slopes_for_this_driver:
                compound = slope_info["compound"]
                slope_val_str = slope_info["slope"]
                compound_abbr = (
                    compound[:3].upper() if len(compound) > 3 else compound.upper()
                )

                compound_text_color = (
                    fastf1.plotting.COMPOUND_COLORS.get(compound, "black")
                    if compound != "HARD"
                    else "grey"
                )

                compound_text_x = current_x_col_base + indent_compound_text
                slope_value_text_x = compound_text_x + 0.05

                ax.text(
                    compound_text_x,
                    current_y_col,
                    f"{compound_abbr}:",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=base_fontsize_val - 1,
                    color=compound_text_color,
                )
                ax.text(
                    slope_value_text_x,
                    current_y_col,
                    slope_val_str,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=base_fontsize_val - 1,
                    color="black",
                )
                current_y_col -= line_height_small * 0.9

        suptitle_for_filename = suptitle_text_global
        filename = save_plot_and_get_filename(fig, suptitle_for_filename, DPI)
        plt.close(fig)

    caption = create_styled_caption_fuel_corrected(year, event_name, stored_driver_data)
    return {"filename": filename, "caption": caption, "post": post}
