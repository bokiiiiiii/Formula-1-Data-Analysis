import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
from matplotlib.collections import PolyCollection
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
import fastf1
import fastf1.plotting
import pandas as pd
import textwrap
import numpy as np
import scienceplots
import matplotlib
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="fastf1")
warnings.filterwarnings("ignore", category=UserWarning, module="fastf1")

QUICKLAP_THRESHOLD = 1.05
DEFAULT_CORRECTION = 0.05
MARKERS = ["."]
LINES = ["-"]

suptitle_text_global = ""
subtitle_lower_text_global = ""
optimized_k_global = DEFAULT_CORRECTION


def load_race_data(race):
    try:
        race.load()
    except Exception as e:
        raise RuntimeError(f"Error loading race data: {e}")


def get_winner_abbr(race):
    results = race.results
    if results is None or results.empty:
        print("Warning: Results are empty. Cannot pick winner.")
        return []
    return list(results["Abbreviation"][:1])


def optimize_fuel_correction(laps_df):
    """
    Finds the optimal universal fuel correction factor (k) by minimizing the
    Root Mean Square Deviation (RMSD) of the corrected mean pace across
    all overlapping stints of the same compound.
    """
    if laps_df.empty:
        return DEFAULT_CORRECTION

    compound_stint_counts = laps_df.groupby("Compound")["Stint"].nunique()
    repeat_compounds = compound_stint_counts[compound_stint_counts > 1].index.tolist()

    if not repeat_compounds:
        print(
            f"No repeating compounds found. Using default correction: {DEFAULT_CORRECTION}"
        )
        return DEFAULT_CORRECTION

    k_candidates = np.linspace(0.01, 0.08, 36)
    best_k = DEFAULT_CORRECTION
    min_rmsd_score = float("inf")

    def calculate_corrected_laps(k_val):
        return laps_df["LapTime(s)"] + k_val * (laps_df["LapNumber"] - 1)

    for k in k_candidates:
        corrected_laps = calculate_corrected_laps(k)
        laps_df["temp_corrected"] = corrected_laps

        total_rmsd = 0
        valid_comparisons = 0

        for comp in repeat_compounds:
            comp_data = laps_df[laps_df["Compound"] == comp]
            stint_ids = comp_data["Stint"].unique()

            for i in range(len(stint_ids)):
                for j in range(i + 1, len(stint_ids)):
                    stint_a_id = stint_ids[i]
                    stint_b_id = stint_ids[j]

                    stint_a = comp_data[comp_data["Stint"] == stint_a_id].set_index(
                        "StintLapNumber"
                    )
                    stint_b = comp_data[comp_data["Stint"] == stint_b_id].set_index(
                        "StintLapNumber"
                    )

                    common_laps = stint_a.index.intersection(stint_b.index)

                    if len(common_laps) >= 5:
                        diff = (
                            stint_a.loc[common_laps, "temp_corrected"]
                            - stint_b.loc[common_laps, "temp_corrected"]
                        )

                        rmsd = np.sqrt(np.mean(diff**2))

                        total_rmsd += rmsd
                        valid_comparisons += 1

        if valid_comparisons > 0:
            score = total_rmsd / valid_comparisons
            if score < min_rmsd_score:
                min_rmsd_score = score
                best_k = k

    laps_df.drop(columns=["temp_corrected"], inplace=True, errors="ignore")

    print(
        f"Optimized Fuel Correction Factor (RMSD Minimized): {best_k:.4f} s/lap (Default: {DEFAULT_CORRECTION})"
    )
    return best_k


def get_driver_laps_cleaned_fuel_corrected(race, driver_abbr):
    global optimized_k_global
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

    best_k = optimize_fuel_correction(laps)
    optimized_k_global = best_k

    laps["FuelCorrectedLapTime(s)"] = laps["LapTime(s)"] + best_k * (
        laps["LapNumber"] - 1
    )
    return laps


def plot_driver_laps_styled_fuel_corrected(
    ax, driver_laps_df, driver_idx, assigned_driver_color
):
    if hasattr(fastf1.plotting, "COMPOUND_COLORS"):
        palette = {**fastf1.plotting.COMPOUND_COLORS, "UNKNOWN": "black"}
    else:
        palette = {
            "SOFT": "red",
            "MEDIUM": "yellow",
            "HARD": "white",
            "INTERMEDIATE": "green",
            "WET": "blue",
            "UNKNOWN": "black",
        }

    sns.scatterplot(
        data=driver_laps_df,
        x="StintLapNumber",
        y="FuelCorrectedLapTime(s)",
        ax=ax,
        hue="Compound",
        palette=palette,
        hue_order=["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN"],
        marker=MARKERS[0],
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
        if len(group) < 3:
            continue

        if hasattr(fastf1.plotting, "COMPOUND_COLORS"):
            c_color = fastf1.plotting.COMPOUND_COLORS.get(compound_name, assigned_color)
        else:
            c_color = assigned_color

        reg_color = c_color if compound_name != "HARD" else "grey"

        X = group["StintLapNumber"].values.reshape(-1, 1)
        y = group["FuelCorrectedLapTime(s)"].values

        kernel = C(1.0, (1e-3, 1e3)) * RBF(
            length_scale=10.0, length_scale_bounds=(1.0, 30.0)
        ) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-3, 0.5))

        gpr = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=20, alpha=0.0
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gpr.fit(X, y)

            x_min, x_max = X.min(), X.max()
            X_plot = np.linspace(x_min, x_max, 100).reshape(-1, 1)

            y_pred, sigma = gpr.predict(X_plot, return_std=True)

            ax.plot(
                X_plot,
                y_pred,
                color=reg_color,
                linestyle=line_styles[0],
                linewidth=1.5,
                alpha=0.9,
                label=f"_nolegend_",
            )

            ax.fill_between(
                X_plot.ravel(),
                y_pred - sigma,
                y_pred + sigma,
                color=reg_color,
                alpha=0.15,
                edgecolor="none",
            )

            delta_y = y_pred[-1] - y_pred[0]
            delta_x = X_plot[-1][0] - X_plot[0][0]

            if delta_x != 0:
                avg_slope_val = delta_y / delta_x
                slope_info_list.append(
                    {"compound": compound_name, "slope": f"{avg_slope_val:+.3f} (avg)"}
                )

        except Exception as e:
            print(f"GPR Fit failed for {compound_name}: {e}")
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
        f"{year_val} {event_name_val} Grand Prix: Fuel-Corrected Lap Time"
    )
    plt.suptitle(suptitle_text_global, fontsize=18, color="black")

    subtitle_upper = f"with Gaussian Process Trendlines for {drivers_abbr_list[0]} Tyre Degradation Analysis"

    if drivers_abbr_list:
        subtitle_lower_text_global = f"Winner Pace Analysis: {drivers_abbr_list[0]}"
    else:
        subtitle_lower_text_global = "Pace Analysis"

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
        "🏎️",
        f"« {year_val} {event_name_val} Grand Prix »",
        "",
        f"• {base_title_for_caption}",
        f"• Focus: {subtitle_lower_text_global}",
        f"• Universal Fuel Correction Factor: {optimized_k_global:.3f} s/lap",
        "",
        "‣ Avg. Fuel-Corrected Lap Time Trend (s/lap) [GPR]:",
    ]
    for i, driver_abbr in enumerate(stored_data_dict["drivers_abbr_list"]):
        caption_parts.append(f"\t◦ {driver_abbr}:")
        slopes_for_driver = stored_data_dict["compound_slope_arrays"][i]
        if slopes_for_driver:
            compound_slope_strs = [
                f"{s_info['compound']}: {s_info['slope']}"
                for s_info in slopes_for_driver
            ]
            caption_parts.append("\t " + " | ".join(compound_slope_strs))
        else:
            caption_parts.append("\t N/A (No trendline data)")
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

    return True


def driver_fuel_corrected_laptimes_gaussian_processes(
    year: int, event_name: str, session_name: str, race, post: bool
) -> dict:
    fastf1.plotting.setup_mpl(
        mpl_timedelta_support=False, color_scheme=None, misc_mpl_mods=False
    )
    DPI = 125
    FIG_SIZE = (1080 / DPI, 1350 / DPI)

    load_race_data(race)

    drivers_to_plot_abbrs = get_winner_abbr(race)

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
        plt.rcParams["figure.dpi"] = DPI
        plt.rcParams["savefig.dpi"] = DPI
        plt.rcParams["figure.autolayout"] = False
        plt.rcParams["figure.constrained_layout.use"] = False
        plt.rcParams["savefig.bbox"] = None

        fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        prop_cycle = plt.rcParams["axes.prop_cycle"]
        driver_plot_colors_map = {
            abbr: prop_cycle.by_key()["color"][0]
            for i, abbr in enumerate(drivers_to_plot_abbrs)
        }

        if global_y_anchor_top_val is None:
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

        # --- Tyre Compound Legend ---
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

            compound_colors_dict = {}
            if hasattr(fastf1.plotting, "COMPOUND_COLORS"):
                compound_colors_dict = fastf1.plotting.COMPOUND_COLORS
            else:
                compound_colors_dict = {
                    "SOFT": "red",
                    "MEDIUM": "yellow",
                    "HARD": "white",
                    "INTERMEDIATE": "green",
                    "WET": "blue",
                    "UNKNOWN": "black",
                }

            compound_legend_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    markersize=6,
                    color=compound_colors_dict.get(c, "black"),
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
                fontsize=10,
                title_fontsize=12,
                labelcolor="black",
                facecolor=ax.get_facecolor(),
                edgecolor=ax.get_facecolor(),
                framealpha=0.5,
            )
            if compound_leg.get_title():
                compound_leg.get_title().set_color("black")
            ax.add_artist(compound_leg)

        # --- Gaussian Process Legend ---
        gp_line = Line2D([0], [0], color="black", linestyle="-", linewidth=1.5)
        gp_patch = Patch(facecolor="black", alpha=0.15, linewidth=0)

        gp_leg = ax.legend(
            handles=[(gp_line, gp_patch)],
            labels=[r"$\mu ~ \pm ~ \sigma$"],
            loc="upper left",
            bbox_to_anchor=(0.00, 1.00),
            title="Gaussian Process",
            fontsize=10,
            title_fontsize=12,
            labelcolor="black",
            facecolor=fig.get_facecolor(),
            edgecolor=fig.get_facecolor(),
            framealpha=0.5,
            handler_map={tuple: HandlerTuple(ndivide=None)},
        )
        if gp_leg.get_title():
            gp_leg.get_title().set_color("black")

        title_x_main = 0.98
        title_y_main = 0.98
        content_start_y = title_y_main - 0.030
        line_height_small = 0.028
        base_fontsize_val = 9
        indent_compound_text = 0.015

        title_text = f"Fuel Correction: {optimized_k_global:.3f} s/lap"
        ax.text(
            title_x_main,
            title_y_main,
            title_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=base_fontsize_val,
            color="black",
        )

        drivers_for_slope_legend = stored_driver_data["drivers_abbr_list"][:1]
        column_x_starts_legend = [0.82]

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

            current_y_col -= line_height_small

            for slope_info in slopes_for_this_driver:
                compound = slope_info["compound"]
                slope_val_str = slope_info["slope"]
                compound_abbr = (
                    compound[:3].upper() if len(compound) > 3 else compound.upper()
                )

                c_text_color = "black"
                if hasattr(fastf1.plotting, "COMPOUND_COLORS"):
                    c_text_color = fastf1.plotting.COMPOUND_COLORS.get(
                        compound, "black"
                    )

                if compound == "HARD":
                    c_text_color = "grey"

                compound_text_x = current_x_col_base + indent_compound_text
                slope_value_text_x = compound_text_x + 0.05

                current_y_col -= line_height_small * 0.9

        suptitle_for_filename = suptitle_text_global
        filename = save_plot_and_get_filename(fig, suptitle_for_filename, DPI)
        plt.close(fig)

    caption = create_styled_caption_fuel_corrected(year, event_name, stored_driver_data)
    return {"filename": filename, "caption": caption, "post": post}
