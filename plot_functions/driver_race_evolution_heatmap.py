import matplotlib

matplotlib.use("Agg")  # Prevent Tcl/Tk errors

import seaborn as sns
from matplotlib import pyplot as plt
import fastf1
import fastf1.plotting
import pandas as pd
import textwrap
import numpy as np
import scipy.ndimage as ndimage
import scienceplots
import warnings

# Configuration
warnings.filterwarnings("ignore", category=FutureWarning, module="fastf1")
warnings.filterwarnings("ignore", category=UserWarning, module="fastf1")

# Global constants (Matching Reference exactly)
DPI = 125
FIG_SIZE = (1080 / DPI, 1350 / DPI)
SMOOTHING_SIGMA = [8, 1]
CONTOUR_STEP = 50  # km/h


def load_race_data(race):
    try:
        race.load(telemetry=True, laps=True, weather=False)
    except Exception as e:
        raise RuntimeError(f"Error loading race data: {e}")


def get_winner_abbr(race):
    if race.results is None or race.results.empty:
        return []
    return list(race.results["Abbreviation"][:1])


def prepare_contour_data(race, driver_abbr, dist_bins=800):
    """Interpolates telemetry data onto a structured grid."""
    try:
        driver_laps = race.laps.pick_driver(driver_abbr).pick_quicklaps(threshold=1.15)
    except Exception:
        driver_laps = race.laps.pick_driver(driver_abbr)

    if driver_laps.empty:
        return None, None, None

    fastest_lap = driver_laps.pick_fastest()
    if fastest_lap is None:
        return None, None, None

    circuit_length = fastest_lap.get_telemetry()["Distance"].max()
    dist_axis = np.linspace(0, circuit_length, num=dist_bins)

    speed_matrix_list = []
    lap_numbers = []

    for _, lap in driver_laps.iterrows():
        try:
            tel = lap.get_telemetry()
            speed_interp = np.interp(dist_axis, tel["Distance"], tel["Speed"])
            speed_matrix_list.append(speed_interp)
            lap_numbers.append(lap["LapNumber"])
        except Exception:
            continue

    if not speed_matrix_list:
        return None, None, None

    # Shape: (Distance, Laps)
    matrix = np.array(speed_matrix_list).T
    return matrix, np.array(lap_numbers), dist_axis


def configure_plot_style(ax, year, event, driver, lap_numbers, dist_axis):
    """Sets titles, labels, and axis limits."""
    # Titles
    title_main = f"{year} {event} GP: {driver} Spatio-Temporal Speed Evolution"
    title_upper = f"Telemetry Topology for Race Pace Analysis"
    title_lower = f""

    plt.suptitle(title_main, fontsize=18, color="black", weight="bold")
    plt.figtext(0.5, 0.94, title_upper, ha="center", fontsize=15, color="black")
    plt.figtext(0.5, 0.915, title_lower, ha="center", fontsize=13, color="black")

    # Labels and Limits
    ax.set_xlabel("Lap Number", fontsize=14, color="black")
    ax.set_ylabel("Track Distance (m)", fontsize=14, color="black")

    ax.tick_params(axis="both", colors="black", labelsize=12)
    ax.set_xlim(lap_numbers.min(), lap_numbers.max())
    ax.set_ylim(dist_axis.min(), dist_axis.max())

    return title_main, title_lower


def plot_contour(ax, matrix, x_vals, y_vals):
    """Draws the smoothed contour map and synchronized colorbar."""
    # Smoothing
    zi = ndimage.gaussian_filter(matrix, sigma=SMOOTHING_SIGMA)

    # Grid generation
    xi, yi = np.meshgrid(x_vals, y_vals)

    # Level definition
    v_min, v_max = np.min(zi), np.max(zi)
    levels_fill = np.linspace(v_min, v_max, 40)

    start_level = np.ceil(v_min / CONTOUR_STEP) * CONTOUR_STEP
    end_level = np.floor(v_max / CONTOUR_STEP) * CONTOUR_STEP

    if start_level > end_level:
        levels_line = np.array([int(v_min), int(v_max)])
    else:
        levels_line = np.arange(start_level, end_level + CONTOUR_STEP, CONTOUR_STEP)

    # Plotting
    contour_filled = ax.contourf(
        xi, yi, zi, levels=levels_fill, cmap="RdBu_r", alpha=0.9
    )
    contour_lines = ax.contour(
        xi, yi, zi, levels=levels_line, colors="black", linewidths=0.6, alpha=0.6
    )

    ax.clabel(contour_lines, inline=True, fontsize=10, fmt="%.0f")

    # Colorbar configuration
    cbar = plt.colorbar(contour_filled, ax=ax, pad=0.02, aspect=30, ticks=levels_line)
    cbar.set_label("Speed (km/h)", fontsize=14, labelpad=10, color="black")
    cbar.ax.yaxis.set_tick_params(color="black", labelcolor="black", labelsize=12)
    cbar.outline.set_edgecolor("black")
    cbar.add_lines(contour_lines)


def save_plot(fig, title):
    filename_safe = title.replace(" ", "_").replace(":", "").replace("/", "_")
    path = f"../pic/{filename_safe}.png"
    # Note: Do NOT use bbox_inches='tight' if you want exact dimensions
    fig.savefig(path, dpi=DPI)
    return path


def generate_caption(year, event, driver, main_title, lower_title):
    return textwrap.dedent(
        f"""
        🏎️
        « {year} {event} Grand Prix »
        
        • {main_title}
        • Driver: {driver}
        • {lower_title}
        
        ‣ Visualization Details:
        \t◦ X-Axis: Temporal evolution (Lap 1 → End)
        \t◦ Y-Axis: Spatial position (Start → Finish line)
        \t◦ Iso-Lines: Speed steps of {CONTOUR_STEP} km/h
        
        #F1 #Formula1 #{event.replace(' ', '')}GP #{driver} #DataViz
    """
    ).strip()


def driver_race_evolution_heatmap(
    year: int, event_name: str, session_name: str, race, post: bool
) -> dict:
    # Initialization
    fastf1.plotting.setup_mpl(
        mpl_timedelta_support=False, color_scheme=None, misc_mpl_mods=False
    )
    load_race_data(race)

    target_drivers = get_winner_abbr(race)
    if not target_drivers:
        return {"filename": None, "caption": "No results found.", "post": False}

    driver_abbr = target_drivers[0]
    print(f"Processing contour data for {driver_abbr}...")

    matrix, laps, dists = prepare_contour_data(race, driver_abbr)
    if matrix is None:
        return {
            "filename": None,
            "caption": f"No valid data for {driver_abbr}.",
            "post": False,
        }

    # Plotting context
    try:
        with plt.style.context(["science", "bright"]):
            # --- CRITICAL: Enforce exact dimensions like Reference Code ---
            plt.rcParams["figure.dpi"] = DPI
            plt.rcParams["savefig.dpi"] = DPI
            plt.rcParams["figure.autolayout"] = False
            plt.rcParams["figure.constrained_layout.use"] = False
            plt.rcParams["savefig.bbox"] = None
            # -----------------------------------------------------------

            fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")

            plot_contour(ax, matrix, laps, dists)
            title_main, title_lower = configure_plot_style(
                ax, year, event_name, driver_abbr, laps, dists
            )

            filename = save_plot(fig, title_main)
            plt.close(fig)

            caption = generate_caption(
                year, event_name, driver_abbr, title_main, title_lower
            )
            return {"filename": filename, "caption": caption, "post": post}

    except Exception as e:
        print(f"Plotting failed: {e}")
        plt.close("all")
        return {"filename": None, "caption": "Plot generation failed.", "post": False}
