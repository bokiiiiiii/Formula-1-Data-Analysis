"""G-G diagram (friction circle) for the top 2 finishers overlaid on one plot."""

import logging
import textwrap

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import scienceplots
import fastf1
import fastf1.plotting
from scipy.signal import savgol_filter

from . import utils

logger = logging.getLogger(__name__)

G_ACC = 9.81
G_LIMIT = 7.0
LOW_SPEED_THR = 30.0
RESAMPLE_HZ = 20
SAVGOL_WIN = 31
SAVGOL_ORD = 3
QUICKLAP_THR = 1.08

DRIVER_CMAPS = ["Blues", "Reds"]
SCATTER_SIZE = 2
SCATTER_ALPHA = 0.55
G_CIRCLE_RADII = [1, 2, 3, 4, 5, 6, 7]


def _g_forces_single_lap(telemetry: pd.DataFrame):
    """Compute (G_long, G_lat, speed_kmh) for one lap."""
    n = len(telemetry)
    if n < SAVGOL_WIN:
        return None

    t = telemetry["Time"].dt.total_seconds().values.astype(float)
    spd = telemetry["Speed"].values.astype(float)
    x = telemetry["X"].values.astype(float)
    y = telemetry["Y"].values.astype(float)

    t -= t[0]
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, spd, x, y = t[keep], spd[keep], x[keep], y[keep]
    if len(t) < SAVGOL_WIN:
        return None

    dt = 1.0 / RESAMPLE_HZ
    t_u = np.arange(t[0], t[-1], dt)
    if len(t_u) < SAVGOL_WIN:
        return None

    spd = np.interp(t_u, t, spd)
    x = np.interp(t_u, t, x)
    y = np.interp(t_u, t, y)

    w = min(SAVGOL_WIN, len(x) if len(x) % 2 else len(x) - 1)
    if w < 5:
        return None

    x_s = savgol_filter(x, w, SAVGOL_ORD)
    y_s = savgol_filter(y, w, SAVGOL_ORD)
    v_s = savgol_filter(spd / 3.6, w, SAVGOL_ORD)

    G_long = np.gradient(v_s, dt) / G_ACC

    dx = np.gradient(x_s, dt)
    dy = np.gradient(y_s, dt)
    heading = np.unwrap(np.arctan2(dy, dx))
    heading = savgol_filter(heading, w, SAVGOL_ORD)
    yaw_rate = np.gradient(heading, dt)

    v_masked = np.where(spd > LOW_SPEED_THR, v_s, 0.0)
    G_lat = (v_masked * yaw_rate) / G_ACC

    ok = (np.abs(G_long) <= G_LIMIT) & (np.abs(G_lat) <= G_LIMIT)
    return G_long[ok], G_lat[ok], spd[ok]


def _collect_driver_g(race, abbr: str):
    """Aggregate G-data over all quick laps for a driver."""
    try:
        laps = race.laps.pick_driver(abbr).pick_quicklaps(QUICKLAP_THR)
    except Exception:
        laps = race.laps.pick_driver(abbr)
    if laps.empty:
        laps = race.laps.pick_driver(abbr)

    arrays = {"long": [], "lat": [], "spd": []}
    for _, lap in laps.iterlaps():
        try:
            tel = lap.get_telemetry()
            if tel is None or tel.empty:
                continue
            if not {"X", "Y", "Speed", "Time"}.issubset(tel.columns):
                continue
            res = _g_forces_single_lap(tel)
            if res is None:
                continue
            arrays["long"].append(res[0])
            arrays["lat"].append(res[1])
            arrays["spd"].append(res[2])
        except Exception:
            continue

    if not arrays["long"]:
        return None

    return (
        np.concatenate(arrays["long"]),
        np.concatenate(arrays["lat"]),
        np.concatenate(arrays["spd"]),
    )


def driver_g_g_diagram(
    year: int, event_name: str, session_name: str, race, post: bool
) -> dict:
    """Create a G-G diagram with both top 2 finishers overlaid on one plot."""
    utils.setup_fastf1_plotting()

    DPI = utils.DEFAULT_DPI
    FIG_SIZE = (1080 / DPI, 1350 / DPI)

    if race.results is None or len(race.results) < 2:
        return {
            "filename": None,
            "caption": "Not enough drivers in results.",
            "post": False,
        }

    drivers = list(race.results["Abbreviation"][:2])

    driver_data = {}
    for abbr in drivers:
        result = _collect_driver_g(race, abbr)
        if result is None:
            logger.warning(f"No G-data for {abbr}")
            continue
        driver_data[abbr] = result

    if len(driver_data) < 2:
        return {
            "filename": None,
            "caption": "Insufficient telemetry data for G-G diagram.",
            "post": False,
        }

    all_speeds = np.concatenate([d[2] for d in driver_data.values()])
    v_min, v_max = float(np.nanmin(all_speeds)), float(np.nanmax(all_speeds))
    norm = Normalize(vmin=v_min, vmax=v_max)

    for abbr in drivers:
        gl, ga, _ = driver_data[abbr]
        logger.info(
            f"{abbr}  |  max |Ax| = {np.max(np.abs(gl)):.2f} G  "
            f"|  max |Ay| = {np.max(np.abs(ga)):.2f} G"
        )

    with utils.apply_scienceplots_style():
        utils.configure_plot_params(DPI)

        fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
        fig.patch.set_facecolor("white")

        gs = gridspec.GridSpec(
            1,
            3,
            width_ratios=[1, 0.02, 0.02],
            wspace=0.05,
            left=0.10,
            right=0.92,
            top=0.90,
            bottom=0.06,
        )

        ax = fig.add_subplot(gs[0, 0])
        cax_0 = fig.add_subplot(gs[0, 1])
        cax_1 = fig.add_subplot(gs[0, 2])
        ax.set_facecolor("white")

        # Get team colours for legend
        driver_labels = {}
        for abbr in drivers:
            try:
                team = race.laps.pick_driver(abbr)["Team"].iloc[0]
                team_color = fastf1.plotting.get_team_color(team, race)
            except Exception:
                team = ""
                team_color = "black"
            driver_labels[abbr] = (team, team_color)

        # Plot both drivers (first driver behind, second on top)
        scatter_objs = {}
        for idx, abbr in enumerate(drivers):
            g_long, g_lat, spd = driver_data[abbr]
            cmap = plt.get_cmap(DRIVER_CMAPS[idx])
            sc = ax.scatter(
                g_lat,
                g_long,
                c=spd,
                cmap=cmap,
                norm=norm,
                s=SCATTER_SIZE,
                alpha=SCATTER_ALPHA,
                edgecolors="none",
                rasterized=True,
                zorder=2 + idx,
            )
            scatter_objs[abbr] = (sc, cmap, idx)

        # Reference friction circles
        for r in G_CIRCLE_RADII:
            circle = plt.Circle(
                (0, 0),
                r,
                fill=False,
                linestyle=":",
                color="grey",
                linewidth=0.5,
                alpha=0.35,
            )
            ax.add_patch(circle)
            ax.annotate(
                f"{r}G",
                xy=(r * 0.707, r * 0.707),
                fontsize=7,
                color="grey",
                alpha=0.6,
                ha="left",
                va="bottom",
            )

        # Symmetric axis limits
        all_g_lat = np.concatenate([driver_data[a][1] for a in drivers])
        all_g_long = np.concatenate([driver_data[a][0] for a in drivers])
        lim = (
            max(
                np.percentile(np.abs(all_g_lat), 99.5),
                np.percentile(np.abs(all_g_long), 99.5),
            )
            + 0.5
        )
        lim = min(lim, G_LIMIT)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.axhline(0, color="grey", linewidth=0.4)
        ax.axvline(0, color="grey", linewidth=0.4)

        ax.set_xlabel(r"Lateral Acceleration (G)", fontsize=12, color="black")
        ax.set_ylabel(r"Longitudinal Acceleration (G)", fontsize=12, color="black")
        ax.tick_params(colors="black", labelsize=10)
        ax.grid(True, linestyle=":", alpha=0.25, color="lightgrey")

        cax_list = [cax_0, cax_1]
        for idx, abbr in enumerate(drivers):
            sc, _, _ = scatter_objs[abbr]
            cbar = fig.colorbar(sc, cax=cax_list[idx])
            cbar.ax.tick_params(labelsize=7, colors="black")
            if idx == 0:
                cbar.ax.set_yticklabels([])
        cax_1.set_ylabel(
            "Speed (km/h)", fontsize=10, color="black", rotation=270, labelpad=14
        )

        # Match colorbar height to main axes
        ax_pos = ax.get_position()
        for c in cax_list:
            c_pos = c.get_position()
            c.set_position([c_pos.x0, ax_pos.y0, c_pos.width, ax_pos.height])

        # Legend with driver colours
        legend_handles = []
        for idx, abbr in enumerate(drivers):
            cmap = plt.get_cmap(DRIVER_CMAPS[idx])
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=cmap(0.7),
                    markersize=7,
                    label=abbr,
                )
            )
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            fontsize=11,
            facecolor="white",
            edgecolor="lightgrey",
            framealpha=0.8,
        )

        suptitle_display = f"{year} {event_name} Grand Prix: G-G Diagram"
        fig.suptitle(suptitle_display, fontsize=18, color="black")

        subtitle = f"{drivers[0]} vs {drivers[1]}"
        fig.text(0.5, 0.935, subtitle, ha="center", fontsize=14, color="black")

        suptitle_for_filename = f"{year} {event_name} Grand Prix G-G Diagram"
        filename = (
            f"../pic/{suptitle_for_filename.replace(' ', '_').replace(':', '')}.png"
        )
        fig.savefig(filename, dpi=DPI, bbox_inches=None)
        plt.close(fig)

    caption = textwrap.dedent(
        f"""\
    🏎️
    « {year} {event_name} Grand Prix »

    • G-G Diagram (Friction Circle)
    • {drivers[0]} vs {drivers[1]}

    ‣ Longitudinal G (braking / acceleration) vs Lateral G (cornering)
    ‣ Colour intensity indicates vehicle speed (dark = high speed)

    #F1 #Formula1 #{event_name.replace(" ", "")}GP #GGDiagram"""
    )

    return {"filename": filename, "caption": caption, "post": post}
