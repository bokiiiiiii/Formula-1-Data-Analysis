import matplotlib.pyplot as plt
import numpy as np
import textwrap
import pandas as pd
from types import SimpleNamespace

from logger_config import get_logger
from . import utils

logger = get_logger(__name__)

suptitle_text_global = ""

def rotate(xy, *, angle_rad):
    rot_mat = np.array(
        [
            [np.cos(angle_rad), np.sin(angle_rad)],
            [-np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )
    return np.matmul(xy, rot_mat)


def plot_track_styled(ax_param, track_pos, circuit_info_obj):
    TRACK_LINEWIDTH = 1.5
    track_coords = track_pos.loc[:, ("X", "Y")].to_numpy()
    track_angle_rad = circuit_info_obj.rotation / 180 * np.pi
    rotated_track_coords = rotate(track_coords, angle_rad=track_angle_rad)
    ax_param.plot(
        rotated_track_coords[:, 0],
        rotated_track_coords[:, 1],
        linewidth=TRACK_LINEWIDTH,
        label="Track Outline",
    )
    return rotated_track_coords, track_angle_rad


def annotate_corners_styled(ax_param, circuit_info_obj, track_angle_rad_param):
    OFFSET_VECTOR = np.array([25, 0])
    CORNER_ANNOTATION_COLOR = "dimgray"
    CORNER_TEXT_COLOR = "white"
    ANNOTATION_LINESTYLE = "--"
    ANNOTATION_LINEWIDTH = 0.8
    SCATTER_SIZE = 150
    TEXT_SIZE = 8
    TEXT_WEIGHT = "bold"

    for _, corner_data in circuit_info_obj.corners.iterrows():
        txt_val = f"{corner_data['Number']}{corner_data['Letter']}"
        offset_angle_rad = corner_data["Angle"] / 180 * np.pi

        offset_x_val, offset_y_val = rotate(OFFSET_VECTOR, angle_rad=offset_angle_rad)
        text_x_abs = corner_data["X"] + offset_x_val
        text_y_abs = corner_data["Y"] + offset_y_val

        text_x_rot, text_y_rot = rotate(
            np.array([[text_x_abs, text_y_abs]]), angle_rad=track_angle_rad_param
        )[0]
        track_x_rot, track_y_rot = rotate(
            np.array([[corner_data["X"], corner_data["Y"]]]),
            angle_rad=track_angle_rad_param,
        )[0]

        ax_param.plot(
            [track_x_rot, text_x_rot],
            [track_y_rot, text_y_rot],
            color=CORNER_ANNOTATION_COLOR,
            linestyle=ANNOTATION_LINESTYLE,
            linewidth=ANNOTATION_LINEWIDTH,
        )
        ax_param.scatter(
            text_x_rot,
            text_y_rot,
            color=CORNER_ANNOTATION_COLOR,
            s=SCATTER_SIZE,
            zorder=3,
        )
        ax_param.text(
            text_x_rot,
            text_y_rot,
            txt_val,
            va="center",
            ha="center",
            size=TEXT_SIZE,
            color=CORNER_TEXT_COLOR,
            weight=TEXT_WEIGHT,
            zorder=4,
        )


def add_scale_styled(ax_param):
    SCALE_BAR_LENGTH_METERS = 500
    SCALE_LINE_COLOR = "black"
    SCALE_LINEWIDTH = 1.2
    TEXT_FONTSIZE = 10

    x_lims = ax_param.get_xlim()
    y_lims = ax_param.get_ylim()

    scale_bar_text = f"{SCALE_BAR_LENGTH_METERS} m"

    margin_x = (x_lims[1] - x_lims[0]) * 0.05
    margin_y = (y_lims[1] - y_lims[0]) * 0.05

    scale_bar_x_start = x_lims[1] - margin_x - SCALE_BAR_LENGTH_METERS
    scale_bar_y_pos = y_lims[0] + margin_y


def save_plot_and_get_filename(fig_param, suptitle_text_param, dpi_val):
    filename = utils.plot_path(suptitle_text_param)
    fig_param.savefig(filename, dpi=dpi_val, bbox_inches=None)
    return filename


def generate_styled_caption(
    year_val, event_name_val, circuit_name_val, country_val, locality_val
):
    caption = textwrap.dedent(
        f"""\
    🏎️
    « {year_val} {event_name_val} Grand Prix »

    • Circuit Layout: {circuit_name_val}
    • Location: {locality_val}, {country_val}
    • Track outline from OpenF1 position samples.

    #F1 #Formula1 #{event_name_val.replace(" ", "")}GP #{circuit_name_val.replace(" ","")}"""
    )
    return caption


def plot_track_with_annotated_corners(
    year: int, event_name: str, session_name: str, race, post: bool
) -> dict:
    global suptitle_text_global

    DPI = 125
    FIG_SIZE = (1080 / DPI, 1350 / DPI)
    PRIMARY_TEXT_COLOR = "black"
    BACKGROUND_COLOR = "white"
    SUPTITLE_FONTSIZE = 18
    SUBTITLE_FONTSIZE = 15
    AXIS_LABEL_FONTSIZE = 14
    SUPTITLE_Y = 0.93
    SUBTITLE_Y = 0.89


    circuit_name = race.meeting["circuit_short_name"]
    locality = race.meeting["location"]
    country = race.meeting["country_name"]

    try:
        lap_for_track = race.fastest_lap()
        if lap_for_track is None or lap_for_track.empty:
            logger.warning(
                f"No fastest lap data available for {year} {event_name} {session_name}."
            )
            return {
                "filename": None,
                "caption": "No lap data for track plotting.",
                "post": False,
            }
        pos_data = race.position_data(lap_for_track)
        if pos_data.empty:
            logger.warning(
                f"No telemetry position data for fastest lap in {year} {event_name} {session_name}."
            )
            return {
                "filename": None,
                "caption": "No telemetry position data.",
                "post": False,
            }

        circuit_info_obj = race.circuit
        if circuit_info_obj is None:
            logger.warning(
                "Circuit metadata unavailable for %s; plotting track without corner labels.",
                event_name,
            )
            circuit_info_obj = SimpleNamespace(
                rotation=0,
                corners=pd.DataFrame(columns=["Number", "Letter", "Angle", "X", "Y"]),
            )

    except Exception as e:
        logger.error(f"Error getting lap/position data: {e}")
        return {
            "filename": None,
            "caption": "Error processing lap data.",
            "post": False,
        }

    with utils.apply_scienceplots_style():
        utils.configure_plot_params(DPI)
        fig, ax = utils.create_styled_figure(FIG_SIZE, DPI, BACKGROUND_COLOR)
        ax.set_facecolor(BACKGROUND_COLOR)

        _, track_angle_rad_val = plot_track_styled(ax, pos_data, circuit_info_obj)
        annotate_corners_styled(ax, circuit_info_obj, track_angle_rad_val)

        suptitle_text_global = f"{year} {event_name} Grand Prix: Circuit Layout"
        plt.suptitle(
            suptitle_text_global,
            fontsize=SUPTITLE_FONTSIZE,
            color=PRIMARY_TEXT_COLOR,
            y=SUPTITLE_Y,
        )

        subtitle_text = f"{circuit_name} - {locality}, {country}"
        plt.figtext(
            0.5,
            SUBTITLE_Y,
            subtitle_text,
            ha="center",
            fontsize=SUBTITLE_FONTSIZE,
            color=PRIMARY_TEXT_COLOR,
        )

        ax.set_xlabel(
            "X Coordinate (m)", fontsize=AXIS_LABEL_FONTSIZE, color=PRIMARY_TEXT_COLOR
        )
        ax.set_ylabel(
            "Y Coordinate (m)", fontsize=AXIS_LABEL_FONTSIZE, color=PRIMARY_TEXT_COLOR
        )
        ax.tick_params(axis="x", colors=PRIMARY_TEXT_COLOR)
        ax.tick_params(axis="y", colors=PRIMARY_TEXT_COLOR)

        plt.axis("equal")

        add_scale_styled(ax)

        suptitle_for_filename = f"{year} {event_name} Grand Prix Circuit"

        filename = save_plot_and_get_filename(fig, suptitle_for_filename, DPI)
        plt.close(fig)

    caption = generate_styled_caption(year, event_name, circuit_name, country, locality)
    return {"filename": filename, "caption": caption, "post": post}
