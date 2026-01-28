import matplotlib.pyplot as plt
import numpy as np
import fastf1
import fastf1.plotting
import textwrap
import scienceplots
import matplotlib

from fastf1.ergast import Ergast
from logger_config import get_logger

logger = get_logger(__name__)

suptitle_text_global = ""

all_country_name = {
    "Bahrain Grand Prix": "Bahrain",
    "Saudi Arabian Grand Prix": "Saudi Arabia",
    "Australian Grand Prix": "Australia",
    "Japanese Grand Prix": "Japan",
    "Chinese Grand Prix": "China",
    "Miami Grand Prix": "USA",
    "Emilia Romagna Grand Prix": "Italy",
    "Monaco Grand Prix": "Monaco",
    "Canadian Grand Prix": "Canada",
    "Spanish Grand Prix": "Spain",
    "Austrian Grand Prix": "Austria",
    "British Grand Prix": "UK",
    "Hungarian Grand Prix": "Hungary",
    "Belgian Grand Prix": "Belgium",
    "Dutch Grand Prix": "Netherlands",
    "Italian Grand Prix": "Italy",
    "Azerbaijan Grand Prix": "Azerbaijan",
    "Singapore Grand Prix": "Singapore",
    "United States Grand Prix": "USA",
    "Mexico City Grand Prix": "Mexico",
    "São Paulo Grand Prix": "Brazil",
    "Las Vegas Grand Prix": "USA",
    "Qatar Grand Prix": "Qatar",
    "Abu Dhabi Grand Prix": "UAE",
}


def find_circuit_index_by_country(country_name_param, allcircuitsinfo_param):
    if not country_name_param:
        return None
    for index, circuit in enumerate(allcircuitsinfo_param):
        if circuit["Location"]["country"] == country_name_param:
            return index
    return None


def rotate(xy, *, angle_rad):
    rot_mat = np.array(
        [
            [np.cos(angle_rad), np.sin(angle_rad)],
            [-np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )
    return np.matmul(xy, rot_mat)


def get_circuit_info_by_country_name(year_val, event_name_val):
    ergast = Ergast()
    allcircuitsinfo = ergast.get_circuits(season=year_val, result_type="raw")
    country_name_to_find = all_country_name.get(event_name_val)

    if not country_name_to_find:
        logger.warning(
            f"Country name for event '{event_name_val}' not found in mapping. Trying fallback lookup..."
        )
        for circuit_data in allcircuitsinfo:
            if circuit_data["Location"]["locality"].lower() == event_name_val.lower():
                country_name_to_find = circuit_data["Location"]["country"]
                logger.info(
                    f"Found country '{country_name_to_find}' by locality for event '{event_name_val}'."
                )
                break
        if not country_name_to_find:
            logger.warning(
                f"Could not find country mapping for '{event_name_val}'. Circuit data unavailable."
            )
            return None, None, None, None

    index = find_circuit_index_by_country(country_name_to_find, allcircuitsinfo)
    if index is None:
        for idx, circuit in enumerate(allcircuitsinfo):
            if circuit["circuitId"].lower() == event_name_val.lower().replace(" ", "_"):
                index = idx
                break
        if index is None:
            logger.warning(
                f"Circuit info not found for {event_name_val} in year {year_val}. Data may not be available yet."
            )
            return None, None, None, None

    circuitsinfo = allcircuitsinfo[index]
    circuit_name_val = circuitsinfo["circuitName"]
    location_val = circuitsinfo["Location"]
    locality_val = location_val["locality"]
    country_val = location_val["country"]
    return circuit_name_val, locality_val, country_val, location_val


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
    OFFSET_VECTOR = np.array([250, 0])
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
    filename = f"../pic/{suptitle_text_param.replace(' ', '_').replace(':', '')}.png"
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
    • Track corners annotated.

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

    fastf1.plotting.setup_mpl(
        mpl_timedelta_support=False, color_scheme=None, misc_mpl_mods=False
    )
    race.load()

    circuit_name, locality, country, _ = get_circuit_info_by_country_name(
        year, event_name
    )

    if circuit_name is None:
        logger.warning(
            f"Could not retrieve circuit information for {year} {event_name}. Skipping track plot. This is normal for upcoming races or incomplete season data."
        )
        return {
            "filename": None,
            "caption": "Circuit information not found.",
            "post": False,
        }

    try:
        lap_for_track = race.laps.pick_fastest()
        if lap_for_track is None or lap_for_track.empty:
            logger.warning(
                f"No fastest lap data available for {year} {event_name} {session_name}."
            )
            return {
                "filename": None,
                "caption": "No lap data for track plotting.",
                "post": False,
            }
        pos_data = lap_for_track.get_pos_data()
        if pos_data.empty:
            logger.warning(
                f"No telemetry position data for fastest lap in {year} {event_name} {session_name}."
            )
            return {
                "filename": None,
                "caption": "No telemetry position data.",
                "post": False,
            }

    except Exception as e:
        logger.error(f"Error getting lap/position data: {e}")
        return {
            "filename": None,
            "caption": "Error processing lap data.",
            "post": False,
        }

    circuit_info_obj = race.get_circuit_info()

    with plt.style.context(["science", "bright"]):
        plt.rcParams["figure.dpi"] = DPI
        plt.rcParams["savefig.dpi"] = DPI

        plt.rcParams["figure.autolayout"] = False
        plt.rcParams["figure.constrained_layout.use"] = False

        plt.rcParams["savefig.bbox"] = None

        fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
        fig.patch.set_facecolor(BACKGROUND_COLOR)
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
