import matplotlib.pyplot as plt
import numpy as np
import fastf1
import fastf1.plotting  # Added for setup_mpl
import textwrap
import scienceplots  # 新增導入

# from auto_ig_post import auto_ig_post # Assuming this is not needed for styling
from fastf1.ergast import Ergast

# Global variable for suptitle, to be used by generate_caption
suptitle_text_global = ""

all_country_name = {
    "Bahrain": "Bahrain",
    "Saudi Arabian": "Saudi Arabia",
    "Australian": "Australia",
    "Japanese": "Japan",
    "Chinese": "China",
    "Miami": "USA",
    "Emilia Romagna": "Italy",  # Assuming Imola is in Italy
    "Monaco": "Monaco",
    "Canadian": "Canada",
    "Spanish": "Spain",
    "Austrian": "Austria",
    "British": "UK",
    "Hungarian": "Hungary",
    "Belgian": "Belgium",
    "Dutch": "Netherlands",
    "Italian": "Italy",
    "Azerbaijan": "Azerbaijan",
    "Singapore": "Singapore",
    "United States": "USA",  # Austin
    "Mexico": "Mexico",
    "São Paulo": "Brazil",
    "Las Vegas": "USA",
    "Qatar": "Qatar",
    "Abu Dhabi": "UAE",  # Assuming Abu Dhabi is in UAE
}


def find_circuit_index_by_country(country_name_param, allcircuitsinfo_param):
    """Find the index of a circuit in the list by country name."""
    if not country_name_param:  # Handle empty country_name_param
        return None
    for index, circuit in enumerate(allcircuitsinfo_param):
        if circuit["Location"]["country"] == country_name_param:
            return index
    return None


def rotate(xy, *, angle_rad):  # Renamed angle to angle_rad for clarity
    """Rotate a point (xy) by a given angle in radians."""
    rot_mat = np.array(
        [
            [np.cos(angle_rad), np.sin(angle_rad)],
            [-np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )
    return np.matmul(xy, rot_mat)


def get_circuit_info_by_country_name(year_val, event_name_val):  # Renamed
    ergast = Ergast()
    allcircuitsinfo = ergast.get_circuits(season=year_val, result_type="raw")
    country_name_to_find = all_country_name.get(event_name_val)  # Use .get for safety

    if not country_name_to_find:
        print(
            f"Warning: Country name for event '{event_name_val}' not found in mapping."
        )
        # Try to find by locality if event_name matches a known locality
        for circuit_data in allcircuitsinfo:
            if circuit_data["Location"]["locality"].lower() == event_name_val.lower():
                country_name_to_find = circuit_data["Location"]["country"]
                print(
                    f"Found country '{country_name_to_find}' by locality for event '{event_name_val}'."
                )
                break
        if not country_name_to_find:  # If still not found
            return None, None, None, None

    index = find_circuit_index_by_country(country_name_to_find, allcircuitsinfo)
    if index is None:
        # Fallback: try to find circuit by event name if it matches circuitId (e.g. "monaco")
        for idx, circuit in enumerate(allcircuitsinfo):
            if circuit["circuitId"].lower() == event_name_val.lower().replace(" ", "_"):
                index = idx
                break
        if index is None:
            print(f"Circuit info not found for {event_name_val} in year {year_val}.")
            return None, None, None, None

    circuitsinfo = allcircuitsinfo[index]
    circuit_name_val = circuitsinfo["circuitName"]
    location_val = circuitsinfo["Location"]
    locality_val = location_val["locality"]
    country_val = location_val["country"]
    return circuit_name_val, locality_val, country_val, location_val


def plot_track_styled(ax_param, track_pos, circuit_info_obj):  # Renamed params
    """Plot the styled track outline."""
    TRACK_LINEWIDTH = 1.5
    track_coords = track_pos.loc[:, ("X", "Y")].to_numpy()
    track_angle_rad = (
        circuit_info_obj.rotation / 180 * np.pi
    )  # Ensure this is in radians
    rotated_track_coords = rotate(track_coords, angle_rad=track_angle_rad)
    # Scienceplots will handle color, lw can be adjusted
    ax_param.plot(
        rotated_track_coords[:, 0],
        rotated_track_coords[:, 1],
        linewidth=TRACK_LINEWIDTH,
        label="Track Outline",
    )
    return rotated_track_coords, track_angle_rad


def annotate_corners_styled(
    ax_param, circuit_info_obj, track_angle_rad_param
):  # Renamed params
    """Annotate corners on the track plot with styled markers and text."""
    OFFSET_VECTOR = np.array([250, 0])  # Reduced offset
    CORNER_ANNOTATION_COLOR = "dimgray"
    CORNER_TEXT_COLOR = "white"
    ANNOTATION_LINESTYLE = "--"
    ANNOTATION_LINEWIDTH = 0.8
    SCATTER_SIZE = 150  # User adjusted
    TEXT_SIZE = 8  # User adjusted
    TEXT_WEIGHT = "bold"

    for _, corner_data in circuit_info_obj.corners.iterrows():
        txt_val = f"{corner_data['Number']}{corner_data['Letter']}"
        offset_angle_rad = corner_data["Angle"] / 180 * np.pi  # Ensure radians

        # Calculate annotation point position
        offset_x_val, offset_y_val = rotate(OFFSET_VECTOR, angle_rad=offset_angle_rad)
        text_x_abs = corner_data["X"] + offset_x_val
        text_y_abs = corner_data["Y"] + offset_y_val

        # Rotate annotation point and corner point to match track rotation
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


def add_scale_styled(ax_param):  # x_max, y_min removed, calculate from current limits
    """Add a styled scale bar to the plot."""
    SCALE_BAR_LENGTH_METERS = 500
    SCALE_LINE_COLOR = "black"
    SCALE_LINEWIDTH = 1.2  # User adjusted
    TEXT_FONTSIZE = 10

    x_lims = ax_param.get_xlim()
    y_lims = ax_param.get_ylim()

    # Position scale bar at bottom right, relative to current view
    scale_bar_text = f"{SCALE_BAR_LENGTH_METERS} m"

    # Calculate position based on a fraction of the view range
    margin_x = (x_lims[1] - x_lims[0]) * 0.05  # 5% margin from right
    margin_y = (y_lims[1] - y_lims[0]) * 0.05  # 5% margin from bottom

    scale_bar_x_start = x_lims[1] - margin_x - SCALE_BAR_LENGTH_METERS
    scale_bar_y_pos = y_lims[0] + margin_y

    ax_param.plot(
        [scale_bar_x_start, scale_bar_x_start + SCALE_BAR_LENGTH_METERS],
        [scale_bar_y_pos, scale_bar_y_pos],
        color=SCALE_LINE_COLOR,
        lw=SCALE_LINEWIDTH,
    )
    # Optional: add small vertical ticks at ends of scale bar
    tick_height = (y_lims[1] - y_lims[0]) * 0.01
    ax_param.plot(
        [scale_bar_x_start, scale_bar_x_start],
        [scale_bar_y_pos - tick_height, scale_bar_y_pos + tick_height],
        color=SCALE_LINE_COLOR,
        lw=SCALE_LINEWIDTH,
    )
    ax_param.plot(
        [
            scale_bar_x_start + SCALE_BAR_LENGTH_METERS,
            scale_bar_x_start + SCALE_BAR_LENGTH_METERS,
        ],
        [scale_bar_y_pos - tick_height, scale_bar_y_pos + tick_height],
        color=SCALE_LINE_COLOR,
        lw=SCALE_LINEWIDTH,
    )

    ax_param.text(
        scale_bar_x_start + SCALE_BAR_LENGTH_METERS / 2,
        scale_bar_y_pos + tick_height * 3,  # Position text above scale bar
        scale_bar_text,
        ha="center",
        va="bottom",
        fontsize=TEXT_FONTSIZE,
        color=SCALE_LINE_COLOR,
    )


def save_plot_and_get_filename(fig_param, suptitle_text_param, dpi_val):  # Renamed
    """Save the plot to a file and return filename."""
    filename = f"../pic/{suptitle_text_param.replace(' ', '_').replace(':', '')}.png"
    # plt.tight_layout(rect=[0, 0, 1, 0.88])  # Adjust rect for titles
    fig_param.savefig(
        filename, dpi=dpi_val, bbox_inches=None
    )  # Explicitly set bbox_inches
    return filename


def generate_styled_caption(
    year_val, event_name_val, circuit_name_val, country_val, locality_val
):  # Renamed
    """Generate a styled caption for the plot."""
    # Uses global suptitle_text_global for consistency if needed, or constructs locally
    # For this plot, the caption is quite specific to circuit details.

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
    global suptitle_text_global  # Ensure it's updated

    # Plotting constants
    DPI = 125
    FIG_SIZE = (1080 / DPI, 1350 / DPI)  # This will now be (1.8, 2.25)
    PRIMARY_TEXT_COLOR = "black"
    BACKGROUND_COLOR = "white"
    SUPTITLE_FONTSIZE = 18
    SUBTITLE_FONTSIZE = 15
    AXIS_LABEL_FONTSIZE = 14
    SUPTITLE_Y = 0.93  # User adjusted
    SUBTITLE_Y = 0.89  # User adjusted

    fastf1.plotting.setup_mpl(
        mpl_timedelta_support=False, color_scheme=None, misc_mpl_mods=False
    )
    race.load()  # Load session data

    circuit_name, locality, country, _ = get_circuit_info_by_country_name(
        year, event_name
    )

    if circuit_name is None:
        print(f"Could not retrieve circuit information for {year} {event_name}.")
        return {
            "filename": None,
            "caption": "Circuit information not found.",
            "post": False,
        }

    # Get a lap for track position data; fastest lap is a good choice
    try:
        lap_for_track = race.laps.pick_fastest()
        if lap_for_track is None or lap_for_track.empty:
            print(
                f"No fastest lap data available for {year} {event_name} {session_name}."
            )
            return {
                "filename": None,
                "caption": "No lap data for track plotting.",
                "post": False,
            }
        pos_data = lap_for_track.get_pos_data()
        if pos_data.empty:
            print(
                f"No telemetry position data for fastest lap in {year} {event_name} {session_name}."
            )
            return {
                "filename": None,
                "caption": "No telemetry position data.",
                "post": False,
            }

    except Exception as e:
        print(f"Error getting lap/position data: {e}")
        return {
            "filename": None,
            "caption": "Error processing lap data.",
            "post": False,
        }

    circuit_info_obj = race.get_circuit_info()

    with plt.style.context(["science", "bright"]):
        # Attempt to override scienceplots' potential dimension-altering rcParams
        # Force DPI settings
        plt.rcParams["figure.dpi"] = DPI
        plt.rcParams["savefig.dpi"] = DPI

        # Disable layout managers that might resize the figure
        plt.rcParams["figure.autolayout"] = False
        plt.rcParams["figure.constrained_layout.use"] = False

        # Ensure savefig does not use 'tight' bounding box from rcParams
        plt.rcParams["savefig.bbox"] = None

        fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)  # dpi here should be honored
        fig.patch.set_facecolor(BACKGROUND_COLOR)
        ax.set_facecolor(BACKGROUND_COLOR)

        _, track_angle_rad_val = plot_track_styled(ax, pos_data, circuit_info_obj)
        annotate_corners_styled(ax, circuit_info_obj, track_angle_rad_val)

        # Set titles and labels
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

        plt.axis("equal")  # Maintain aspect ratio
        # ax.legend(loc="upper right") # Legend for "Track Outline" if needed, scienceplots might handle
        # For this plot, a legend might be too simple; can be omitted if track is obvious.
        # If kept, ensure text color is black.
        # legend = ax.legend(loc="best", fontsize=10, title_fontsize=12)
        # if legend:
        #     plt.setp(legend.get_texts(), color='black')
        #     if legend.get_title(): legend.get_title().set_color('black')
        # Removing legend for cleaner look, as there's only one line for the track.

        add_scale_styled(ax)  # Add styled scale bar

        # Remove axis spines and ticks for a cleaner map look, if desired
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        # This makes it more map-like, but removes coordinate context. Keep for now.

        # Define the suptitle string that should be used for generating the filename,
        # matching the assumed original pattern.
        suptitle_for_filename = f"{year} {event_name} Grand Prix Circuit"

        filename = save_plot_and_get_filename(
            fig, suptitle_for_filename, DPI
        )  # Use specific suptitle for filename
        plt.close(fig)

    # generate_styled_caption uses suptitle_text_global which is set for display purposes
    caption = generate_styled_caption(year, event_name, circuit_name, country, locality)
    return {"filename": filename, "caption": caption, "post": post}
