import matplotlib.pyplot as plt
import numpy as np
import fastf1
import textwrap

from auto_ig_post import auto_ig_post
from fastf1.ergast import Ergast

all_country_name = {
    "Bahrain": "Bahrain",
    "Saudi Arabian": "Saudi Arabia",
    "Australian": "Australia",
    "Japanese": "Japan",
    "Chinese": "China",
    "Miami": "USA",
    "Emilia Romagna": "",
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
    "United States": "United States",
    "Mexico": "Mexico",
    "São Paulo": "Brazil",
    "Las Vegas": "USA",
    "Qatar": "Qatar",
    "Abu Dhabi": "",
}


def find_circuit_index_by_country(country_name, allcircuitsinfo):
    for index, circuit in enumerate(allcircuitsinfo):
        if circuit["Location"]["country"] == country_name:
            return index
    return None


def rotate(xy, *, angle):
    rot_mat = np.array(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    )
    return np.matmul(xy, rot_mat)


def get_circuit_info_by_country(year, event_name):
    ergast = Ergast()
    allcircuitsinfo = ergast.get_circuits(season=year, result_type="raw")
    country_name = all_country_name[event_name]
    index = find_circuit_index_by_country(country_name, allcircuitsinfo)
    if index is None:
        return None, None, None, None
    circuitsinfo = allcircuitsinfo[index]
    circuit_name = circuitsinfo["circuitName"]
    location = circuitsinfo["Location"]
    locality = location["locality"]
    country = location["country"]
    return circuit_name, locality, country, location


def plot_track(ax, pos, circuit_info):
    track = pos.loc[:, ("X", "Y")].to_numpy()
    track_angle = circuit_info.rotation / 180 * np.pi
    rotated_track = rotate(track, angle=track_angle)
    ax.plot(rotated_track[:, 0], rotated_track[:, 1], label="Track")
    return rotated_track, track_angle


def annotate_corners(ax, circuit_info, rotated_track, track_angle):
    offset_vector = [500, 0]
    for _, corner in circuit_info.corners.iterrows():
        txt = f"{corner['Number']}{corner['Letter']}"
        offset_angle = corner["Angle"] / 180 * np.pi
        offset_x, offset_y = rotate(offset_vector, angle=offset_angle)
        text_x = corner["X"] + offset_x
        text_y = corner["Y"] + offset_y
        text_x, text_y = rotate([text_x, text_y], angle=track_angle)
        track_x, track_y = rotate([corner["X"], corner["Y"]], angle=track_angle)
        ax.scatter(
            text_x,
            text_y,
            color="grey",
            s=140,
            label="Corners" if corner["Number"] == 1 else "",
        )
        ax.plot([track_x, text_x], [track_y, text_y], color="grey")
        ax.text(
            text_x,
            text_y,
            txt,
            va="center_baseline",
            ha="center",
            size="small",
            color="white",
        )


def add_scale(ax, x_max, y_min):
    scale_length = 1000
    scale_text = f"{scale_length} m"
    scale_pos_x = x_max - 2 * scale_length
    scale_pos_y = y_min - 300
    ax.plot(
        [scale_pos_x, scale_pos_x + scale_length],
        [scale_pos_y, scale_pos_y],
        color="white",
        lw=1.7,
    )
    ax.plot(
        [scale_pos_x, scale_pos_x],
        [scale_pos_y, scale_pos_y + 100],
        color="white",
        lw=1.7,
    )
    ax.plot(
        [scale_pos_x + scale_length, scale_pos_x + scale_length],
        [scale_pos_y, scale_pos_y + 100],
        color="white",
        lw=1.7,
    )
    ax.text(
        scale_pos_x + scale_length / 2,
        scale_pos_y + 250,
        scale_text,
        ha="center",
        fontsize=10,
    )


def save_plot(fig, filename):
    plt.tight_layout()
    plt.savefig(filename)


def generate_caption(year, event_name, circuit_name, country, locality):
    return textwrap.dedent(
        f"""\
        🏎️
        « {year} {event_name} Grand Prix »

        • Circuit: {circuit_name}
        • Country: {country}
        • Locality: {locality}

        #F1 #Formula1 #{event_name.replace(" ", "")}GP"""
    )


def plot_track_with_annotated_corners(
    year: int, event_name: str, session_name: str, race, post: bool
) -> dict:
    race.load()
    circuit_name, locality, country, location = get_circuit_info_by_country(
        year, event_name
    )

    if circuit_name is None:
        return {"filename": None, "caption": "Country not found.", "post": post}

    lap = race.laps.pick_fastest()
    pos = lap.get_pos_data()
    circuit_info = race.get_circuit_info()

    fig, ax = plt.subplots(figsize=(10.8, 10.8), dpi=100, linewidth=0)
    rotated_track, track_angle = plot_track(ax, pos, circuit_info)
    annotate_corners(ax, circuit_info, rotated_track, track_angle)

    suptitle = f"{year} {event_name} Grand Prix Circuit"
    plt.suptitle(suptitle, fontweight="bold", fontsize=16)
    ax.set_xlabel("X Location (m)", fontweight="bold", fontsize=14)
    ax.set_ylabel("Y Location (m)", fontweight="bold", fontsize=14)
    plt.axis("equal")
    ax.legend(loc="upper right")

    x_max = plt.xlim()[1]
    y_min = plt.ylim()[0]
    add_scale(ax, x_max, y_min)

    subtitle = "with Track Corners Annotated"
    bg_color = ax.get_facecolor()
    plt.figtext(
        0.5,
        0.935,
        subtitle,
        ha="center",
        fontsize=14,
        bbox=dict(facecolor=bg_color, alpha=0.5, edgecolor="none"),
    )

    filename = f"../pic/{suptitle.replace(' ', '_')}.png"
    save_plot(fig, filename)

    caption = generate_caption(year, event_name, circuit_name, country, locality)
    return {"filename": filename, "caption": caption, "post": post}
