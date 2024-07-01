import matplotlib.pyplot as plt
import numpy as np
import fastf1, textwrap

from auto_ig_post import auto_ig_post
from fastf1.ergast import Ergast

all_country_name = {
   'Bahrain': 'Bahrain',
   'Saudi Arabian': 'Saudi Arabia',
   'Australian': 'Australia',
   'Japanese': 'Japan',
   'Chinese': 'China',
   'Miami': 'USA',
   'Emilia Romagna': '',
   'Monaco': 'Monaco',
   'Canadian': 'Canada',
   'Spanish': 'Spain',
   'Austrian': 'Austria',
   'British': 'UK',
   'Hungarian': 'Hungary',
   'Belgian': 'Belgium',
   'Dutch': '',
   'Italian': 'Italy',
   'Azerbaijan': 'Azerbaijan',
   'Singapore': 'Singapore',
   'United States': 'United States',
   'Mexico City': 'Mexico',
   'São Paulo': '',
   'Las Vegas': 'USA',
   'Qatar': 'Qatar',
   'Abu Dhabi': '',
}

def find_circuit_index_by_country(country_name, allcircuitsinfo):
    for index, circuit in enumerate(allcircuitsinfo):
        if circuit['Location']['country'] == country_name:
            return index
        

def rotate(xy, *, angle):
    rot_mat = np.array(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    )
    return np.matmul(xy, rot_mat)


# @brief plot_track_with_annotated_corners: Plot the track map with annotated corners
def plot_track_with_annotated_corners(
    Year: int, EventName: str, SessionName: str, race, post: bool
) -> dict:

    race.load()
    ergast = Ergast()
    
    allcircuitsinfo = ergast.get_circuits(season=Year, result_type='raw')
    country_name = all_country_name[EventName]
    index = find_circuit_index_by_country(country_name, allcircuitsinfo)
    print(index)
    print(type(index))
    if not index:
        circuitName = ''
        Location = ''
        locality = ''
        country = ''
        print("Country not found.")
    else:    
        circuitsinfo = ergast.get_circuits(season=Year, result_type='raw')[index]
        circuitName = circuitsinfo['circuitName']
        Location = circuitsinfo['Location']
        locality = Location['locality']
        country = Location['country']
    
    lap = race.laps.pick_fastest()
    pos = lap.get_pos_data()

    circuit_info = race.get_circuit_info()

    print(circuit_info)

    track = pos.loc[:, ("X", "Y")].to_numpy()

    track_angle = circuit_info.rotation / 180 * np.pi

    rotated_track = rotate(track, angle=track_angle)

    fig, ax = plt.subplots(figsize=(10.8, 10.8), dpi=100, linewidth=0)

    plt.plot(rotated_track[:, 0], rotated_track[:, 1], label="Track")

    offset_vector = [500, 0]

    for _, corner in circuit_info.corners.iterrows():
        txt = f"{corner['Number']}{corner['Letter']}"
        offset_angle = corner["Angle"] / 180 * np.pi
        offset_x, offset_y = rotate(offset_vector, angle=offset_angle)
        text_x = corner["X"] + offset_x
        text_y = corner["Y"] + offset_y
        text_x, text_y = rotate([text_x, text_y], angle=track_angle)
        track_x, track_y = rotate([corner["X"], corner["Y"]], angle=track_angle)
        if corner["Number"] == 1:
            plt.scatter(text_x, text_y, color="grey", s=140, label="Corners")
        else:
            plt.scatter(text_x, text_y, color="grey", s=140)
        plt.plot([track_x, text_x], [track_y, text_y], color="grey")
        plt.text(
            text_x,
            text_y,
            txt,
            va="center_baseline",
            ha="center",
            size="small",
            color="white",
        )

    suptitle = f"{Year} {EventName} Grand Prix Circuit"

    plt.suptitle(
        suptitle,
        fontweight="bold",
        fontsize=16,
    )

    ax.set_xlabel("X  Location (m)", fontweight="bold", fontsize=14)
    ax.set_ylabel("Y  Location (m)", fontweight="bold", fontsize=14)

    plt.axis("equal")

    x_max = plt.xlim()[1]
    y_min = plt.ylim()[0]

    ax.legend(loc="upper right")

    scale_length = 1000
    scale_text = f"{scale_length} m"

    scale_pos_x = x_max - 2 * scale_length
    scale_pos_y = y_min - 300

    plt.plot(
        [scale_pos_x, scale_pos_x + scale_length],
        [scale_pos_y, scale_pos_y],
        color="white",
        lw=1.7,
    )
    plt.plot(
        [scale_pos_x, scale_pos_x],
        [scale_pos_y, scale_pos_y + 100],
        color="white",
        lw=1.7,
    )
    plt.plot(
        [scale_pos_x + scale_length, scale_pos_x + scale_length],
        [scale_pos_y, scale_pos_y + 100],
        color="white",
        lw=1.7,
    )
    plt.text(
        scale_pos_x + scale_length / 2,
        scale_pos_y + 250,
        scale_text,
        ha="center",
        fontsize=10,
    )

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

    plt.tight_layout()

    filename = "../pic/" + suptitle.replace(" ", "_") + ".png"

    titles_str = (
        suptitle.replace(f"{Year} ", "")
        .replace(f"{EventName} ", "")
        .replace("Grand Prix ", "")
    )

    plt.savefig(filename)

    caption = textwrap.dedent(
        f"""\
🏎️
« {Year} {EventName} Grand Prix »

• Circuit: {circuitName}
• Country: {country}
• Locality: {locality}

#F1 #Formula1 #{EventName.replace(" ", "")}GP"""
    )

    return {"filename": filename, "caption": caption, "post": post}
