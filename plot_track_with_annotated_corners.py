import matplotlib.pyplot as plt
import numpy as np
import fastf1


def rotate(xy, *, angle):
    rot_mat = np.array(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    )
    return np.matmul(xy, rot_mat)


# @brief plot_track_with_annotated_corners: Plot the track map with annotated corners
def plot_track_with_annotated_corners(
    Year: int, EventName: str, SessionName: str, race
):
    race.load()

    lap = race.laps.pick_fastest()
    pos = lap.get_pos_data()

    circuit_info = race.get_circuit_info()

    track = pos.loc[:, ("X", "Y")].to_numpy()

    track_angle = circuit_info.rotation / 180 * np.pi

    rotated_track = rotate(track, angle=track_angle)

    fig, ax = plt.subplots(figsize=(10.8, 10.8), dpi=100)

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
    plt.title(suptitle, fontweight="bold", fontsize=16)

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

    plt.tight_layout()

    filename = "../pic/" + suptitle.replace(" ", "_") + ".png"
    plt.savefig(filename)