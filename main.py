import os
import textwrap
import time
from matplotlib import pyplot as plt
import fastf1
import fastf1.plotting

from auto_ig_post import auto_ig_post
from plot_functions import *

# Parameters
YEAR = 2024
EVENT_NAME = "Belgian"
SESSION_NAME = "R"

FUNC_PARAMS = {
    # Free Practice
    "plot_track_with_annotated_corners": {"enabled": False, "session": "FP1"},
    # Qualify
    "annotated_qualifying_flying_lap": {"enabled": False, "session": "Q"},
    # Race
    "driver_laptimes_distribution": {"enabled": False, "session": "R"},
    "team_pace_ranking": {"enabled": False, "session": "R"},
    "driver_laptimes_scatterplot": {"enabled": False, "session": "R"},
    "annotated_race_fatest_lap": {"enabled": False, "session": "R"},
    "race_fatest_lap_telemetry_data": {"enabled": False, "session": "R"},
    # Sprint Qualify
    "annotated_sprint_qualifying_flying_lap": {"enabled": False, "session": "SQ"},
}

FOLDER_PATH = "../Pic"
BLOCK = all(not value["enabled"] for value in FUNC_PARAMS.values())
POST_IG_DICT = {}


def get_event_names(year: int) -> None:
    """Get event names in a specific year."""
    event_names = fastf1.get_event_schedule(year)["EventName"]
    print(event_names)


def get_png_files(folder_path: str) -> list:
    """Get png files in the folder."""
    files = os.listdir(folder_path)
    return [file for file in files if file.endswith(".png")]


def organize_png_files_name() -> None:
    """Organize all png file names generated in this event."""
    png_files = get_png_files(FOLDER_PATH)
    titles = []

    event_name_formatted = EVENT_NAME.replace(" ", "_")

    for png_file in png_files:
        if f"{YEAR}" in png_file and f"{event_name_formatted}" in png_file:
            title = (
                png_file.replace(f"{YEAR}_", "")
                .replace(f"{event_name_formatted}_", "")
                .replace("Grand_Prix_", "")
                .replace(".png", "")
                .replace("_", " ")
            )
            titles.append(title)

    titles_str = "\n• ".join(titles)
    caption = textwrap.dedent(
        f"""\
        🏎️
        « {YEAR} {EVENT_NAME} Grand Prix »

        • {titles_str}

        #F1 #Formula1 #{EVENT_NAME.replace(" ", "")}GP"""
    )

    output_file_path = f"{FOLDER_PATH}/{YEAR}_{EVENT_NAME}_images.txt"
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(caption)


def post_ig() -> None:
    """Post images on Instagram."""
    for key, value in POST_IG_DICT.items():
        if value["post"]:
            auto_ig_post(value["filename"], value["caption"])
            time.sleep(60)


def plot_image_and_post_ig(key: str, race) -> None:
    """Plot images and post on Instagram."""
    POST_IG_DICT[key] = globals()[key](
        YEAR,
        EVENT_NAME,
        SESSION_NAME,
        race,
        FUNC_PARAMS[key]["enabled"],
    )


def plot_f1_data_analysis_images(block: bool) -> None:
    """Plot F1 data analysis images."""
    fastf1.plotting.setup_mpl(mpl_timedelta_support=False, misc_mpl_mods=False)
    fastf1.Cache.enable_cache("../cache")
    plt.ion()

    for key, params in FUNC_PARAMS.items():
        if params["session"] in SESSION_NAME:
            race = fastf1.get_session(YEAR, EVENT_NAME, params["session"])
            plot_image_and_post_ig(key, race)

    save_captions_to_file()
    plt.ioff()
    plt.show(block=block)
    if not block:
        plt.close("all")


def save_captions_to_file() -> None:
    """Save Instagram captions to file."""
    for key, value in POST_IG_DICT.items():
        output_file_path = f"{FOLDER_PATH}/{value['filename']}_ig.txt"
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(f"{value['caption']}\n")


if __name__ == "__main__":
    get_event_names(YEAR)
    plot_f1_data_analysis_images(BLOCK)
    post_ig()
    organize_png_files_name()
