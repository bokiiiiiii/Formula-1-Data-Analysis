from matplotlib import pyplot as plt
import os, textwrap, time

import fastf1
import fastf1.plotting

from auto_ig_post import auto_ig_post
from driver_laptimes_distribution import driver_laptimes_distribution
from team_pace_ranking import team_pace_ranking
from annotated_qualifying_flying_lap import annotated_qualifying_flying_lap
from driver_laptimes_scatterplot import driver_laptimes_scatterplot
from plot_track_with_annotated_corners import plot_track_with_annotated_corners
from annotated_race_fatest_lap import annotated_race_fatest_lap
from race_fatest_lap_telemetry_data import race_fatest_lap_telemetry_data


Year: int = 2024
EventName: str = "Spanish"
SessionName: str = "R"
folder_path: str = "../Pic"
block: bool = True
post_ig_params: dict = {
    # FP1
    "plot_track_with_annotated_corners": False,
    # Q
    "annotated_qualifying_flying_lap": False,
    # R
    "driver_laptimes_distribution": False,
    "team_pace_ranking": False,
    "driver_laptimes_scatterplot": False,
    "annotated_race_fatest_lap": False,
    "race_fatest_lap_telemetry_data": False,
}
post_ig_dict: dict = {}


# @brief get_event_names: Get event names in specific year
# @param year: [in] year
def get_event_names(year: int) -> None:
    event_names = fastf1.get_event_schedule(year)["EventName"]
    print(event_names)


# @brief get_png_files: Get png files in the folder
# @param floder_path: [in] folder path
# @return: png files
def get_png_files(folder_path):

    files = os.listdir(folder_path)
    png_files = [file for file in files if file.endswith(".png")]

    return png_files


# @brief organize_png_files_name: Organize all png file names generated in this event
def organize_png_files_name() -> None:
    png_files = get_png_files(folder_path)
    titles = []

    EventName_ = EventName.replace(" ", "_")

    for png_file in png_files:

        if f"{Year}" in png_file and f"{EventName_}" in png_file:

            title = (
                png_file.replace(f"{Year}_", "")
                .replace(f"{EventName_}_", "")
                .replace("Grand_Prix_", "")
                .replace(".png", "")
                .replace("_", " ")
            )
            titles.append(title)

    titles_str = "\n• ".join(titles)
    caption = textwrap.dedent(
        f"""\
🏎️
« {Year} {EventName} Grand Prix »

• {titles_str}

#F1 #Formula1 #{EventName.replace(" ", "")}GP"""
    )

    output_file_path = f"../Pic/{Year}_{EventName}_images.txt"
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(caption)


# @brief post_ig: Post images on instagram
def post_ig() -> None:
    for key, value in post_ig_dict.items():
        if value["post"]:
            auto_ig_post(value["filename"], value["caption"])
            time.sleep(60)
            

# @brief plot_image_and_post_ig: Plot images and post on instagram
# @param key: [in] function name
# @param race: [in] race session       
def plot_image_and_post_ig(key, race):
    post_ig_dict[key] = globals()[key](
        Year,
        EventName,
        SessionName,
        race,
        post_ig_params.get(key, False),
    )
                

# @brief plot_f1_data_analysis_images: Plot F1 data analysis images
# @param block: [in] plt.show block or not
def plot_f1_data_analysis_images(block: bool) -> None:
    fastf1.plotting.setup_mpl(mpl_timedelta_support=False, misc_mpl_mods=False)
    fastf1.Cache.enable_cache("../cache")
    plt.ion()

    # Qualify
    if "FP1" in SessionName:
        race = fastf1.get_session(Year, EventName, "FP1")
        plot_image_and_post_ig("plot_track_with_annotated_corners", race)
                
    # Qualify
    if "Q" in SessionName:
        race = fastf1.get_session(Year, EventName, "Q")
        # plot_image_and_post_ig("plot_track_with_annotated_corners", race)
        plot_image_and_post_ig("annotated_qualifying_flying_lap", race)

    # Race
    if "R" in SessionName:
        race = fastf1.get_session(Year, EventName, "R")
        plot_image_and_post_ig("driver_laptimes_distribution", race)
        # plot_image_and_post_ig("team_pace_ranking", race) 
        plot_image_and_post_ig("driver_laptimes_scatterplot", race) 
        plot_image_and_post_ig("annotated_race_fatest_lap", race) 
        plot_image_and_post_ig("race_fatest_lap_telemetry_data", race)     
        
    for keys, values in post_ig_dict.items():
        # print(f"{keys}: {values}")
        output_file_path = f"../Pic/{values['filename']}_ig.txt"
        with open(output_file_path, "w", encoding="utf-8") as f:
            ig_caption = values["caption"]
            f.write(f"{ig_caption}\n")

    plt.ioff()
    plt.show(block=block)
    if not block:
        plt.close("all")


# @brief main: Plot F1 data analysis images
# @ref: https://github.com/theOehrly/Fast-F1
if __name__ == "__main__":

    # get_event_names(Year)

    plot_f1_data_analysis_images(block)

    post_ig()

    organize_png_files_name()
