from matplotlib import pyplot as plt
import os, textwrap

import fastf1
import fastf1.plotting

from driver_laptimes_distribution import driver_laptimes_distribution
from team_pace_ranking import team_pace_ranking
from annotated_qualifying_flying_lap import annotated_qualifying_flying_lap
from driver_laptimes_scatterplot import driver_laptimes_scatterplot


Year: int = 2024
EventName: str = "Emilia"
folder_path = "../Pic"


# @brief get_png_files: Get png files in the folder
# @param floder_path: [in] folder path
# @return: png files
def get_png_files(folder_path):

    files = os.listdir(folder_path)
    png_files = [file for file in files if file.endswith(".png")]

    return png_files


# @brief main: Plot F1 data analysis results
# @ref: https://github.com/theOehrly/Fast-F1
if __name__ == "__main__":

    fastf1.plotting.setup_mpl(mpl_timedelta_support=False, misc_mpl_mods=False)

    plt.ion()

    # Race
    SessionName: str = "R"
    race = fastf1.get_session(Year, EventName, SessionName)
    driver_laptimes_distribution(Year, EventName, SessionName, race)
    team_pace_ranking(Year, EventName, SessionName, race)
    driver_laptimes_scatterplot(Year, EventName, SessionName, race)

    # Qualify
    SessionName: str = "Q"
    race = fastf1.get_session(Year, EventName, SessionName)
    annotated_qualifying_flying_lap(Year, EventName, SessionName, race)

    plt.ioff()
    plt.show(block=True)

    png_files = get_png_files(folder_path)
    titles = []

    for png_file in png_files:

        if f"{Year}" in png_file and f"{EventName}" in png_file:

            title = (
                png_file.replace(f"{Year}_", "")
                .replace(f"{EventName}_", "")
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

#formula1"""
    )

    output_file_path = f"../Pic/{Year}_{EventName}_caption.txt"
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(caption)
