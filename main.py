import os
import textwrap
import time
from matplotlib import pyplot as plt
import fastf1
import fastf1.plotting
import customtkinter as ctk
from tkinter import messagebox

from auto_ig_post import auto_ig_post
from plot_functions import *

YEAR = 2025
SESSION_NAME = "FP1+Q+R"
ENABLE_ALL = False

ALL_EVENT_OPTIONS = [
    "Bahrain",
    "Saudi Arabian",
    "Australian",
    "Japanese",
    "Chinese",
    "Miami",
    "Emilia Romagna",
    "Monaco",
    "Canadian",
    "Spanish",
    "Austrian",
    "British",
    "Hungarian",
    "Belgian",
    "Dutch",
    "Italian",
    "Azerbaijan",
    "Singapore",
    "United States",
    "Mexico",
    "São Paulo",
    "Las Vegas",
    "Qatar",
    "Abu Dhabi",
]

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


def select_event_name(options: list) -> str:
    """Allows the user to select an event name from a GUI list using customtkinter."""
    selected_event = None

    root = ctk.CTk()
    root.title("Select F1 Event")

    window_width = 350
    window_height = 800

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int((screen_width - window_width) / 2)
    center_y = int((screen_height - window_height) / 2)
    root.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")
    root.resizable(False, False)

    font_label = ("Times New Roman", 14)
    font_button_list = ("Times New Roman", 13)
    font_select_button = ("Times New Roman", 14, "bold")
    button_corner_radius = 6

    label = ctk.CTkLabel(
        root,
        text="Please select an F1 event:",
        font=font_label,
    )
    label.pack(pady=(10, 5), padx=20, anchor="w")

    scrollable_frame = ctk.CTkScrollableFrame(root, width=280, height=500)
    scrollable_frame.pack(pady=10, padx=20, fill="both", expand=True)

    event_buttons = []

    def create_event_button_callback(event_name, current_button):
        def callback():
            nonlocal selected_event
            selected_event = event_name
            for btn in event_buttons:
                btn.configure(
                    fg_color=(
                        ctk.ThemeManager.theme["CTkButton"]["hover_color"]
                        if btn == current_button
                        else ctk.ThemeManager.theme["CTkButton"]["fg_color"]
                    )
                )

        return callback

    for option in options:
        button = ctk.CTkButton(
            scrollable_frame,
            text=option,
            font=font_button_list,
            corner_radius=button_corner_radius,
            anchor="w",
        )
        button.configure(command=create_event_button_callback(option, button))
        button.pack(pady=3, padx=5, fill="x")
        event_buttons.append(button)

    def on_select_confirm():
        if selected_event:
            root.destroy()
        else:
            messagebox.showwarning(
                "No Selection", "Please select an event from the list."
            )

    def on_closing():
        if selected_event is None:
            print("No event selected, exiting.")
            exit()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    select_button = ctk.CTkButton(
        root,
        text="Select Event",
        command=on_select_confirm,
        font=font_select_button,
        corner_radius=button_corner_radius,
    )
    select_button.pack(pady=(10, 10))

    root.mainloop()

    if selected_event is None:
        print("No event was selected. Exiting application.")
        exit()
    return selected_event


EVENT_NAME = select_event_name(ALL_EVENT_OPTIONS)
print(f"Selected Event: {EVENT_NAME}")

FUNC_PARAMS = {
    # FP1
    "plot_track_with_annotated_corners": {"enabled": False, "session": "FP1"},
    # Q
    "annotated_qualifying_flying_lap": {"enabled": False, "session": "Q"},
    # R
    "annotated_race_fatest_lap": {"enabled": False, "session": "R"},
    "driver_laptimes_distribution": {"enabled": False, "session": "R"},
    "driver_laptimes_scatterplot": {"enabled": False, "session": "R"},
    "race_fatest_lap_telemetry_data": {"enabled": False, "session": "R"},
    "team_pace_ranking": {"enabled": False, "session": "R"},
    "driver_fuel_corrected_laptimes_scatterplot": {"enabled": False, "session": "R"},
    # SQ
    "annotated_sprint_qualifying_flying_lap": {"enabled": False, "session": "SQ"},
}
FOLDER_PATH = "../Pic"
BLOCK = not ENABLE_ALL and all(not value["enabled"] for value in FUNC_PARAMS.values())
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

    titles_str = "\n• ".join(titles) if titles else "No specific data generated"
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
        ENABLE_ALL or FUNC_PARAMS[key]["enabled"],
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
