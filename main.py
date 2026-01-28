"""F1 Data Analysis application."""

import os
import textwrap
import time
from pathlib import Path
from matplotlib import pyplot as plt
import fastf1
import customtkinter as ctk
from tkinter import messagebox

from config import Config, get_config
from logger_config import setup_logging, get_logger
from plot_functions import utils
from plot_functions import *
from auto_ig_post import InstagramPoster

from f1_analyzer import F1Analyzer
from retry_utils import retry_on_network_error
from performance_monitor import measure

setup_logging()
logger = get_logger(__name__)

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


@retry_on_network_error(max_attempts=3, delay=5.0)
def get_event_list_from_api(year: int) -> list:
    """Get event list from FastF1 API with retry logic.

    Args:
        year: Season year

    Returns:
        List of event names
    """
    try:
        logger.info(f"Fetching event list for {year} from FastF1 API...")
        analyzer = F1Analyzer(year, "Bahrain", "R")  # Temporary instance
        event_names = analyzer.get_event_names()
        logger.info(f"Found {len(event_names)} events")
        return event_names
    except Exception as e:
        logger.error(f"API fetch failed: {e}")
        logger.warning("Using fallback events")
        return [
            "Bahrain Grand Prix",
            "Saudi Arabian Grand Prix",
            "Australian Grand Prix",
            "Japanese Grand Prix",
            "Chinese Grand Prix",
            "Miami Grand Prix",
            "Emilia Romagna Grand Prix",
            "Monaco Grand Prix",
            "Canadian Grand Prix",
            "Spanish Grand Prix",
            "Austrian Grand Prix",
            "British Grand Prix",
            "Hungarian Grand Prix",
            "Belgian Grand Prix",
            "Dutch Grand Prix",
            "Italian Grand Prix",
            "Azerbaijan Grand Prix",
            "Singapore Grand Prix",
            "United States Grand Prix",
            "Mexico City Grand Prix",
            "São Paulo Grand Prix",
            "Las Vegas Grand Prix",
            "Qatar Grand Prix",
            "Abu Dhabi Grand Prix",
        ]


def select_event_name(year: int = None) -> str:
    """Select event from GUI list.

    Args:
        year: Optional year for dynamic loading

    Returns:
        Selected event name
    """
    if year is None:
        year = 2025

    with measure("get_event_list"):
        options = get_event_list_from_api(year)
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

    label = ctk.CTkLabel(root, text="Please select an F1 event:", font=font_label)
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
            root.withdraw()
            root.quit()
        else:
            messagebox.showwarning(
                "No Selection", "Please select an event from the list."
            )

    def on_closing():
        if selected_event is None:
            logger.info("No event selected, exiting.")
            exit()
        root.withdraw()
        root.quit()

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
        logger.error("No event was selected. Exiting application.")
        exit()

    return selected_event


def get_png_files(folder_path: str) -> list:
    """Get list of PNG files in folder.

    Args:
        folder_path: Path to folder

    Returns:
        List of PNG filenames
    """
    try:
        files = os.listdir(folder_path)
        return [file for file in files if file.endswith(".png")]
    except Exception as e:
        logger.warning(f"Failed to list PNG files: {str(e)}")
        return []


def organize_png_files_name(
    event_name: str,
    year: int,
    folder_path: str,
    config: Config,
) -> None:
    """Organize and create caption file for generated PNG files.

    Args:
        event_name: Grand Prix name
        year: Season year
        folder_path: Path to image folder
        config: Configuration object
    """
    try:
        png_files = get_png_files(folder_path)
        titles = []

        event_name_formatted = event_name.replace(" ", "_")

        for png_file in png_files:
            if f"{year}" in png_file and f"{event_name_formatted}" in png_file:
                title = (
                    png_file.replace(f"{year}_", "")
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
            « {year} {event_name} Grand Prix »

            • {titles_str}

            #F1 #Formula1 #{event_name.replace(" ", "")}GP"""
        )

        output_file_path = os.path.join(folder_path, f"{year}_{event_name}_images.txt")
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(caption)

        logger.info(f"Caption file created: {output_file_path}")
    except Exception as e:
        logger.error(f"Failed to organize PNG files: {str(e)}")


def post_to_instagram(post_dict: dict, config: Config) -> None:
    """Post images to Instagram if enabled.

    Args:
        post_dict: Dictionary with post information
        config: Configuration object
    """
    if not config.instagram_enabled:
        logger.info("Instagram posting is disabled")
        return

    try:
        poster = InstagramPoster()
        delay = config.instagram_delay_seconds

        for key, value in post_dict.items():
            if value.get("post") and value.get("success", True):
                try:
                    logger.info(f"Posting to Instagram: {key}")
                    poster.post(value["filename"], value["caption"])
                    logger.info(f"Posted successfully: {key}")
                    time.sleep(delay)
                except Exception as e:
                    logger.error(f"Failed to post {key}: {str(e)}")
    except ValueError as e:
        logger.warning(f"Instagram credentials not available: {str(e)}")
    except Exception as e:
        logger.error(f"Instagram posting error: {str(e)}")


def run_analysis(config: Config, event_name: str) -> dict:
    """Run the F1 data analysis pipeline.

    Args:
        config: Configuration object
        event_name: Grand Prix event name

    Returns:
        Dictionary with analysis results
    """
    logger.info("=" * 60)
    logger.info(f"Starting F1 Data Analysis")
    logger.info(f"Year: {config.year}, Session: {config.session_name}")
    logger.info("=" * 60)

    # Setup matplotlib
    utils.setup_matplotlib_style(
        utils.PlotConfig(
            dpi=config.figure_dpi,
            fig_width_inch=config.figure_width,
            fig_height_inch=config.figure_height,
        )
    )

    # Enable cache
    utils.enable_cache(config.cache_path)
    plt.ion()

    # Create results directory
    Path(config.folder_path).mkdir(parents=True, exist_ok=True)

    # Process plot functions
    post_ig_dict = {}
    processed_count = 0

    for func_name, func_config in config.plot_functions.items():
        # Check if should process
        if not config.should_process_plot_function(func_name):
            logger.debug(f"Skipping {func_name} (not in {config.session_name})")
            continue

        # Check if enabled
        if not (config.enable_all or func_config.enabled):
            logger.debug(f"Skipping {func_name} (disabled)")
            continue

        try:
            logger.info(f"Processing: {func_name}")

            plot_func = globals().get(func_name)

            if not plot_func:
                logger.warning(f"Function {func_name} not found")
                continue

            # Load race data
            race = fastf1.get_session(
                config.year,
                event_name,
                func_config.session,
            )

            # Call plot function
            result = plot_func(
                config.year,
                event_name,
                config.session_name,
                race,
                config.instagram_enabled,
            )

            post_ig_dict[func_name] = result
            processed_count += 1
            logger.info(f"Completed: {func_name}")

        except Exception as e:
            logger.error(f"Error processing {func_name}: {str(e)}", exc_info=True)

    logger.info(f"Processed {processed_count}/{len(config.plot_functions)} functions")

    plt.ioff()
    plt.show(block=False)

    return post_ig_dict


def select_config_gui() -> dict:
    """GUI for selecting analysis configuration.

    Returns:
        Dict with year, sessions, enable_all, instagram_enabled
    """
    selected_config = {
        "year": None,
        "sessions": [],
        "enable_all": True,
        "instagram_enabled": False,
    }

    root = ctk.CTk()
    root.title("F1 Analysis Configuration")

    window_width = 550
    window_height = 520
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int((screen_width - window_width) / 2)
    center_y = int((screen_height - window_height) / 2)
    root.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")
    root.resizable(False, False)

    # Year selection
    ctk.CTkLabel(root, text="Select Year:", font=("Times New Roman", 16, "bold")).pack(
        pady=(20, 5)
    )
    year_var = ctk.StringVar(value="2026")
    year_frame = ctk.CTkFrame(root)
    year_frame.pack(pady=5, padx=20, fill="x")

    years = ["2023", "2024", "2025", "2026"]
    year_buttons = []
    for year in years:
        btn = ctk.CTkButton(
            year_frame,
            text=year,
            width=120,
            command=lambda y=year: [
                year_var.set(y),
                [
                    b.configure(
                        fg_color=(
                            ("gray75", "gray25")
                            if b.cget("text") == y
                            else ("gray86", "gray17")
                        )
                    )
                    for b in year_buttons
                ],
            ],
        )
        btn.pack(side="left", padx=5)
        year_buttons.append(btn)
    year_buttons[3].configure(fg_color=("gray75", "gray25"))

    # Session selection
    ctk.CTkLabel(
        root, text="Select Sessions (Multiple):", font=("Times New Roman", 16, "bold")
    ).pack(pady=(20, 5))
    session_frame = ctk.CTkScrollableFrame(root, width=400, height=180)
    session_frame.pack(pady=5, padx=20)

    sessions = [
        ("FP1", "Free Practice 1"),
        ("FP2", "Free Practice 2"),
        ("FP3", "Free Practice 3"),
        ("Q", "Qualifying"),
        ("SQ", "Sprint Qualifying"),
        ("S", "Sprint"),
        ("R", "Race"),
    ]
    session_vars = {}
    for code, name in sessions:
        var = ctk.BooleanVar(value=(code == "R"))
        cb = ctk.CTkCheckBox(
            session_frame,
            text=f"{name} ({code})",
            variable=var,
            font=("Times New Roman", 14),
        )
        cb.pack(pady=3, padx=10, anchor="w")
        session_vars[code] = var

    # Instagram enabled
    instagram_var = ctk.BooleanVar(value=False)
    ctk.CTkCheckBox(
        root,
        text="Post to Instagram",
        variable=instagram_var,
        font=("Times New Roman", 14, "bold"),
    ).pack(pady=15)

    # Confirm button
    def on_confirm():
        selected_config["year"] = int(year_var.get())
        selected_config["sessions"] = [
            code for code, var in session_vars.items() if var.get()
        ]
        selected_config["enable_all"] = True
        selected_config["instagram_enabled"] = instagram_var.get()

        if not selected_config["sessions"]:
            messagebox.showwarning("Warning", "Please select at least one session")
            return

        root.withdraw()
        root.quit()

    def on_closing():
        if not selected_config["sessions"]:
            logger.error("No sessions selected, exiting")
            exit()
        root.withdraw()
        root.quit()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    ctk.CTkButton(
        root,
        text="Start Analysis",
        command=on_confirm,
        font=("Times New Roman", 16, "bold"),
        width=200,
        height=40,
    ).pack(pady=20)

    root.mainloop()

    if not selected_config["sessions"]:
        logger.error("No sessions selected, exiting")
        exit()

    return selected_config


def main():
    """Main entry point."""
    try:
        # GUI select configuration
        user_config = select_config_gui()
        logger.info(
            f"User config: Year={user_config['year']}, Sessions={user_config['sessions']}, Instagram={user_config['instagram_enabled']}"
        )

        # Load base config
        config = get_config("config.json")
        config.year = user_config["year"]
        config.enable_all = user_config["enable_all"]
        config.instagram_enabled = user_config["instagram_enabled"]

        # Select event once for all sessions
        event_name = select_event_name(config.year)
        logger.info(f"Selected event: {event_name}")

        # Run analysis for each selected session
        all_posts = {}
        for session in user_config["sessions"]:
            config.session_name = session
            logger.info("=" * 60)
            logger.info(f"Starting analysis for Session: {session}")
            logger.info("=" * 60)

            post_ig_dict = run_analysis(config, event_name)
            if post_ig_dict:
                all_posts[session] = post_ig_dict

        # Post to Instagram if enabled
        if config.instagram_enabled and all_posts:
            for session, posts in all_posts.items():
                logger.info(f"Uploading to Instagram - Session: {session}")
                post_to_instagram(posts, config)

        # Organize files
        organize_png_files_name(event_name, config.year, config.folder_path, config)

        logger.info("=" * 60)
        logger.info("All analysis completed!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
