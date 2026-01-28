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
from plot_functions.plot_runner import PlotRunner
from auto_ig_post import InstagramPoster
from retry_utils import retry_on_network_error
from performance_monitor import measure

setup_logging()
logger = get_logger(__name__)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

COLORS = {
    "bg_dark": "#0a0a0f",
    "bg_card": "#12121a",
    "bg_hover": "#1e1e2e",
    "accent": "#3b82f6",
    "accent_hover": "#60a5fa",
    "text_primary": "#e4e4e7",
    "text_secondary": "#6b7280",
    "border": "#27272a",
    "success": "#22c55e",
}


@retry_on_network_error(max_attempts=3, delay=5.0)
def get_event_list_from_api(year: int) -> list:
    """Get event list from FastF1 API with retry logic."""
    try:
        logger.info(f"Fetching event list for {year} from FastF1 API...")
        schedule = fastf1.get_event_schedule(year)
        event_names = schedule["EventName"].dropna().unique().tolist()
        logger.info(f"Found {len(event_names)} events")
        return sorted(event_names)
    except Exception as e:
        logger.error(f"API fetch failed: {e}")
        return [
            "Bahrain Grand Prix",
            "Saudi Arabian Grand Prix",
            "Australian Grand Prix",
            "Japanese Grand Prix",
            "Chinese Grand Prix",
            "Miami Grand Prix",
            "Monaco Grand Prix",
            "Canadian Grand Prix",
            "Spanish Grand Prix",
            "Austrian Grand Prix",
            "British Grand Prix",
            "Hungarian Grand Prix",
            "Belgian Grand Prix",
            "Dutch Grand Prix",
            "Italian Grand Prix",
            "Singapore Grand Prix",
            "United States Grand Prix",
            "Mexico City Grand Prix",
            "São Paulo Grand Prix",
            "Las Vegas Grand Prix",
            "Qatar Grand Prix",
            "Abu Dhabi Grand Prix",
        ]


class F1AnalysisApp(ctk.CTk):
    """Modern unified F1 Analysis configuration app."""

    def __init__(self):
        super().__init__()
        self.title("F1 Data Analysis")
        self.configure(fg_color=COLORS["bg_dark"])

        window_width, window_height = 520, 540
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        x = (screen_w - window_width) // 2
        y = (screen_h - window_height) // 2
        self.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.resizable(False, False)

        self.selected_year = ctk.StringVar(value="2025")
        self.selected_event = ctk.StringVar(value="")
        self.session_vars = {}
        self.instagram_var = ctk.BooleanVar(value=False)
        self.result = None
        self.event_buttons = []

        self._build_ui()
        self._load_events()

    def _build_ui(self):
        """Build the main UI layout."""
        header = ctk.CTkFrame(self, fg_color=COLORS["bg_card"], height=50)
        header.pack(fill="x", padx=0, pady=0)
        header.pack_propagate(False)

        ctk.CTkLabel(
            header,
            text="F1 Data Analysis",
            font=("Segoe UI", 18, "bold"),
            text_color=COLORS["text_primary"],
        ).pack(side="left", padx=16, pady=12)

        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=12, pady=10)

        left_panel = ctk.CTkFrame(content, fg_color=COLORS["bg_card"], corner_radius=8)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 6))

        self._build_event_panel(left_panel)

        right_panel = ctk.CTkFrame(
            content, fg_color=COLORS["bg_card"], corner_radius=8, width=140
        )
        right_panel.pack(side="right", fill="y", padx=(6, 0))
        right_panel.pack_propagate(False)

        self._build_config_panel(right_panel)

        footer = ctk.CTkFrame(self, fg_color="transparent", height=50)
        footer.pack(fill="x", padx=12, pady=(0, 10))

        self.start_btn = ctk.CTkButton(
            footer,
            text="START",
            font=("Segoe UI", 14, "bold"),
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            height=40,
            corner_radius=6,
            command=self._on_start,
        )
        self.start_btn.pack(fill="x")

    def _build_event_panel(self, parent):
        """Build the event selection panel."""
        year_frame = ctk.CTkFrame(parent, fg_color="transparent")
        year_frame.pack(fill="x", padx=10, pady=(10, 6))

        for year in ["2023", "2024", "2025", "2026"]:
            btn = ctk.CTkButton(
                year_frame,
                text=year,
                width=52,
                height=28,
                font=("Segoe UI", 12),
                fg_color=(COLORS["accent"] if year == "2025" else COLORS["bg_hover"]),
                hover_color=COLORS["accent_hover"],
                corner_radius=6,
                command=lambda y=year: self._on_year_change(y),
            )
            btn.pack(side="left", padx=2)
            setattr(self, f"year_btn_{year}", btn)

        self.event_frame = ctk.CTkScrollableFrame(
            parent,
            fg_color="transparent",
            scrollbar_button_color=COLORS["border"],
            scrollbar_button_hover_color=COLORS["text_secondary"],
        )
        self.event_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def _build_config_panel(self, parent):
        """Build the configuration panel."""
        sessions = [
            ("FP1", "FP1"),
            ("FP2", "FP2"),
            ("FP3", "FP3"),
            ("Q", "Quali"),
            ("SQ", "Sprint Q"),
            ("S", "Sprint"),
            ("R", "Race"),
        ]

        for code, name in sessions:
            var = ctk.BooleanVar(value=(code == "R"))
            self.session_vars[code] = var

            cb = ctk.CTkCheckBox(
                parent,
                text=name,
                variable=var,
                font=("Segoe UI", 13),
                text_color=COLORS["text_primary"],
                fg_color=COLORS["accent"],
                hover_color=COLORS["accent_hover"],
                border_color=COLORS["border"],
                checkmark_color=COLORS["text_primary"],
                corner_radius=3,
                checkbox_width=20,
                checkbox_height=20,
            )
            cb.pack(padx=10, pady=4, anchor="w")

        ctk.CTkFrame(parent, fg_color=COLORS["border"], height=1).pack(
            fill="x", padx=10, pady=10
        )

        cb_ig = ctk.CTkCheckBox(
            parent,
            text="IG Post",
            variable=self.instagram_var,
            font=("Segoe UI", 13),
            text_color=COLORS["text_primary"],
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            border_color=COLORS["border"],
            checkmark_color=COLORS["text_primary"],
            corner_radius=3,
            checkbox_width=20,
            checkbox_height=20,
        )
        cb_ig.pack(padx=10, pady=4, anchor="w")

        self.status_frame = ctk.CTkFrame(parent, fg_color="transparent")
        self.status_frame.pack(fill="x", padx=10, pady=(10, 8), side="bottom")

        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="",
            font=("Segoe UI", 10),
            text_color=COLORS["text_secondary"],
        )
        self.status_label.pack()

    def _load_events(self):
        """Load events for the selected year."""
        year = int(self.selected_year.get())

        for btn in self.event_buttons:
            btn.destroy()
        self.event_buttons.clear()

        self.status_label.configure(text="Loading...")
        self.update()

        events = get_event_list_from_api(year)

        for event in events:
            btn = ctk.CTkButton(
                self.event_frame,
                text=event,
                font=("Segoe UI", 13),
                fg_color="transparent",
                hover_color=COLORS["bg_hover"],
                text_color=COLORS["text_primary"],
                anchor="w",
                height=34,
                corner_radius=4,
                command=lambda e=event: self._on_event_select(e),
            )
            btn.pack(fill="x", pady=1)
            self.event_buttons.append(btn)

        self.status_label.configure(text="")

    def _on_year_change(self, year: str):
        """Handle year button click."""
        self.selected_year.set(year)
        self.selected_event.set("")

        for y in ["2023", "2024", "2025", "2026"]:
            btn = getattr(self, f"year_btn_{y}")
            btn.configure(
                fg_color=COLORS["accent"] if y == year else COLORS["bg_hover"]
            )

        self._load_events()

    def _on_event_select(self, event_name: str):
        """Handle event selection."""
        self.selected_event.set(event_name)

        for btn in self.event_buttons:
            is_selected = btn.cget("text") == event_name
            btn.configure(
                fg_color=COLORS["accent"] if is_selected else "transparent",
                text_color=COLORS["text_primary"],
            )

        short_name = event_name.replace(" Grand Prix", "")
        self.status_label.configure(
            text=f"✓ {short_name}", text_color=COLORS["success"]
        )

    def _on_start(self):
        """Handle start button click."""
        if not self.selected_event.get():
            messagebox.showwarning("No Event", "Please select an event first.")
            return

        sessions = [code for code, var in self.session_vars.items() if var.get()]
        if not sessions:
            messagebox.showwarning("No Sessions", "Please select at least one session.")
            return

        self.result = {
            "year": int(self.selected_year.get()),
            "event": self.selected_event.get(),
            "sessions": sessions,
            "instagram_enabled": self.instagram_var.get(),
        }

        self.withdraw()
        self.quit()

    def get_result(self):
        """Get the configuration result."""
        return self.result


def run_config_gui() -> dict:
    """Run the unified configuration GUI.

    Returns:
        Dict with year, event, sessions, instagram_enabled
    """
    app = F1AnalysisApp()
    app.protocol("WM_DELETE_WINDOW", lambda: (logger.info("Cancelled"), exit()))
    app.mainloop()

    result = app.get_result()
    if result is None:
        logger.error("No configuration selected")
        exit()

    app.destroy()
    return result


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


@retry_on_network_error(max_attempts=3, delay=5.0)
def run_analysis(config: Config, event_name: str) -> dict:
    """Run F1 data analysis using PlotRunner.

    Args:
        config: Configuration object
        event_name: Grand Prix event name

    Returns:
        Dictionary with analysis results
    """
    runner = PlotRunner(config)
    results = runner.run_all(
        config.year, event_name, config.session_name, config.instagram_enabled
    )
    runner.cleanup()
    return results


def main():
    """Main entry point."""
    try:
        user_config = run_config_gui()
        logger.info(
            f"User config: Year={user_config['year']}, Event={user_config['event']}, "
            f"Sessions={user_config['sessions']}, Instagram={user_config['instagram_enabled']}"
        )

        config = get_config("config.json")
        config.year = user_config["year"]
        config.enable_all = True
        config.instagram_enabled = user_config["instagram_enabled"]

        event_name = user_config["event"]

        all_posts = {}
        for session in user_config["sessions"]:
            config.session_name = session
            logger.info("=" * 60)
            logger.info(f"Starting analysis for Session: {session}")
            logger.info("=" * 60)

            post_ig_dict = run_analysis(config, event_name)
            if post_ig_dict:
                all_posts[session] = post_ig_dict

        if config.instagram_enabled and all_posts:
            for session, posts in all_posts.items():
                logger.info(f"Uploading to Instagram - Session: {session}")
                post_to_instagram(posts, config)

        organize_png_files_name(event_name, config.year, config.folder_path, config)

        logger.info("=" * 60)
        logger.info("All analysis completed!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
