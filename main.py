"""F1 Data Analysis application."""

import os
import textwrap
from pathlib import Path
from datetime import datetime, timezone
from openf1_data import ArchiveError, OpenF1Archive
import customtkinter as ctk
from tkinter import messagebox

from config import Config, get_config
from logger_config import setup_logging, get_logger
from plot_functions.plot_runner import PlotRunner
from auto_ig_post import auto_ig_post
from retry_utils import retry_on_network_error

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


class F1AnalysisApp(ctk.CTk):
    """Modern unified F1 Analysis configuration app."""

    def __init__(self, config=None):
        super().__init__()
        config = config or get_config("config.json")
        self.archive = OpenF1Archive(config.cache_path)
        self.plot_sessions = {plot.session for plot in config.plot_functions.values()}
        self.calendar = {}
        self.session_boxes = {}
        self.title("F1 Data Analysis")
        self.configure(fg_color=COLORS["bg_dark"])

        window_width, window_height = 520, 540
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        x = (screen_w - window_width) // 2
        y = (screen_h - window_height) // 2
        self.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.resizable(False, False)

        self.selected_year = ctk.StringVar(value=str(datetime.now(timezone.utc).year))
        self.selected_event = ctk.StringVar(value="")
        self.session_vars = {}
        self.instagram_var = ctk.BooleanVar(value=False)
        self.instagram_region_var = ctk.StringVar(value="UK")
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

        ctk.CTkOptionMenu(
            year_frame,
            values=[str(year) for year in range(int(self.selected_year.get()), 2022, -1)],
            variable=self.selected_year,
            command=self._on_year_change,
        ).pack(side="left")

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
            self.session_boxes[code] = cb
            cb.configure(state="disabled")
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

        ctk.CTkLabel(
            parent,
            text="IG Region",
            font=("Segoe UI", 12),
            text_color=COLORS["text_secondary"],
        ).pack(padx=10, pady=(6, 2), anchor="w")
        ctk.CTkOptionMenu(
            parent,
            values=["Taiwan", "UK"],
            variable=self.instagram_region_var,
            font=("Segoe UI", 12),
            fg_color=COLORS["bg_hover"],
            button_color=COLORS["accent"],
            button_hover_color=COLORS["accent_hover"],
        ).pack(padx=10, pady=(0, 4), anchor="w")

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
        self.update_idletasks()
        self.calendar = {}
        try:
            self.calendar = self.archive.calendar(year)
        except (ArchiveError, OSError, ValueError) as exc:
            logger.error("Historical calendar unavailable: %s", exc)
            self.status_label.configure(text="Archive unavailable")
            messagebox.showerror("OpenF1 archive", str(exc))
            return
        events = sorted(self.calendar)

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

        self.status_label.configure(text="" if events else "No completed sessions")

    def _on_year_change(self, year: str):
        """Handle year button click."""
        self.selected_year.set(year)
        self.selected_event.set("")

        for box in self.session_boxes.values():
            box.configure(state="disabled")
        self._load_events()

    def _on_event_select(self, event_name: str):
        """Handle event selection."""
        self.selected_event.set(event_name)
        available = self.calendar[event_name][1].keys() & self.plot_sessions
        for code, box in self.session_boxes.items():
            box.configure(state="normal" if code in available else "disabled")
            if code not in available:
                self.session_vars[code].set(False)
        if available and not any(var.get() for var in self.session_vars.values()):
            preferred = next(code for code in ("R", "Q", "SQ", "S", "FP3", "FP2", "FP1") if code in available)
            self.session_vars[preferred].set(True)

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
            "instagram_region": self.instagram_region_var.get(),
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

    app.archive.http.close()
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
    image_paths: list[str] | None = None,
) -> None:
    """Organize and create caption file for generated PNG files.

    Args:
        event_name: Grand Prix name
        year: Season year
        folder_path: Path to image folder
        config: Configuration object
    """
    try:
        if image_paths is None:
            prefix = f"{year}_{event_name.removesuffix(' Grand Prix').replace(' ', '_')}_"
            image_paths = [name for name in get_png_files(folder_path) if name.startswith(prefix)]
        titles = [chart_title(year, event_name, path) for path in image_paths]

        titles_str = "\n• ".join(titles) if titles else "No specific data generated"
        caption = textwrap.dedent(f"""\
            🏎️
            « {year} {event_name} »

            • {titles_str}

            #F1 #Formula1 #{event_name.replace(" ", "")}GP""")

        output_file_path = os.path.join(folder_path, f"{year}_{event_name}_images.txt")
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(caption)

        logger.info(f"Caption file created: {output_file_path}")
    except Exception as e:
        logger.error(f"Failed to organize PNG files: {str(e)}")


def chart_title(year: int, event_name: str, path: str) -> str:
    title = Path(path).stem.replace("_", " ").removeprefix(f"{year} ")
    title = title.removeprefix(event_name.removesuffix(" Grand Prix") + " ")
    return title.removeprefix("Grand Prix ").removeprefix("GP ").strip()


def create_event_caption(year: int, event_name: str, image_paths: list[str]) -> str:
    """Create a short Instagram caption with the carousel chart names."""
    chart_names = [chart_title(year, event_name, path) for path in image_paths]
    lines = [f"{year} {event_name}"]
    lines.extend(f"• {name}" for name in chart_names)
    return "\n".join(lines)


def post_to_instagram(all_posts: dict, config: Config, event_name: str) -> None:
    """Publish every successful chart as one Instagram carousel post.

    Args:
        all_posts: Results grouped by session
        config: Configuration object
        event_name: Grand Prix name for the combined caption
    """
    if not config.instagram_enabled:
        logger.info("Instagram posting is disabled")
        return

    try:
        image_paths = []
        omitted_names = []
        for posts in all_posts.values():
            for plot_name, value in posts.items():
                filename = value.get("filename")
                if value.get("success", True) and filename:
                    image_paths.append(filename)
                elif value.get("success", True):
                    omitted_names.append(plot_name)

        # The FP1 circuit map is the cover slide when it was generated.
        circuit_path = next(
            (
                path
                for path in image_paths
                if Path(path).stem.lower().endswith("_circuit")
            ),
            None,
        )
        if circuit_path:
            image_paths.remove(circuit_path)
            image_paths.insert(0, circuit_path)

        logger.info("Instagram carousel images (%d): %s", len(image_paths), image_paths)
        if omitted_names:
            logger.warning(
                "Instagram omitted plots without usable files: %s", omitted_names
            )

        if not image_paths:
            logger.warning("No successful charts available for Instagram posting")
            return

        caption = create_event_caption(config.year, event_name, image_paths)
        auto_ig_post(image_paths, caption, config.instagram_region)
        logger.info(
            "Grand Prix carousel posted successfully with %d chart(s)", len(image_paths)
        )
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
    try:
        return runner.run_all(config.year, event_name, config.session_name, config.instagram_enabled)
    finally:
        runner.cleanup()


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
        config.instagram_region = user_config["instagram_region"]

        event_name = user_config["event"]

        all_posts = {}
        analysis_sessions = list(user_config["sessions"])
        logger.info("Sessions selected in GUI: %s", analysis_sessions)

        for session in analysis_sessions:
            config.session_name = session
            logger.info("=" * 60)
            logger.info(f"Starting analysis for Session: {session}")
            logger.info("=" * 60)

            post_ig_dict = run_analysis(config, event_name)
            if post_ig_dict:
                all_posts[session] = post_ig_dict

        if config.instagram_enabled and all_posts:
            post_to_instagram(all_posts, config, event_name)

        images = [item["filename"] for plots in all_posts.values() for item in plots.values()
                  if item.get("filename") and item.get("success", True)]
        organize_png_files_name(event_name, config.year, config.folder_path, config, images)

        logger.info("=" * 60)
        logger.info("Analysis finished with %d chart(s).", len(images))
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
