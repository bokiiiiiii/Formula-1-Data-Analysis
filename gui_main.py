"""Professional GUI for F1 Data Analysis."""

import threading
from typing import List, Optional, Dict, Callable
from pathlib import Path
import customtkinter as ctk
from tkinter import messagebox, scrolledtext
import yaml

from logger_config import get_logger
from f1_analyzer import get_or_create_analyzer, F1Analyzer
from progress_manager import ProgressTracker, ProgressPhase, ProgressLogger
from parallel_executor import ParallelPlotExecutor, PlotTask

logger = get_logger(__name__)

# Configure CustomTkinter
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class F1AnalysisGUI:
    """Main GUI window for F1 Data Analysis."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize GUI.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

        self.root = ctk.CTk()
        self.root.title("F1 Data Analysis Suite")
        self.root.geometry("1000x800")

        self.analyzer: Optional[F1Analyzer] = None
        self.progress_tracker: Optional[ProgressTracker] = None
        self.executor: Optional[ParallelPlotExecutor] = None

        self._setup_ui()
        logger.info("GUI initialized")

    def _load_config(self) -> Dict:
        """Load configuration from YAML file.

        Returns:
            Configuration dictionary
        """
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return {}

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}

    def _setup_ui(self) -> None:
        """Setup the GUI layout."""

        # Main container with tabs
        self.tabview = ctk.CTkTabview(self.root)
        self.tabview.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        # Create tabs
        self.tab_selection = self.tabview.add("Analysis Settings")
        self.tab_progress = self.tabview.add("Progress")
        self.tab_logs = self.tabview.add("Logs")

        self._setup_selection_tab()
        self._setup_progress_tab()
        self._setup_logs_tab()

        # Control buttons on right
        self._setup_control_panel()

    def _setup_selection_tab(self) -> None:
        """Setup the selection/configuration tab."""

        # Year selection
        year_frame = ctk.CTkFrame(self.tab_selection)
        year_frame.pack(padx=10, pady=5, fill="x")

        ctk.CTkLabel(year_frame, text="Year:", font=("Arial", 12, "bold")).pack(
            side="left", padx=5
        )
        self.year_var = ctk.StringVar(value="2025")
        year_spinbox = ctk.CTkSpinbox(
            year_frame,
            from_=2018,
            to=2026,
            textvariable=self.year_var,
            width=100,
        )
        year_spinbox.pack(side="left", padx=5)

        # Event selection
        event_frame = ctk.CTkFrame(self.tab_selection)
        event_frame.pack(padx=10, pady=5, fill="x")

        ctk.CTkLabel(event_frame, text="Event:", font=("Arial", 12, "bold")).pack(
            side="left", padx=5
        )
        self.event_var = ctk.StringVar(value="Abu Dhabi")

        self.event_dropdown = ctk.CTkComboBox(
            event_frame,
            variable=self.event_var,
            values=[],
            width=200,
            command=self._on_event_selected,
        )
        self.event_dropdown.pack(side="left", padx=5)

        # Refresh events button
        ctk.CTkButton(
            event_frame,
            text="Refresh Events",
            width=100,
            command=self._refresh_events,
        ).pack(side="left", padx=5)

        # Session selection
        session_frame = ctk.CTkFrame(self.tab_selection)
        session_frame.pack(padx=10, pady=5, fill="x")

        ctk.CTkLabel(session_frame, text="Session:", font=("Arial", 12, "bold")).pack(
            side="left", padx=5
        )
        self.session_var = ctk.StringVar(value="R")
        session_dropdown = ctk.CTkComboBox(
            session_frame,
            variable=self.session_var,
            values=["FP1", "FP2", "FP3", "Q", "SS", "S", "R"],
            width=100,
        )
        session_dropdown.pack(side="left", padx=5)

        # Driver selection
        driver_frame = ctk.CTkFrame(self.tab_selection)
        driver_frame.pack(padx=10, pady=5, fill="both", expand=True)

        ctk.CTkLabel(driver_frame, text="Drivers:", font=("Arial", 12, "bold")).pack(
            anchor="w", padx=5, pady=5
        )

        # Scrollable driver list
        self.driver_listbox = ctk.CTkScrollableFrame(
            driver_frame, width=250, height=150
        )
        self.driver_listbox.pack(padx=5, pady=5, fill="both", expand=True)

        self.driver_checkboxes: Dict[str, ctk.CTkCheckBox] = {}

        # Plot function selection
        plot_frame = ctk.CTkFrame(self.tab_selection)
        plot_frame.pack(padx=10, pady=5, fill="both", expand=True)

        ctk.CTkLabel(
            plot_frame, text="Plot Functions:", font=("Arial", 12, "bold")
        ).pack(anchor="w", padx=5, pady=5)

        # Scrollable plot list
        self.plot_listbox = ctk.CTkScrollableFrame(plot_frame, width=250, height=200)
        self.plot_listbox.pack(padx=5, pady=5, fill="both", expand=True)

        self.plot_checkboxes: Dict[str, ctk.CTkCheckBox] = {}
        self._populate_plot_functions()

    def _setup_progress_tab(self) -> None:
        """Setup the progress tab."""

        # Progress label
        self.progress_label = ctk.CTkLabel(
            self.tab_progress,
            text="Ready",
            font=("Arial", 12),
        )
        self.progress_label.pack(pady=10, padx=10, fill="x")

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(self.tab_progress, height=25)
        self.progress_bar.pack(pady=10, padx=10, fill="x")
        self.progress_bar.set(0)

        # Details text
        self.progress_text = scrolledtext.ScrolledText(
            self.tab_progress,
            height=25,
            width=80,
            font=("Courier", 9),
        )
        self.progress_text.pack(pady=10, padx=10, fill="both", expand=True)

    def _setup_logs_tab(self) -> None:
        """Setup the logs tab."""

        self.logs_text = scrolledtext.ScrolledText(
            self.tab_logs,
            height=30,
            width=100,
            font=("Courier", 8),
        )
        self.logs_text.pack(pady=10, padx=10, fill="both", expand=True)

        # Clear logs button
        ctk.CTkButton(
            self.tab_logs,
            text="Clear Logs",
            width=100,
            command=lambda: self.logs_text.delete(1.0, "end"),
        ).pack(pady=5)

    def _setup_control_panel(self) -> None:
        """Setup control buttons panel."""

        control_frame = ctk.CTkFrame(self.root)
        control_frame.pack(side="right", fill="both", padx=10, pady=10)

        ctk.CTkLabel(control_frame, text="Controls", font=("Arial", 14, "bold")).pack(
            pady=10
        )

        # Load session button
        ctk.CTkButton(
            control_frame,
            text="Load Session",
            height=40,
            width=150,
            font=("Arial", 12, "bold"),
            command=self._load_session,
        ).pack(pady=5, padx=5, fill="x")

        # Start analysis button
        self.start_button = ctk.CTkButton(
            control_frame,
            text="Start Analysis",
            height=40,
            width=150,
            font=("Arial", 12, "bold"),
            fg_color="green",
            command=self._start_analysis,
        )
        self.start_button.pack(pady=5, padx=5, fill="x")

        # Stop button
        self.stop_button = ctk.CTkButton(
            control_frame,
            text="Stop Analysis",
            height=40,
            width=150,
            font=("Arial", 12, "bold"),
            fg_color="red",
            state="disabled",
            command=self._stop_analysis,
        )
        self.stop_button.pack(pady=5, padx=5, fill="x")

        # Settings button
        ctk.CTkButton(
            control_frame,
            text="Settings",
            height=35,
            width=150,
            command=self._open_settings,
        ).pack(pady=5, padx=5, fill="x")

        # Exit button
        ctk.CTkButton(
            control_frame,
            text="Exit",
            height=35,
            width=150,
            fg_color="gray",
            command=self.root.quit,
        ).pack(pady=5, padx=5, fill="x")

    def _populate_plot_functions(self) -> None:
        """Populate plot functions from configuration."""

        if "plot_functions" not in self.config:
            return

        for plot_name, plot_config in self.config["plot_functions"].items():
            var = ctk.BooleanVar(value=plot_config.get("enabled", True))

            checkbox = ctk.CTkCheckBox(
                self.plot_listbox,
                text=plot_name.replace("_", " ").title(),
                variable=var,
                onvalue=True,
                offvalue=False,
            )
            checkbox.pack(anchor="w", padx=5, pady=2)

            self.plot_checkboxes[plot_name] = checkbox

    def _refresh_events(self) -> None:
        """Refresh available events for selected year."""
        try:
            year = int(self.year_var.get())
            analyzer = F1Analyzer(year, "dummy", "R")
            events = analyzer.get_event_names()

            self.event_dropdown.configure(values=events)
            logger.info(f"Loaded {len(events)} events for {year}")

        except Exception as e:
            logger.error(f"Error refreshing events: {e}")
            messagebox.showerror("Error", f"Failed to load events: {e}")

    def _on_event_selected(self, value: str) -> None:
        """Handle event selection."""
        logger.info(f"Event selected: {value}")
        self._load_drivers()

    def _load_session(self) -> None:
        """Load FastF1 session data."""
        try:
            year = int(self.year_var.get())
            event = self.event_var.get()
            session = self.session_var.get()

            self.progress_label.configure(text=f"Loading {year} {event} {session}...")
            self.root.update()

            self.analyzer = get_or_create_analyzer(year, event, session)
            success = self.analyzer.load_session()

            if success:
                self._load_drivers()
                self.progress_label.configure(text="Session loaded successfully")
                messagebox.showinfo("Success", "Session loaded successfully")
                logger.info("Session loaded")
            else:
                messagebox.showerror("Error", "Failed to load session")
                logger.error("Session loading failed")

        except Exception as e:
            logger.error(f"Error loading session: {e}")
            messagebox.showerror("Error", f"Error: {e}")

    def _load_drivers(self) -> None:
        """Load and display drivers for current session."""
        try:
            if self.analyzer is None:
                year = int(self.year_var.get())
                event = self.event_var.get()
                session = self.session_var.get()
                self.analyzer = get_or_create_analyzer(year, event, session)
                self.analyzer.load_session()

            # Clear existing drivers
            for widget in self.driver_listbox.winfo_children():
                widget.destroy()
            self.driver_checkboxes.clear()

            # Load drivers
            drivers = self.analyzer.get_driver_abbreviations()

            for driver in drivers:
                var = ctk.BooleanVar(value=True)
                checkbox = ctk.CTkCheckBox(
                    self.driver_listbox,
                    text=driver,
                    variable=var,
                    onvalue=True,
                    offvalue=False,
                )
                checkbox.pack(anchor="w", padx=5, pady=2)
                self.driver_checkboxes[driver] = checkbox

            logger.info(f"Loaded {len(drivers)} drivers")

        except Exception as e:
            logger.error(f"Error loading drivers: {e}")

    def _start_analysis(self) -> None:
        """Start the analysis in a background thread."""

        # Validate inputs
        if self.analyzer is None:
            messagebox.showerror("Error", "Please load a session first")
            return

        # Get selected plots
        selected_plots = [
            name for name, checkbox in self.plot_checkboxes.items() if checkbox.get()
        ]

        if not selected_plots:
            messagebox.showerror("Error", "Please select at least one plot")
            return

        # Disable start button
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")

        # Start analysis in background thread
        thread = threading.Thread(target=self._run_analysis, args=(selected_plots,))
        thread.start()

    def _run_analysis(self, plots: List[str]) -> None:
        """Run analysis in background thread."""
        try:
            self.progress_tracker = ProgressTracker(len(plots))

            # Update progress callback
            def update_progress(update):
                self.progress_label.configure(text=update.phase.value)
                if update.total > 0:
                    percentage = update.percentage / 100.0
                    self.progress_bar.set(percentage)

                msg = f"[{update.phase.value}] {update.message}\n"
                if update.details:
                    msg += f"  {update.details}\n"

                self.progress_text.insert("end", msg)
                self.progress_text.see("end")

            self.progress_tracker.add_callback(update_progress)

            # Simulate analysis (replace with actual analysis)
            for i, plot in enumerate(plots):
                self.progress_tracker.update(
                    current=i + 1,
                    message=f"Processing {plot}...",
                )
                self.root.update()

        except Exception as e:
            logger.error(f"Analysis error: {e}")
            messagebox.showerror("Error", f"Analysis failed: {e}")

        finally:
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")

    def _stop_analysis(self) -> None:
        """Stop the currently running analysis."""
        logger.warning("Analysis stopped by user")
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")

    def _open_settings(self) -> None:
        """Open settings dialog."""
        messagebox.showinfo(
            "Settings",
            "Settings dialog would open here.\n" "Currently using config.yaml",
        )

    def run(self) -> None:
        """Start the GUI event loop."""
        self.root.mainloop()


def main():
    """Entry point for GUI application."""
    app = F1AnalysisGUI("config.yaml")
    app.run()


if __name__ == "__main__":
    main()
