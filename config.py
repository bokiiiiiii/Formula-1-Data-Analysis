"""Configuration management."""

import json
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict, field


@dataclass
class PlotFunctionConfig:
    """Configuration for individual plot functions."""

    enabled: bool = False
    session: str = "R"


@dataclass
class Config:
    """Main configuration class for the F1 analysis project."""

    # Basic settings
    year: int = 2025
    session_name: str = "R"
    enable_all: bool = True
    folder_path: str = "../pic"
    cache_path: str = "../cache"

    # Logging settings
    log_level: str = "INFO"
    log_dir: str = "./logs"

    # Plot functions configuration
    plot_functions: Dict[str, PlotFunctionConfig] = field(default_factory=dict)

    # Instagram settings
    instagram_enabled: bool = False
    instagram_delay_seconds: int = 60

    # Matplotlib settings
    figure_dpi: int = 125
    figure_width: float = 8.64
    figure_height: float = 10.8

    def __post_init__(self):
        """Initialize default plot functions if not provided."""
        if not self.plot_functions:
            self.plot_functions = {
                # FP1
                "plot_track_with_annotated_corners": PlotFunctionConfig(False, "FP1"),
                # Q
                "annotated_qualifying_flying_lap": PlotFunctionConfig(False, "Q"),
                # R
                "annotated_race_fatest_lap": PlotFunctionConfig(False, "R"),
                "driver_laptimes_distribution": PlotFunctionConfig(False, "R"),
                "driver_laptimes_scatterplot": PlotFunctionConfig(False, "R"),
                "race_fatest_lap_telemetry_data": PlotFunctionConfig(False, "R"),
                "team_pace_ranking": PlotFunctionConfig(False, "R"),
                "driver_fuel_corrected_laptimes_scatterplot": PlotFunctionConfig(
                    False, "R"
                ),
                "driver_fuel_corrected_laptimes_gaussian_processes": PlotFunctionConfig(
                    False, "R"
                ),
                "driver_race_evolution_heatmap": PlotFunctionConfig(False, "R"),
                "monte_carlo_race_strategy": PlotFunctionConfig(False, "R"),
                # SQ
                "annotated_sprint_qualifying_flying_lap": PlotFunctionConfig(
                    False, "SQ"
                ),
            }

    def is_plot_function_enabled(self, func_name: str) -> bool:
        """Check if a plot function is enabled.

        Args:
            func_name: Name of the plot function

        Returns:
            True if enabled (or enable_all is True), False otherwise
        """
        if self.enable_all:
            return True

        func_config = self.plot_functions.get(func_name)
        return func_config.enabled if func_config else False

    def get_session_for_plot(self, func_name: str) -> str:
        """Get the session for a plot function.

        Args:
            func_name: Name of the plot function

        Returns:
            Session name (e.g., "R", "Q", "FP1")
        """
        func_config = self.plot_functions.get(func_name)
        return func_config.session if func_config else "R"

    def should_process_plot_function(self, func_name: str) -> bool:
        """Check if a plot function should be processed based on session_name.

        Args:
            func_name: Name of the plot function

        Returns:
            True if the function's session is in session_name
        """
        func_session = self.get_session_for_plot(func_name)
        return func_session in self.session_name

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        config_dict = asdict(self)
        # Convert PlotFunctionConfig objects to dicts
        config_dict["plot_functions"] = {
            name: asdict(cfg) for name, cfg in config_dict["plot_functions"].items()
        }
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        # Convert plot_functions dicts to PlotFunctionConfig objects
        plot_funcs = {}
        if "plot_functions" in config_dict:
            for name, cfg in config_dict["plot_functions"].items():
                if isinstance(cfg, dict):
                    plot_funcs[name] = PlotFunctionConfig(**cfg)
                else:
                    plot_funcs[name] = cfg
            config_dict["plot_functions"] = plot_funcs

        return cls(**config_dict)

    def save(self, filepath: str) -> None:
        """Save configuration to JSON or YAML file.

        Args:
            filepath: Path to save the configuration file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        if filepath.endswith(".yaml") or filepath.endswith(".yml"):
            with open(filepath, "w", encoding="utf-8") as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "Config":
        """Load configuration from JSON or YAML file.

        Args:
            filepath: Path to the configuration file

        Returns:
            Config instance

        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        if filepath.endswith(".yaml") or filepath.endswith(".yml"):
            with open(filepath, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                config_dict = json.load(f)

        return cls.from_dict(config_dict)


def create_default_config(filepath: str = "config.json") -> Config:
    """Create and save a default configuration file.

    Args:
        filepath: Path to save the default configuration

    Returns:
        Config instance
    """
    config = Config()
    config.save(filepath)
    return config


def get_config(filepath: str = "config.json") -> Config:
    """Get configuration, creating default if it doesn't exist.

    Args:
        filepath: Path to the configuration file

    Returns:
        Config instance
    """
    if os.path.exists(filepath):
        return Config.load(filepath)
    else:
        return create_default_config(filepath)
