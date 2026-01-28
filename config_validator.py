"""Configuration validation using Pydantic models."""

from typing import Optional, List, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field, validator, ValidationError
from logger_config import get_logger

logger = get_logger(__name__)


class DataConfig(BaseModel):
    """Data paths configuration."""

    cache_path: str = Field(default="../cache")
    output_folder: str = Field(default="../pic")
    log_dir: str = Field(default="./logs")

    @validator("cache_path", "output_folder", "log_dir")
    def validate_path(cls, v):
        """Ensure paths are valid strings."""
        if not isinstance(v, str):
            raise ValueError("Path must be a string")
        return v


class GUIConfig(BaseModel):
    """GUI settings configuration."""

    theme: str = Field(default="System")
    color_theme: str = Field(default="blue")
    window_width: int = Field(default=900, ge=600, le=2000)
    window_height: int = Field(default=700, ge=400, le=1500)

    @validator("theme")
    def validate_theme(cls, v):
        """Validate theme value."""
        allowed = ["Light", "Dark", "System"]
        if v not in allowed:
            raise ValueError(f"Theme must be one of {allowed}")
        return v

    @validator("color_theme")
    def validate_color_theme(cls, v):
        """Validate color theme."""
        allowed = ["blue", "green", "dark-blue"]
        if v not in allowed:
            raise ValueError(f"Color theme must be one of {allowed}")
        return v


class ParallelConfig(BaseModel):
    """Parallel processing configuration."""

    enabled: bool = Field(default=True)
    max_workers: int = Field(default=4, ge=1, le=32)
    timeout_seconds: int = Field(default=300, ge=30, le=3600)

    @validator("max_workers")
    def validate_workers(cls, v):
        """Ensure reasonable worker count."""
        import os

        cpu_count = os.cpu_count() or 4
        if v > cpu_count * 2:
            logger.warning(
                f"max_workers ({v}) > 2*CPU_count ({cpu_count}), " "may cause overhead"
            )
        return v


class MonteCarloConfig(BaseModel):
    """Monte Carlo simulation configuration."""

    simulations: int = Field(default=10000, ge=1000, le=100000)
    random_seed: Optional[int] = Field(default=None)
    base_sc_probability: float = Field(default=0.40, ge=0.0, le=1.0)
    use_track_data: bool = Field(default=True)

    @validator("simulations")
    def validate_simulations(cls, v):
        """Validate simulation count."""
        if v < 1000:
            logger.warning(f"Low simulation count ({v}) may reduce accuracy")
        return v


class PlotFunctionConfig(BaseModel):
    """Single plot function configuration."""

    enabled: bool = Field(default=True)
    session: str = Field(default="R")
    description: Optional[str] = None

    @validator("session")
    def validate_session(cls, v):
        """Validate session type."""
        allowed = ["FP1", "FP2", "FP3", "Q", "R", "SS", "S"]
        if v not in allowed:
            raise ValueError(f"Session must be one of {allowed}")
        return v


class F1AnalysisConfig(BaseModel):
    """Main F1 analysis configuration."""

    default_year: int = Field(default=2025, ge=2018, le=2030)
    default_session: str = Field(default="R")
    default_event: str = Field(default="Abu Dhabi")
    enable_all: bool = Field(default=True)

    @validator("default_year")
    def validate_year(cls, v):
        """Validate year range."""
        if v < 2018:
            raise ValueError("FastF1 data only available from 2018")
        return v

    @validator("default_session")
    def validate_session(cls, v):
        """Validate default session."""
        allowed = ["FP1", "FP2", "FP3", "Q", "R", "SS", "S"]
        if v not in allowed:
            raise ValueError(f"Session must be one of {allowed}")
        return v


class ConfigValidator:
    """Configuration validator using Pydantic models."""

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize validator with configuration dictionary.

        Args:
            config_dict: Configuration dictionary to validate
        """
        self.config_dict = config_dict
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self) -> bool:
        """Validate entire configuration.

        Returns:
            True if valid, False if errors found
        """
        logger.info("Validating configuration...")

        # Validate main F1 analysis config
        if "f1_analysis" in self.config_dict:
            try:
                F1AnalysisConfig(**self.config_dict["f1_analysis"])
            except ValidationError as e:
                self.errors.append(f"f1_analysis: {e}")

        # Validate data config
        if "data" in self.config_dict:
            try:
                DataConfig(**self.config_dict["data"])
            except ValidationError as e:
                self.errors.append(f"data: {e}")

        # Validate GUI config
        if "gui" in self.config_dict:
            try:
                GUIConfig(**self.config_dict["gui"])
            except ValidationError as e:
                self.errors.append(f"gui: {e}")

        # Validate parallel config
        if "performance" in self.config_dict:
            perf_config = self.config_dict["performance"]
            if "parallel" in perf_config:
                try:
                    ParallelConfig(**perf_config["parallel"])
                except ValidationError as e:
                    self.errors.append(f"performance.parallel: {e}")

        # Validate Monte Carlo config
        if "race_strategy" in self.config_dict:
            strategy_config = self.config_dict["race_strategy"]
            if "monte_carlo" in strategy_config:
                try:
                    MonteCarloConfig(**strategy_config["monte_carlo"])
                except ValidationError as e:
                    self.errors.append(f"race_strategy.monte_carlo: {e}")

        # Validate plot functions
        if "plot_functions" in self.config_dict:
            for func_name, func_config in self.config_dict["plot_functions"].items():
                try:
                    PlotFunctionConfig(**func_config)
                except ValidationError as e:
                    self.errors.append(f"plot_functions.{func_name}: {e}")

        # Report results
        if self.errors:
            logger.error(
                f"Configuration validation failed with {len(self.errors)} errors:"
            )
            for error in self.errors:
                logger.error(f"  - {error}")
            return False

        if self.warnings:
            logger.warning(f"Configuration has {len(self.warnings)} warnings:")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")

        logger.info("Configuration validation passed")
        return True

    def validate_paths_exist(self) -> bool:
        """Validate that critical paths exist.

        Returns:
            True if all paths valid
        """
        if "data" not in self.config_dict:
            return True

        data_config = self.config_dict["data"]
        paths_valid = True

        # Check cache path
        if "cache_path" in data_config:
            cache_path = Path(data_config["cache_path"])
            if not cache_path.exists():
                try:
                    cache_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created cache directory: {cache_path}")
                except Exception as e:
                    self.errors.append(f"Cannot create cache_path: {e}")
                    paths_valid = False

        # Check output folder
        if "output_folder" in data_config:
            output_path = Path(data_config["output_folder"])
            if not output_path.exists():
                try:
                    output_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created output directory: {output_path}")
                except Exception as e:
                    self.errors.append(f"Cannot create output_folder: {e}")
                    paths_valid = False

        # Check log directory
        if "log_dir" in data_config:
            log_path = Path(data_config["log_dir"])
            if not log_path.exists():
                try:
                    log_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created log directory: {log_path}")
                except Exception as e:
                    self.errors.append(f"Cannot create log_dir: {e}")
                    paths_valid = False

        return paths_valid

    def get_errors(self) -> List[str]:
        """Get validation errors.

        Returns:
            List of error messages
        """
        return self.errors

    def get_warnings(self) -> List[str]:
        """Get validation warnings.

        Returns:
            List of warning messages
        """
        return self.warnings


def validate_config(config_dict: Dict[str, Any]) -> bool:
    """Validate configuration dictionary.

    Args:
        config_dict: Configuration to validate

    Returns:
        True if valid, False otherwise
    """
    validator = ConfigValidator(config_dict)

    if not validator.validate():
        return False

    if not validator.validate_paths_exist():
        return False

    return True


def validate_config_file(filepath: str) -> bool:
    """Validate configuration file.

    Args:
        filepath: Path to configuration file

    Returns:
        True if valid, False otherwise
    """
    import yaml
    import json

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            if filepath.endswith(".yaml") or filepath.endswith(".yml"):
                config_dict = yaml.safe_load(f)
            else:
                config_dict = json.load(f)

        return validate_config(config_dict)

    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        return False
