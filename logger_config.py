"""Logging configuration for F1 Data Analysis project."""

import logging
import os
from pathlib import Path


def setup_logging(
    log_level: int = logging.INFO,
    log_dir: str = "./logs",
    log_file: str = "f1_analysis.log",
) -> logging.Logger:
    """Setup logging configuration.

    Args:
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_dir: Directory to store log files
        log_file: Name of the log file

    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger("f1_analysis")
    logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_formatter = logging.Formatter(
        "%(levelname)s: %(message)s",
    )

    # File handler
    log_path = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Log file: {log_path}")
    return logger


def get_logger(name: str = "f1_analysis") -> logging.Logger:
    """Get or create a logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
