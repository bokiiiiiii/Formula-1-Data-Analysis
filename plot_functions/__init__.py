import os
import importlib
from logger_config import get_logger

logger = get_logger(__name__)

# Get the current directory path
current_dir = os.path.dirname(__file__)

# List all Python files in the current directory excluding __init__.py and support modules
excluded = {"__init__.py", "utils.py", "plot_registry.py", "plot_runner.py"}
modules = [
    f[:-3] for f in os.listdir(current_dir) if f.endswith(".py") and f not in excluded
]

# Dynamically import each module
for module_name in modules:
    try:
        module = importlib.import_module(f".{module_name}", package=__name__)
        globals()[module_name] = module
    except Exception as e:
        logger.error(f"Failed to import module {module_name}: {e}")

# Define the __all__ list
__all__ = modules
