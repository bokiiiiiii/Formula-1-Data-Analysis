import os
import importlib

# Get the current directory path
current_dir = os.path.dirname(__file__)

# List all Python files in the current directory excluding __init__.py
modules = [f[:-3] for f in os.listdir(current_dir) if f.endswith('.py') and f != '__init__.py']

# Dynamically import each module and add the function to the globals
for module_name in modules:
    module = importlib.import_module(f'.{module_name}', package=__name__)
    function = getattr(module, module_name, None)
    if function:
        globals()[module_name] = function

# Define the __all__ list to explicitly specify the public API of this package
__all__ = modules
