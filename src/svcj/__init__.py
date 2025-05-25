import os
import importlib

module_dir = os.path.dirname(__file__)
module_files = [f for f in os.listdir(module_dir) if f.endswith(".py") and f != "__init__.py"]

for module_file in module_files:
    module_name = module_file[:-3]  # Remove `.py`
    globals()[module_name] = importlib.import_module(f".{module_name}", package=__name__)

__all__ = module_files