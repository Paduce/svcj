from pathlib import Path
from dynaconf import Dynaconf

# Base directory of the project (two levels up from this file)
BASE_DIR = Path(__file__).resolve().parent.parent

# Initialize Dynaconf to load settings from pyproject.toml and .env
settings = Dynaconf(
    settings_files=[
        str(BASE_DIR / "pyproject.toml"),  # load TOML config
        str(BASE_DIR / ".env"),            # load environment overrides
    ],
    envvar_prefix=False,  # read env vars without any prefix
    load_dotenv=True,     # auto-load .env file into os.environ
)

# Expose all attributes of Dynaconf and raiseP error if any existing global is overwritten
for key in settings.keys():
    if key in globals():
        raise RuntimeError(f"Environment variable '{key}' would overwrite an existing global variable.")
    globals()[key] = getattr(settings, key)

__all__ = list(settings.keys())