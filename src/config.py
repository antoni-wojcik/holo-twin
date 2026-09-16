"""
Central place for local, machine-specific configuration.

Reads from a `.env` file at the repo root (gitignored -- copy `.env.example`
to `.env` and edit it) via python-dotenv, falling back to real environment
variables, falling back to sensible defaults. This is the one place that
should read os.environ for this project; everything else imports from here.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from sys import platform


def _find_repo_root(marker: str = "pyproject.toml") -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).exists():
            return parent
    return Path.cwd()

# Load environment variables from .env file at repo root, if it exists
REPO_ROOT = _find_repo_root()
load_dotenv(REPO_ROOT / ".env")  # no-op if the file doesn't exist -- fine, defaults below still apply

# Set default paths and configuration values
DATA_ROOT = Path(os.environ.get("DATA_DIR_ROOT", str(REPO_ROOT / "data")))

# Find Inkscape path based on platform
if platform == "win32":
    # Default path for Inkscape on Windows
    INSKAPE_PATH = Path(os.environ.get("INSKAPE_PATH", r"C:\Program Files\Inkscape\bin\inkscape.exe"))
elif platform == "darwin":
    # Default path for Inkscape on macOS
    INSKAPE_PATH = Path(os.environ.get("INSKAPE_PATH", "/Applications/Inkscape.app/Contents/MacOS/inkscape"))
else:
    # Default: assume `inkscape` is available on PATH (e.g. Linux)
    INSKAPE_PATH = Path(os.environ.get("INSKAPE_PATH", "inkscape"))

# check if cairosvg package is installed. Installation may return errors on Windows,
# so catch any exceptions and set CAIROSVG_AVAILABLE to False if it fails.
try:
    import cairosvg  # noqa: F401
    CAIROSVG_AVAILABLE = True
except Exception:
    CAIROSVG_AVAILABLE = False