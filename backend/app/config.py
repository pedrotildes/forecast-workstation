"""Paths and global settings."""
from __future__ import annotations

import os
import sys
from pathlib import Path

VERSION = "1.5.1"
FROZEN = getattr(sys, "frozen", False)   # running from a PyInstaller bundle


def _user_data_dir() -> Path:
    """Writable per-user folder for the packaged app (never inside the bundle)."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "ForecastWorkstation"


if FROZEN:
    ROOT = Path(sys._MEIPASS)                      # noqa: SLF001
    FRONTEND_DIR = ROOT / "frontend"
    _default_cache = _user_data_dir() / "cache"
else:
    ROOT = Path(__file__).resolve().parents[2]
    FRONTEND_DIR = ROOT / "frontend"
    _default_cache = ROOT / "cache"

CACHE_DIR = Path(os.environ.get("FORECAST_CACHE", _default_cache))
# Event archive: one location shared by the development server and the packaged app, and
# never touched by the automatic cache clean-up.
ARCHIVE_DIR = Path(os.environ.get("FORECAST_ARCHIVE", _user_data_dir() / "archive"))
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
# Alert service (python -m app.alerts): state, figures and settings, shared like the archive.
ALERTS_DIR = Path(os.environ.get("FORECAST_ALERTS", _user_data_dir() / "alerts"))
ALERTS_DIR.mkdir(parents=True, exist_ok=True)
GRIB_CACHE = CACHE_DIR / "grib"
IMG_CACHE = CACHE_DIR / "img"

for _d in (GRIB_CACHE, IMG_CACHE):
    _d.mkdir(parents=True, exist_ok=True)

USER_AGENT = f"Forecast-Workstation/{VERSION} (research)"
HTTP_TIMEOUT = 60.0
