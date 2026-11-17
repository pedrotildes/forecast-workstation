"""ERA5 reanalysis as a model: it exists only in the event archive (historical events
imported with :mod:`app.era5`), so runs and steps come from the archive and nothing is
downloaded on demand."""
from __future__ import annotations

import datetime as dt

from ..era5 import LEVELS
from .base import FieldUnavailable, Model


class ERA5(Model):
    id = "era5"
    name = "ERA5 (reanálise)"
    resolution = "0.25°"
    description = ("Reanálise ERA5 do ECMWF (Copernicus) — 0,25°, horária, 27 níveis até 100 hPa; "
                   "só para eventos históricos importados no separador Eventos")
    levels = LEVELS
    native = {"gh", "t", "r", "u", "v", "w", "msl", "sp", "u10", "v10", "t2m", "d2m", "tcc", "pwat",
              "mucape", "tp", "gust"}
    global_download = True
    source = "ECMWF ERA5 (Copernicus C3S), via NCAR RDA ds633.0"
    run_every = 6
    reanalysis = True                  # hourly analyses stored as run + step: no "forecast" wording

    async def list_runs(self) -> list[dt.datetime]:
        return []                      # the base class adds the archived runs

    async def list_steps(self, run: dt.datetime) -> list[int]:
        return []

    async def _download(self, run, step, keys, domain):
        raise FieldUnavailable("ERA5: só existe para eventos históricos importados (separador Eventos)")

    def last_nominal_step(self, run):
        return 6
