"""DWD ICON-EU (0.0625° ≈ 7 km) from opendata.dwd.de.

Every field / level / step is a separate bzip2-compressed GRIB2 file covering the whole
ICON-EU domain (23.5°W–62.5°E, 29.5°N–70.5°N), so files are downloaded whole and cropped
locally (like the ECMWF global files). The server keeps only the latest run of each
cycle (≈ the last 24 h); runs 00/06/12/18Z reach +120 h, 03/09/15/21Z +48 h.
"""
from __future__ import annotations

import asyncio
import bz2
import datetime as dt
import re

import httpx

from ..config import HTTP_TIMEOUT, USER_AGENT
from ..grib import Field
from .base import Domain, Key, Model, TTLCache

BASE = "https://opendata.dwd.de/weather/nwp/icon-eu/grib"
PREFIX = "icon-eu_europe_regular-lat-lon"
PL = [1000, 950, 925, 900, 875, 850, 825, 800, 775, 700, 600, 500, 400, 300, 250, 200, 150, 100]

# canonical -> (DWD directory / parameter name, kind)
_UPPER = {"gh": "fi", "t": "t", "r": "relhum", "q": "qv", "u": "u", "v": "v", "w": "omega"}
_SINGLE = {
    "msl": "pmsl", "sp": "ps", "t2m": "t_2m", "d2m": "td_2m", "u10": "u_10m", "v10": "v_10m",
    "tp": "tot_prec", "mlcape": "cape_ml", "pwat": "tqv", "tcc": "clct", "lcc": "clcl",
    "mcc": "clcm", "hcc": "clch", "gust": "vmax_10m", "frzl": "hzerocl",
    "tmax": "tmax_2m", "tmin": "tmin_2m",
}
COVERAGE = (-23.5, 62.5, 29.5, 70.5)   # lon0, lon1, lat0, lat1


class ICON_EU(Model):
    id = "icon_eu"
    name = "ICON-EU"
    resolution = "0.0625°"
    description = ("DWD ICON-EU — 0,0625° (~7 km), Europa; 00/06/12/18Z até +120 h, "
                   "03/09/15/21Z até +48 h (não cobre Açores nem Canárias)")
    levels = PL
    native = set(_UPPER) | set(_SINGLE) | {"orog"}
    global_download = True        # whole-domain files, cropped locally
    coverage = COVERAGE
    source = "© Deutscher Wetterdienst (DWD), dados abertos"

    def __init__(self):
        self._runs_cache = TTLCache(300)
        self._steps_cache = TTLCache(300)
        self._sem = asyncio.Semaphore(8)

    def last_nominal_step(self, run):
        return 120 if run.hour % 6 == 0 else 48

    # ------------------------------------------------------------ discovery
    async def _cycle(self, c: httpx.AsyncClient, hh: int):
        try:
            r = await c.get(f"{BASE}/{hh:02d}/pmsl/")
        except httpx.HTTPError:
            return None
        if r.status_code != 200:
            return None
        found = re.findall(r"single-level_(\d{10})_(\d{3})_PMSL", r.text)
        if not found:
            return None
        run = dt.datetime.strptime(max(x[0] for x in found), "%Y%m%d%H")
        steps = sorted({int(s) for rr, s in found if rr == run.strftime("%Y%m%d%H")})
        return run, steps

    async def list_runs(self) -> list[dt.datetime]:
        cached = self._runs_cache.get("runs")
        if cached is not None:
            return cached
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT}) as c:
            res = await asyncio.gather(*(self._cycle(c, h) for h in range(0, 24, 3)))
        runs = []
        for item in res:
            if item and 0 in item[1]:
                self._steps_cache.put(item[0], item[1])
                runs.append(item[0])
        runs.sort(reverse=True)
        self._runs_cache.put("runs", runs[:5])
        return runs[:5]

    async def list_steps(self, run: dt.datetime) -> list[int]:
        s = self._steps_cache.get(run)
        if s is None:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT,
                                         headers={"User-Agent": USER_AGENT}) as c:
                item = await self._cycle(c, run.hour)
            s = item[1] if item and item[0] == run else []
            self._steps_cache.put(run, s)
        return s

    # ------------------------------------------------------------ download
    def _url(self, run: dt.datetime, step: int, key: Key) -> str:
        name, level = key
        r = run.strftime("%Y%m%d%H")
        hh = run.strftime("%H")
        if name == "orog":
            return f"{BASE}/{hh}/hsurf/{PREFIX}_time-invariant_{r}_HSURF.grib2.bz2"
        if name in _UPPER:
            p = _UPPER[name]
            return (f"{BASE}/{hh}/{p}/{PREFIX}_pressure-level_{r}_{step:03d}_{level}_"
                    f"{p.upper()}.grib2.bz2")
        p = _SINGLE[name]
        return f"{BASE}/{hh}/{p}/{PREFIX}_single-level_{r}_{step:03d}_{p.upper()}.grib2.bz2"

    async def _get(self, c: httpx.AsyncClient, url: str) -> bytes | None:
        async with self._sem:
            for attempt in range(3):
                try:
                    r = await c.get(url)
                    if r.status_code == 404:
                        return None
                    if r.status_code == 200:
                        return await asyncio.to_thread(bz2.decompress, r.content)
                except (httpx.HTTPError, OSError):
                    pass
                await asyncio.sleep(1.0 * (attempt + 1))
        return None

    async def _download(self, run, step, keys, domain: Domain) -> dict[Key, bytes]:
        keys = [k for k in keys if k[0] in self.native and (not k[1] or k[1] in self.levels)]
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT}) as c:
            blobs = await asyncio.gather(*(self._get(c, self._url(run, step, k)) for k in keys))
        return {k: b for k, b in zip(keys, blobs) if b}

    def _postprocess(self, key: Key, f: Field) -> Field:
        name = key[0]
        if name == "gh":                      # geopotential (m² s⁻²) -> gpm
            f.values = f.values / 9.80665
            f.units = "gpm"
        elif name == "tp":
            f.units = "mm"
        return f
