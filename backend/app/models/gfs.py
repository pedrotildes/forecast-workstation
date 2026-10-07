"""NCEP GFS 0.25° via the NOMADS grib_filter service.

grib_filter returns only the requested variables/levels, already clipped to a
sub-region, so even the full 0.25° vertical profile for a region is small.
Runs and forecast steps are discovered from the NOMADS directory listings.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import re
import time

import httpx

from ..config import HTTP_TIMEOUT, USER_AGENT
from ..grib import Field, split_messages
from .base import Domain, Key, Model, TTLCache, cycle_floor, utcnow

FILTER = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
LISTING = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{ymd}/{hh}/atmos/"

PL = [1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500,
      450, 400, 350, 300, 250, 200, 150, 100, 70, 50]

# canonical name -> (NOMADS var, NOMADS level or None for pressure levels,
#                    accepted shortNames, accepted typeOfLevel, level or None)
_UPPER = {
    "gh": ("HGT", {"gh"}), "t": ("TMP", {"t"}), "r": ("RH", {"r"}),
    "q": ("SPFH", {"q"}), "u": ("UGRD", {"u"}), "v": ("VGRD", {"v"}),
    "w": ("VVEL", {"w"}),
}
_SINGLE = {
    "msl": ("PRMSL", "mean_sea_level", {"prmsl"}, None, None),
    "sp": ("PRES", "surface", {"sp"}, {"surface"}, None),
    "orog": ("HGT", "surface", {"orog"}, {"surface"}, None),
    "t2m": ("TMP", "2_m_above_ground", {"2t"}, None, None),
    "d2m": ("DPT", "2_m_above_ground", {"2d"}, None, None),
    "u10": ("UGRD", "10_m_above_ground", {"10u"}, None, None),
    "v10": ("VGRD", "10_m_above_ground", {"10v"}, None, None),
    "tp": ("APCP", "surface", {"tp"}, None, None),
    "cape": ("CAPE", "surface", {"cape"}, {"surface"}, None),
    "mlcape": ("CAPE", "90-0_mb_above_ground", {"cape"}, {"pressureFromGroundLayer"}, 9000),
    "mucape": ("CAPE", "255-0_mb_above_ground", {"cape"}, {"pressureFromGroundLayer"}, 25500),
    "cin": ("CIN", "surface", {"cin"}, {"surface"}, None),
    "pwat": ("PWAT", "entire_atmosphere_(considered_as_a_single_layer)", {"pwat"}, None, None),
    "tcc": ("TCDC", "entire_atmosphere", {"tcc"}, {"atmosphere", "entireAtmosphere"}, None),
    "lcc": ("LCDC", "low_cloud_layer", {"tcc", "lcc"}, {"lowCloudLayer"}, None),
    "mcc": ("MCDC", "middle_cloud_layer", {"tcc", "mcc"}, {"middleCloudLayer"}, None),
    "hcc": ("HCDC", "high_cloud_layer", {"tcc", "hcc"}, {"highCloudLayer"}, None),
    "gust": ("GUST", "surface", {"gust", "i10fg"}, {"surface"}, None),
    "refc": ("REFC", "entire_atmosphere", {"refc"}, None, None),
    "frzl": ("HGT", "0C_isotherm", {"gh"}, {"isothermZero"}, None),
    # max/min 2 m temperature over the preceding window (6 h at steps multiple of 6)
    "tmax": ("TMAX", "2_m_above_ground", {"tmax"}, None, None),
    "tmin": ("TMIN", "2_m_above_ground", {"tmin"}, None, None),
}


# NOMADS blocks clients above 120 requests/minute: keep a safe global pace.
_MIN_INTERVAL = 0.62
_last_req = 0.0
_throttle_lock: asyncio.Lock | None = None


async def _throttle():
    global _last_req, _throttle_lock
    if _throttle_lock is None:
        _throttle_lock = asyncio.Lock()
    async with _throttle_lock:
        wait = _last_req + _MIN_INTERVAL - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        _last_req = time.monotonic()


def _instantaneous(sr: str) -> bool:
    return "-" not in str(sr)


class GFS(Model):
    id = "gfs"
    name = "GFS"
    resolution = "0.25°"
    description = "NCEP Global Forecast System — 0.25°, 4 runs/dia, até +384 h"
    levels = PL
    native = set(_UPPER) | set(_SINGLE)
    source = "NOAA/NCEP NOMADS"

    def __init__(self):
        self._steps_cache = TTLCache(300)
        self._runs_cache = TTLCache(300)
        self._sem = asyncio.Semaphore(6)

    async def _listing(self, client: httpx.AsyncClient, run: dt.datetime) -> list[int]:
        url = LISTING.format(ymd=run.strftime("%Y%m%d"), hh=run.strftime("%H"))
        try:
            r = await client.get(url)
        except httpx.HTTPError:
            return []
        if r.status_code != 200:
            return []
        hh = run.strftime("%H")
        steps = {int(m) for m in re.findall(
            rf'gfs\.t{hh}z\.pgrb2\.0p25\.f(\d{{3}})"', r.text)}
        return sorted(steps)

    def last_nominal_step(self, run):
        return 384

    async def list_runs(self) -> list[dt.datetime]:
        cached = self._runs_cache.get("runs")
        if cached is not None:
            return cached
        base = cycle_floor(utcnow())
        cands = [base - dt.timedelta(hours=6 * i) for i in range(8)]
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT,
                                     headers={"User-Agent": USER_AGENT}) as c:
            res = await asyncio.gather(*(self._listing(c, r) for r in cands))
        runs = []
        for run, steps in zip(cands, res):
            if steps and 0 in steps:
                self._steps_cache.put(run, steps)
                runs.append(run)
        runs = runs[:5]
        self._runs_cache.put("runs", runs)
        return runs

    async def list_steps(self, run: dt.datetime) -> list[int]:
        s = self._steps_cache.get(run)
        if s is None:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT,
                                         headers={"User-Agent": USER_AGENT}) as c:
                s = await self._listing(c, run)
            self._steps_cache.put(run, s)
        return s

    async def _download(self, run, step, keys, domain: Domain) -> dict[Key, bytes]:
        upper = [k for k in keys if k[0] in _UPPER]
        single = [k for k in keys if k[0] in _SINGLE]
        # Group requests so that the var x level cross product stays small.
        groups: list[tuple[set, set]] = []
        if upper:
            groups.append(({_UPPER[k[0]][0] for k in upper}, {f"{k[1]}_mb" for k in upper}))
        if single:
            groups.append(({_SINGLE[k[0]][0] for k in single}, {_SINGLE[k[0]][1] for k in single}))
        area = (domain.lon1 - domain.lon0) * (domain.lat1 - domain.lat0)
        if len(groups) == 2 and area <= 64:
            # small point-extraction tile: one request (the var x level cross product is cheap)
            groups = [(groups[0][0] | groups[1][0], groups[0][1] | groups[1][1])]

        blobs: list[bytes] = []
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT,
                                     headers={"User-Agent": USER_AGENT}) as c:
            for vars_, levs in groups:
                params = [("dir", f"/gfs.{run:%Y%m%d}/{run:%H}/atmos"),
                          ("file", f"gfs.t{run:%H}z.pgrb2.0p25.f{step:03d}")]
                params += [(f"var_{v}", "on") for v in sorted(vars_)]
                params += [(f"lev_{lv}", "on") for lv in sorted(levs)]
                params += [("subregion", ""), ("leftlon", domain.lon0),
                           ("rightlon", domain.lon1), ("toplat", domain.lat1),
                           ("bottomlat", domain.lat0)]
                async with self._sem:
                    for attempt in range(3):
                        await _throttle()
                        try:
                            r = await c.get(FILTER, params=params)
                            if r.status_code == 200 and r.content[:4] == b"GRIB":
                                blobs.append(r.content)
                                break
                        except httpx.HTTPError:
                            pass
                        await asyncio.sleep(1.5 * (attempt + 1))

        out: dict[Key, bytes] = {}
        best_rank: dict[Key, int] = {}
        for blob in blobs:
            for hdr, msg in split_messages(blob):
                for k in keys:
                    rank = self._match(k, hdr, step)
                    if rank is not None and rank < best_rank.get(k, 99):
                        out[k], best_rank[k] = msg, rank
        return out

    @staticmethod
    def _match(k: Key, hdr: dict, step: int) -> int | None:
        """Return a preference rank (lower is better) if ``hdr`` matches key."""
        name, level = k
        sn, tl, sr = hdr["shortName"], hdr["typeOfLevel"], str(hdr["stepRange"])
        if name in _UPPER:
            if sn in _UPPER[name][1] and tl == "isobaricInhPa" and hdr["level"] == level:
                return 0
            return None
        _, _, sns, tls, lev = _SINGLE[name]
        if sn not in sns or (tls and tl not in tls):
            return None
        # pressureFromGroundLayer: ecCodes reports the layer top in Pa (9000 = 90-0 hPa)
        if lev is not None and hdr.get("level") != lev:
            return None
        if name == "tp":
            return 0 if sr.startswith("0-") else None
        if name in ("tmax", "tmin"):
            a, b = (int(x) for x in sr.split("-")) if "-" in sr else (step, step)
            return 0 if b - a == 6 else (1 if b - a == 3 else None)
        return 0 if _instantaneous(sr) else 1

    def _postprocess(self, key: Key, f: Field) -> Field:
        if key[0] == "tp":
            f.units = "mm"
        return f
