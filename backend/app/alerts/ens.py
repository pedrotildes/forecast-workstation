"""ECMWF ENS for the alert service: the 50 perturbed members ("ef" files) and the
probability products ("ep" file) of ECMWF Open Data.

Member fields are global GRIB messages (one per member, ~0.3–0.7 MB each); only the
ones needed are fetched by byte range and immediately cropped to the Portuguese box
(areas.BOX), so a whole run of the variables used here stays in a few tens of MB on disk.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import shutil

import httpx
import numpy as np

from ..config import CACHE_DIR, HTTP_TIMEOUT, USER_AGENT
from ..grib import decode_message
from ..models.base import FieldUnavailable
from ..models.ecmwf import _ECMWFBase
from .areas import BOX

ENS_CACHE = CACHE_DIR / "alerts" / "ens"
H = dt.timedelta(hours=1)

# canonical name -> ECMWF param(s) in the member files (3-h windows to +144 h, 6-h after)
MEMBER_PARAMS = {
    "gust": ("10fg",),                 # maximum gust since the previous post-processing step
    "tp": ("tp",),                     # accumulated from the start (m)
    "tmax": ("mx2t3", "mx2t6"),
    "tmin": ("mn2t3", "mn2t6"),
    "t2m": ("2t",),
    "msl": ("msl",),
    "swh": ("swh",),                   # significant height of wind waves and swell (m)
    "mwd": ("mwd",),                   # mean wave direction (degrees, where waves come from)
    "t850": ("t",),                    # temperature at 850 hPa (snow: temperature aloft)
    # thunderstorms: most-unstable CAPE (the 2024 archive has only "cape", computed the same
    # way from the most unstable parcel below 350 hPa) and the wind at 925 and 500 hPa
    "mucape": ("mucape", "cape"),
    "u925": ("u",), "v925": ("v",), "u500": ("u",), "v500": ("v",),
}
# fields on pressure levels: name -> level (hPa); the others are single-level ("sfc")
MEMBER_LEVEL = {"t850": 850, "u925": 925, "v925": 925, "u500": 500, "v500": 500}
WAVE_NAMES = {"swh", "mwd"}
# probability products of the "ep" file (24-h windows every 12 h)
EP_PARAMS = {
    "p_gust54": "10fgg15", "p_gust90": "10fgg25",
    "p_tp10": "tpg10", "p_tp25": "tpg25", "p_tp50": "tpg50", "p_tp100": "tpg100",
    "p_t850_gt1p5": "ptsa_gt_1p5stdev", "p_t850_gt2": "ptsa_gt_2stdev",
    "p_t850_lt1p5": "ptsa_lt_1p5stdev", "p_t850_lt2": "ptsa_lt_2stdev",
    # wave ensemble ("waef" ep file): probability of Hs above 2/4/6/8 m (instantaneous, 12-hourly)
    "p_swh2": "swhg2", "p_swh4": "swhg4", "p_swh6": "swhg6", "p_swh8": "swhg8",
}
WAVE_EP = {"p_swh2", "p_swh4", "p_swh6", "p_swh8"}


class EnsMembers(_ECMWFBase):
    id = "ens_members"
    name = "ECMWF ENS"
    path_model = "ifs"
    levels = []
    run_every = 12

    def stream(self, run):
        return "enfo"

    def nominal_steps(self, run):
        return list(range(0, 144, 3)) + list(range(144, 361, 6))

    def _url(self, mirror, run, step, ext, s=None):
        return (f"{mirror}/{run:%Y%m%d}/{run:%H}z/ifs/0p25/enfo/"
                f"{run:%Y%m%d%H}0000-{step}h-enfo-ef.{ext}")

    async def available(self, run: dt.datetime, step: int) -> bool:
        async with httpx.AsyncClient(timeout=20, headers={"User-Agent": USER_AGENT}) as c:
            return await self._exists(c, run, step)


class EnsProb(_ECMWFBase):
    id = "ens_prob"
    name = "ECMWF ENS (probabilidades)"
    path_model = "ifs"
    levels = []
    run_every = 12

    def stream(self, run):
        return "enfo"

    def nominal_steps(self, run):
        return [240]

    def _url(self, mirror, run, step, ext, s=None):
        return (f"{mirror}/{run:%Y%m%d}/{run:%H}z/ifs/0p25/enfo/"
                f"{run:%Y%m%d%H}0000-240h-enfo-ep.{ext}")


class WaveMembers(EnsMembers):
    id = "ens_wave_members"
    name = "ECMWF ENS (ondas)"

    def _url(self, mirror, run, step, ext, s=None):
        return (f"{mirror}/{run:%Y%m%d}/{run:%H}z/ifs/0p25/waef/"
                f"{run:%Y%m%d%H}0000-{step}h-waef-ef.{ext}")


class WaveProb(EnsProb):
    id = "ens_wave_prob"
    name = "ECMWF ENS ondas (probabilidades)"

    def _url(self, mirror, run, step, ext, s=None):
        return (f"{mirror}/{run:%Y%m%d}/{run:%H}z/ifs/0p25/waef/"
                f"{run:%Y%m%d%H}0000-240h-waef-ep.{ext}")


MEMBERS = EnsMembers()
PROB = EnsProb()
WAVE_MEMBERS = WaveMembers()
WAVE_PROB = WaveProb()
_SEM = asyncio.Semaphore(12)


def _cache_path(run: dt.datetime, name: str, step: int):
    return ENS_CACHE / run.strftime("%Y%m%d%H") / f"{name}_{step:03d}.npz"


def _decode_crop(buf: bytes):
    lon0, lon1, lat0, lat1 = BOX
    _, f = decode_message(buf, (lon0, lon1, lat0, lat1))
    return f


async def _get(c: httpx.AsyncClient, url: str, e: dict) -> bytes:
    async with _SEM:
        return await MEMBERS._range(c, url, e)


async def latest_complete_run(min_step: int = 168) -> dt.datetime | None:
    """Newest 00/12Z ENS run whose member files reach ``min_step`` and whose ep file exists."""
    runs = await MEMBERS.list_runs()
    async with httpx.AsyncClient(timeout=20, headers={"User-Agent": USER_AGENT}) as c:
        for r in runs[:3]:
            if r.hour not in (0, 12):
                continue
            if await MEMBERS._exists(c, r, min_step) and await PROB._exists(c, r, 240):
                return r
    return None


async def members(run: dt.datetime, name: str, step: int) -> dict | None:
    """All 50 members of one field at one step, cropped to BOX:
    {"lats", "lons", "values" (50, ny, nx) float32, "units"}. None if not published."""
    p = _cache_path(run, name, step)
    if p.exists():
        z = np.load(p)
        return {"lats": z["lats"], "lons": z["lons"], "values": z["values"].astype(np.float32),
                "units": str(z["units"])}
    params = MEMBER_PARAMS[name]
    src = WAVE_MEMBERS if name in WAVE_NAMES else MEMBERS
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT}) as c:
        try:
            url, entries = await src._index(c, run, step)
        except FieldUnavailable:
            return None
        lev = MEMBER_LEVEL.get(name)
        todo = []
        for prm in params:                  # the first of the alternative params present
            todo = [e for e in entries if e.get("param") == prm and e.get("type") == "pf"
                    and (e.get("levtype") == "sfc" if lev is None
                         else e.get("levtype") == "pl" and str(e.get("levelist")) == str(lev))]
            if todo:
                break
        if not todo:
            return None
        todo.sort(key=lambda e: int(e.get("number", 0)))
        blobs = await asyncio.gather(*(_get(c, url, e) for e in todo))
    fields = await asyncio.to_thread(lambda: [_decode_crop(b) for b in blobs])
    vals = np.stack([f.values for f in fields]).astype(np.float32)
    units = fields[0].units
    if name == "tp" and units.strip() == "m":
        vals, units = vals * 1000.0, "mm"
    elif name in ("tmax", "tmin", "t2m", "t850") and units.strip() == "K":
        vals, units = vals - 273.15, "°C"
    elif name == "gust":
        vals, units = vals * 3.6, "km/h"
    elif name == "msl":
        vals, units = vals / 100.0, "hPa"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".part.npz")
    np.savez_compressed(tmp, lats=fields[0].lats, lons=fields[0].lons,
                        values=vals.astype(np.float16), units=units)
    tmp.replace(p)
    return {"lats": fields[0].lats, "lons": fields[0].lons, "values": vals, "units": units}


async def ep_fields(run: dt.datetime, names: list[str]) -> dict:
    """Probability products: {name: {"windows": [(start_step, end_step)], "values": (n, ny, nx) %}}."""
    wave = bool(set(names) & WAVE_EP)
    if wave and not set(names) <= WAVE_EP:
        raise ValueError("ep_fields: wave and atmospheric products come from different files")
    src = WAVE_PROB if wave else PROB
    p = ENS_CACHE / run.strftime("%Y%m%d%H") / ("ep_wave.npz" if wave else "ep.npz")
    if p.exists():
        z = np.load(p, allow_pickle=False)
        have = {k[:-5] for k in z.files if k.endswith("__val")}
        if set(names) <= have:
            return {"lats": z["lats"], "lons": z["lons"],
                    **{n: {"windows": [tuple(w) for w in z[f"{n}__win"]],
                           "values": z[f"{n}__val"].astype(np.float32)} for n in names}}
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT}) as c:
        url, entries = await src._index(c, run, 240)
        want = {EP_PARAMS[n]: n for n in names}
        todo = [e for e in entries if e.get("param") in want]
        blobs = await asyncio.gather(*(_get(c, url, e) for e in todo))
    fields = await asyncio.to_thread(lambda: [_decode_crop(b) for b in blobs])
    out: dict = {}
    for e, f in zip(todo, fields):
        n = want[e["param"]]
        st = str(e.get("step"))
        a, b = (int(x) for x in st.split("-")) if "-" in st else (int(st) - 12, int(st))
        out.setdefault(n, []).append(((a, b), f))
        lats, lons = f.lats, f.lons
    res = {"lats": lats, "lons": lons}
    save = {"lats": lats, "lons": lons}
    for n, lst in out.items():
        lst.sort(key=lambda x: x[0])
        res[n] = {"windows": [w for w, _ in lst], "values": np.stack([f.values for _, f in lst])}
        save[f"{n}__win"] = np.array(res[n]["windows"], dtype=np.int32)
        save[f"{n}__val"] = res[n]["values"].astype(np.float16)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".part.npz")
    np.savez_compressed(tmp, **save)
    tmp.replace(p)
    return res


def cleanup(keep: int = 6) -> None:
    """Keep the member crops of the newest ``keep`` runs (they feed the trend figures)."""
    if not ENS_CACHE.exists():
        return
    runs = sorted(d for d in ENS_CACHE.iterdir() if d.is_dir())
    for d in runs[:-keep]:
        shutil.rmtree(d, ignore_errors=True)
