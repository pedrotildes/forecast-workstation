"""Sea state («agitação marítima»): significant wave height Hs off each coastal area.

Warning areas: the mainland districts with a coast, Madeira (north coast, south coast,
Porto Santo) and the three Azores groups. Each has a few points 10–25 km offshore; the
wave grids have no values over land and near the coast, so a point reads the nearest grid
cell that has a value (within 0.6°), not a bilinear interpolation.

IPMA criteria: orange 5–7 m, red > 7 m; Azores 7–9 m, red > 9 m; south coast of the
Algarve with south-easterly swell 3–5 m, red > 5 m (here: mean wave direction from
90°–180° at the points east of Cape São Vicente).

Ensemble: the 50 members of the ECMWF wave ensemble ("waef"), 6-hourly. Deterministic:
the IFS wave model (ECMWF Open Data, "wave" stream) and GFS-Wave (NOAA NOMADS).
"""
from __future__ import annotations

import asyncio
import datetime as dt
from functools import lru_cache

import httpx
import numpy as np

from ..config import CACHE_DIR, HTTP_TIMEOUT, USER_AGENT
from ..grib import decode_message
from ..models.base import FieldUnavailable, cycle_floor, utcnow
from ..models.ecmwf import _ECMWFBase
from . import criteria as C
from . import ens
from .areas import BOX, PointSet

H = dt.timedelta(hours=1)
WAVE_CACHE = CACHE_DIR / "alerts" / "waves"

# code -> (group, [(lat, lon), …]) — offshore points
SEA = {
    "VCT": ("continente", [(41.70, -9.05), (41.90, -9.05)]),
    "BRG": ("continente", [(41.50, -9.00)]),
    "PTO": ("continente", [(41.15, -8.90), (41.35, -8.95)]),
    "AVR": ("continente", [(40.65, -8.95), (40.90, -8.90)]),
    "CBR": ("continente", [(40.15, -9.05)]),
    "LRA": ("continente", [(39.75, -9.20), (39.40, -9.60)]),
    "LSB": ("continente", [(39.05, -9.60), (38.70, -9.60)]),
    "STB": ("continente", [(38.40, -9.35), (37.95, -9.05)]),
    "BJA": ("continente", [(37.55, -9.00)]),
    "FAR": ("continente", [(37.20, -9.10), (36.85, -8.70), (36.85, -8.10), (36.90, -7.60)]),
    "MCN": ("madeira", [(32.95, -17.15), (32.92, -16.85)]),
    "MCS": ("madeira", [(32.55, -17.15), (32.55, -16.85), (32.62, -16.60)]),
    "MPS": ("madeira", [(32.98, -16.45), (33.15, -16.25)]),
    "AOC": ("acores", [(39.50, -31.40), (39.25, -31.10)]),
    "ACE": ("acores", [(38.45, -28.85), (38.80, -28.40), (38.30, -28.20), (39.20, -28.20),
                       (38.55, -27.05)]),
    "AOR": ("acores", [(37.60, -25.65), (37.95, -25.45), (36.85, -25.30)]),
}
SE_SWELL = {"laranja": 3.0, "vermelho": 5.0}     # south Algarve with south-easterly swell
SE_DIR = (90.0, 180.0)


@lru_cache(maxsize=1)
def sea_points() -> PointSet:
    return PointSet([(c, g, la, lo) for c, (g, pts) in SEA.items() for la, lo in pts], "sea")


def south_algarve() -> np.ndarray:
    ps = sea_points()
    return np.array([c == "FAR" and lo > -8.95 for c, lo in zip(ps.codes, ps.lons)])


def thresholds() -> np.ndarray:
    """(2, n_points) general thresholds (the south-easterly swell rule is applied per member)."""
    ps = sea_points()
    out = np.zeros((2, len(ps)), np.float32)
    for i, g in enumerate(ps.groups):
        t = C.waves(g)
        out[0, i], out[1, i] = t["laranja"], t["vermelho"]
    return out


# ---------------------------------------------------------------- nearest valid cell
class Nearest:
    """Nearest grid cell with a value (within ``max_deg``) for every sea point."""

    def __init__(self, lats, lons, valid: np.ndarray, max_deg: float = 0.6):
        ps = sea_points()
        LA, LO = np.meshgrid(np.asarray(lats, float), np.asarray(lons, float), indexing="ij")
        vj, vi = np.nonzero(valid)
        self.j = np.full(len(ps), -1)
        self.i = np.full(len(ps), -1)
        for k, (la, lo) in enumerate(zip(ps.lats, ps.lons)):
            if not len(vj):
                break
            d = np.hypot(LA[vj, vi] - la, (LO[vj, vi] - lo) * np.cos(np.radians(la)))
            m = int(np.argmin(d))
            if d[m] <= max_deg:
                self.j[k], self.i[k] = vj[m], vi[m]

    def points(self, arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, np.float32)
        out = a[..., np.maximum(self.j, 0), np.maximum(self.i, 0)].copy()
        out[..., self.j < 0] = np.nan
        return out


_NEAREST: dict = {}


def nearest_for(source: str, lats, lons, sample: np.ndarray) -> Nearest:
    """Cached per source (the land mask differs between wave models)."""
    key = (source, round(float(lats[0]), 3), len(lats), round(float(lons[0]), 3), len(lons))
    if key not in _NEAREST:
        valid = np.isfinite(sample)
        while valid.ndim > 2:
            valid = valid.all(0)
        _NEAREST[key] = Nearest(lats, lons, valid)
    return _NEAREST[key]


# ---------------------------------------------------------------- screening (ep file)
async def screen(run: dt.datetime, days: list[dt.date], out: dict, screen_p: float) -> list[dt.date]:
    """Days on which some sea point has a chance of Hs above (roughly) its orange threshold:
    P(Hs > 4 m) mainland/Madeira, P(Hs > 6 m) Azores, P(Hs > 2 m) south Algarve."""
    ep = await ens.ep_fields(run, ["p_swh2", "p_swh4", "p_swh6"])
    ps = sea_points()
    ref = ep["p_swh4"]["values"][0]
    nr = nearest_for("waef", ep["lats"], ep["lons"], ref)
    az = np.array([g == "acores" for g in ps.groups])
    sa = south_algarve()
    res = []
    for d in days:
        t0 = dt.datetime(d.year, d.month, d.day)
        best = 0.0
        for k, (a, b) in enumerate(ep["p_swh4"]["windows"]):
            v = run + b * H                       # instantaneous probability at the step
            if not (t0 < v <= t0 + 24 * H):
                continue
            p4 = nr.points(ep["p_swh4"]["values"][k])
            p6 = nr.points(ep["p_swh6"]["values"][k])
            p2 = nr.points(ep["p_swh2"]["values"][k])
            p = np.where(az, p6, np.where(sa, np.maximum(p2 * 0.5, p4), p4))
            best = max(best, float(np.nanmax(p)) if np.isfinite(p).any() else 0.0)
        out["screen"].setdefault("p_swh", {})[d.isoformat()] = round(best, 1)
        if best >= screen_p:
            res.append(d)
    return res


# ---------------------------------------------------------------- ensemble members
def _steps_day(run: dt.datetime, day: dt.date) -> list[int]:
    t0 = dt.datetime(day.year, day.month, day.day)
    return [s for s in ens.WAVE_MEMBERS.nominal_steps(run)
            if s > 0 and s % 6 == 0 and t0 < run + s * H <= t0 + 24 * H]


async def ens_daily(run: dt.datetime, days: list[dt.date], log) -> dict:
    """Per day: daily max Hs of every member at every sea point, the level reached (with the
    south-easterly swell rule in the Algarve), the series and the grid for the figures."""
    from .detect import _members_many
    steps = sorted({s for d in days for s in _steps_day(run, d)})
    log(f"  mar: {len(days)} dia(s), {len(steps)} passos × 50 membros")
    swh = await _members_many(run, "swh", steps)
    if not swh:
        return {}
    first = swh[min(swh)]
    nr = nearest_for("waef", first["lats"], first["lons"], first["values"][0])
    thr = thresholds()
    sa = south_algarve()
    # mean direction only where it can matter: Hs ≥ 2.5 m at a south Algarve point
    need_dir = [s for s in swh if np.nanmax(nr.points(swh[s]["values"])[:, sa], initial=0) >= 2.5]
    mwd = await _members_many(run, "mwd", need_dir) if need_dir else {}
    out = {}
    for d in days:
        st = [s for s in _steps_day(run, d) if s in swh]
        if not st:
            continue
        series, levs, se_at = [], [], []
        for s in st:
            v = nr.points(swh[s]["values"])                                  # (50, npts)
            lev = (v >= thr[0]).astype(np.int8) + (v >= thr[1]).astype(np.int8)
            se = np.zeros_like(v, dtype=bool)
            if s in mwd:
                dirs = nr.points(mwd[s]["values"])
                se = (dirs >= SE_DIR[0]) & (dirs <= SE_DIR[1]) & sa[None, :]
                lev_se = (v >= SE_SWELL["laranja"]).astype(np.int8) + (v >= SE_SWELL["vermelho"]).astype(np.int8)
                lev = np.where(se, np.maximum(lev, lev_se), lev)
            series.append(v)
            levs.append(lev)
            se_at.append(se)
        series = np.stack(series)                                           # (steps, 50, npts)
        k = np.nanargmax(np.where(np.isfinite(series), series, -1), 0)      # step of the daily max
        se_max = np.take_along_axis(np.stack(se_at), k[None], 0)[0]         # SE swell at that step
        thr_day = thr.copy()
        frac = se_max.mean(0)
        thr_day[:, sa & (frac >= 0.5)] = [[SE_SWELL["laranja"]], [SE_SWELL["vermelho"]]]
        out[d] = {"pts": np.nanmax(series, 0), "lev": np.stack(levs).max(0), "series": series,
                  "steps": st, "thr": thr_day, "se_frac": frac,
                  "grid": np.nanmax(np.stack([swh[s]["values"] for s in st]), 0),
                  "lats": first["lats"], "lons": first["lons"]}
    return out


# ---------------------------------------------------------------- deterministic models
class IfsWave(_ECMWFBase):
    id = "ifs_wave"
    name = "ECMWF IFS (ondas)"
    path_model = "ifs"
    levels = []
    run_every = 12

    def stream(self, run):
        return "wave"

    def nominal_steps(self, run):
        return list(range(0, 144, 3)) + list(range(144, 361, 6))


IFS_WAVE = IfsWave()


async def _ifs_field(run, step):
    p = WAVE_CACHE / "ifs" / run.strftime("%Y%m%d%H") / f"{step:03d}.npz"
    if p.exists():
        z = np.load(p)
        return z["lats"], z["lons"], z["values"]
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT}) as c:
        url, entries = await IFS_WAVE._index(c, run, step)
        e = next((e for e in entries if e.get("param") == "swh"), None)
        if e is None:
            raise FieldUnavailable("IFS ondas: sem swh")
        buf = await IFS_WAVE._range(c, url, e)
    _, f = decode_message(buf, BOX)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, lats=f.lats, lons=f.lons, values=f.values)
    return f.lats, f.lons, f.values


GFSW = ("https://nomads.ncep.noaa.gov/cgi-bin/filter_gfswave.pl?dir=%2Fgfs.{d}%2F{h}%2Fwave%2Fgridded"
        "&file=gfswave.t{h}z.global.0p25.f{s:03d}.grib2&var_HTSGW=on&subregion="
        "&toplat={lat1}&leftlon={lon0}&rightlon={lon1}&bottomlat={lat0}")


async def _gfs_field(run, step):
    p = WAVE_CACHE / "gfs" / run.strftime("%Y%m%d%H") / f"{step:03d}.npz"
    if p.exists():
        z = np.load(p)
        return z["lats"], z["lons"], z["values"]
    from ..models.gfs import _throttle
    url = GFSW.format(d=run.strftime("%Y%m%d"), h=run.strftime("%H"), s=step, lon0=BOX[0], lon1=BOX[1],
                      lat0=BOX[2], lat1=BOX[3])
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT}) as c:
        for attempt in range(3):
            await _throttle()
            r = await c.get(url)
            if r.status_code == 200 and r.content[:4] == b"GRIB":
                break
            if r.status_code == 404:
                raise FieldUnavailable("GFS-Wave: passo indisponível")
            await asyncio.sleep(3 * (attempt + 1))
        else:
            raise FieldUnavailable("GFS-Wave: falhou")
    _, f = decode_message(r.content, None)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, lats=f.lats, lons=f.lons, values=f.values)
    return f.lats, f.lons, f.values


async def _gfs_run() -> dt.datetime | None:
    """Newest GFS-Wave cycle already published to +120 h."""
    base = cycle_floor(utcnow(), 6)
    for k in range(5):
        r = base - dt.timedelta(hours=6 * k)
        try:
            await _gfs_field(r, 120)
            return r
        except FieldUnavailable:
            continue
    return None


async def _ifs_run() -> dt.datetime | None:
    """Newest 00/12Z IFS wave run already published to +144 h."""
    for r in (await IFS_WAVE.list_runs())[:4]:
        if r.hour not in (0, 12):
            continue
        async with httpx.AsyncClient(timeout=20, headers={"User-Agent": USER_AGENT}) as c:
            if await IFS_WAVE._exists(c, r, 144):
                return r
    return None


async def _daily(get, run, days) -> dict:
    out = {}
    for d in days:
        t0 = dt.datetime(d.year, d.month, d.day)
        steps = [int((t0 + h * H - run) / H) for h in (6, 12, 18, 24)]
        if steps[0] <= 0:
            continue
        vals = []
        for s in steps:
            try:
                lats, lons, v = await get(run, s)
            except FieldUnavailable:
                continue
            vals.append((lats, lons, v))
        if not vals:
            continue
        lats, lons, _ = vals[0]
        stack = np.stack([v for _, _, v in vals])
        out[d] = (lats, lons, np.nanmax(stack, 0))
    return out


async def deterministic(days: list[dt.date], log) -> dict:
    """{model: {"run", "name", "days": {day: (n_points,)}}} for IFS waves and GFS-Wave."""
    res = {}
    if not days:
        return res
    for mid, name, get, find in (("ifs_wave", "IFS (ondas)", _ifs_field, _ifs_run),
                                 ("gfs_wave", "GFS-Wave", _gfs_field, _gfs_run)):
        try:
            run = await find()
            if run is None:
                log(f"  {name}: sem run disponível")
                continue
            daily = await _daily(get, run, days)
        except Exception as e:  # noqa: BLE001
            log(f"  {name}: {e}")
            continue
        vals = {}
        for d, (lats, lons, arr) in daily.items():
            nr = nearest_for(mid, lats, lons, arr)
            vals[d] = nr.points(arr)
        if vals:
            res[mid] = {"run": run.isoformat(), "name": name, "days": vals}
    return res


async def det_grid(mid: str, day: dt.date):
    """Daily max Hs grid of a deterministic wave model (for the figures)."""
    get, find = (_ifs_field, _ifs_run) if mid == "ifs_wave" else (_gfs_field, _gfs_run)
    run = await find()
    if run is None:
        return None
    daily = await _daily(get, run, [day])
    if day not in daily:
        return None
    lats, lons, arr = daily[day]
    return lats, lons, arr.astype(np.float32), run
