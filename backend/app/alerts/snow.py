"""Snow («neve»): snow accumulated in one UTC day, by the IPMA altitude bands.

IPMA criteria («queda de neve com acumulação»): below 1000 m orange 6–15 cm, red > 15;
1000–1600 m orange 11–30, red > 30; above 1600 m orange 26–50, red > 50 (criteria.py).

Why the snow is estimated and not read from the models' snow cover: the 0.25° grid smooths
the mountains. No sample point of the districts is above ~950 m (the IFS puts the Serra da
Estrela at 865 m), so the model snowpack never reaches the 1000–1600 m and > 1600 m bands,
and in the lower band it sits below the altitudes where snow actually settles. So, at the
altitudes each district really has, snow is estimated from the precipitation and the
temperature of every grid point:

* temperature at altitude z: linear in height between the 2 m temperature (at the model
  orography of the point) and the 850 hPa temperature (taken at Z850 = 1450 m, typical of
  the cold air masses that bring snow); above 850 hPa, −6.5 °C/km;
* share of the precipitation that falls and settles as snow: 1 at T ≤ 0 °C, 0 at T ≥ 2 °C,
  linear in between (6-h windows, mean of the temperatures at the two ends);
* 1 cm of snow per mm of water (density 100 kg/m³);
* the value of a day is the largest snowfall in 24 h ending at 06, 12, 18 or 24 UTC of it.

Altitudes evaluated in each district (its representative top, ``TOPS``): the lower band at
min(top, 1000 m), the middle band at min(top, 1600 m) when the top is above 1000 m, the
upper band at the top when it is above 1600 m (Guarda: Serra da Estrela). The value of
a band is the largest estimate over the district's grid points. Orographic enhancement of
the precipitation is not modelled (it underestimates high ground); a cold pool at low
levels under warmer air aloft is partly caught through the 850 hPa temperature.

Areas: the 18 mainland districts, the mountains of Madeira (MRM) and the three groups of
the Azores (the Central group through Pico, whose upper slopes are the only ground there
above 1000 m). The Madeira coasts are left out: snow there would need the sea-level
temperature of a mid-latitude winter, which the island never gets.
"""
from __future__ import annotations

import asyncio
import datetime as dt
from functools import lru_cache

import numpy as np

from ..config import ALERTS_DIR
from ..models import MODELS, FieldUnavailable
from . import criteria as C
from .areas import PointSet, areas, district_geometry, land_points

H = dt.timedelta(hours=1)
Z850 = 1450.0            # m — height given to the 850 hPa temperature
GAMMA = 6.5e-3           # °C/m above 850 hPa
T_ALL, T_NONE = 0.0, 2.0  # °C — all snow / no snow
CM_PER_MM = 1.0          # cm of snow per mm of water
SCREEN_RATIO = 0.3       # IFS estimate ≥ 30 % of the orange threshold -> fetch the members
WINDOW = 6               # h

# Representative top of each area (m): the altitude above which it has at least 25 km² of
# ground, so that a summit of a few hectares on the border (the Marão in Porto, the Estrela
# in Castelo Branco) does not decide the warning of a whole district. From the Copernicus
# DEM GLO-90 (© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018, provided
# under COPERNICUS by the European Union and ESA) inside the Natural Earth district outlines
# (computed 2026-11-22; MRM: the whole island of Madeira; the Azores, computed 2026-11-23:
# the land of each group — Ocidental 157 km², summit 906 m; Central 1324 km², Pico 2308 m,
# 28 km² above 1000 m; Oriental 843 km², summit 1085 m).
TOPS = {
    "VCT": 1210, "BRG": 1200, "PTO": 980, "VRL": 1300, "BGC": 1180, "AVR": 940,
    "VIS": 1160, "GDA": 1670, "CBR": 1040, "CBO": 1440, "LRA": 760, "STM": 470, "PTG": 730,
    "LSB": 360, "STB": 270, "EVR": 460, "BJA": 460, "FAR": 560, "MRM": 1470,
    "AOC": 610, "ACE": 1030, "AOR": 760,
}


def band_alts(top: float) -> list[float]:
    out = [min(top, 1000.0)]
    if top > 1000:
        out.append(min(top, 1600.0))
    if top > 1600:
        out.append(float(top))
    return out


class SnowPoints(PointSet):
    """One point per (area, altitude band): ``alt[i]`` the altitude evaluated and
    ``nodes[i]`` the slice of the area's grid points in land_points() order."""

    def __init__(self):
        lp = land_points()
        entries, alts, nodes = [], [], []
        for a in areas():
            if a.code not in TOPS or a.code not in lp.slices:
                continue
            la, lo = a.label if a.group == "continente" else lp.label(a.code)
            for z in band_alts(TOPS[a.code]):
                entries.append((a.code, a.group, la, lo))
                alts.append(z)
                nodes.append(lp.slices[a.code])
        super().__init__(entries, "snow")
        self.alt = np.array(alts, float)
        self.nodes = nodes

    def index(self, code: str, alt: float) -> int:
        sl = self.slices[code]
        k = int(np.argmin(np.abs(self.alt[sl] - alt)))
        return sl.start + k


@lru_cache(maxsize=1)
def snow_points() -> SnowPoints:
    return SnowPoints()


def thresholds() -> np.ndarray:
    """(2, n_points) orange/red thresholds of the band of every point."""
    sp = snow_points()
    out = np.zeros((2, len(sp)), np.float32)
    for i, z in enumerate(sp.alt):
        t = C.snow(z)
        out[0, i], out[1, i] = t["laranja"], t["vermelho"]
    return out


# ---------------------------------------------------------------- the estimate
def t_at(z, t2, t8, zn):
    """Temperature (°C) at altitude z from the 2 m temperature at the orography zn and the
    850 hPa temperature (at Z850). Broadcasts over leading axes."""
    z = np.asarray(z, float)
    zn = np.asarray(zn, float)
    dz = Z850 - zn
    ok = dz > 200.0
    slope = np.where(ok, (t8 - t2) / np.where(ok, dz, 1.0), -GAMMA)
    below = t2 + slope * (z - zn)
    above = t8 - GAMMA * (z - Z850)
    return np.where(ok & (z > Z850), above, below)


def accumulate(P, T2, T8, zn, z):
    """Snow (cm) per window: P (W, …) precipitation of each window (mm); T2, T8 (W+1, …)
    temperatures (°C) at the window ends; zn orography, z altitude (broadcast to the last axes)."""
    tz = t_at(z, T2, T8, zn)
    tw = 0.5 * (tz[:-1] + tz[1:])
    f = np.clip((T_NONE - tw) / (T_NONE - T_ALL), 0.0, 1.0)
    return np.maximum(P, 0.0) * f * CM_PER_MM


def rolling24(acc: np.ndarray) -> np.ndarray:
    """Per-window snow (W, …) -> snow in the 24 h before the end of each window (W, …)."""
    c = np.concatenate([np.zeros_like(acc[:1]), np.cumsum(np.nan_to_num(acc), 0)], 0)
    idx = np.arange(1, acc.shape[0] + 1)
    return c[idx] - c[np.maximum(idx - 24 // WINDOW, 0)]


def points_from_nodes(P, T2, T8, zn, last: int = 4) -> np.ndarray:
    """Node fields (W[+1], M, N land nodes) -> snow in the 24 h before each of the ``last``
    window ends (the four 6-h steps of the day) at every snow point: (last, M, n_points), the
    largest over the area's nodes."""
    sp = snow_points()
    M = P.shape[1]
    out = np.full((last, M, len(sp)), np.nan, np.float32)
    for k in range(len(sp)):
        sl = sp.nodes[k]
        acc = accumulate(P[..., sl], T2[..., sl], T8[..., sl], zn[sl], sp.alt[k])   # (W, M, n)
        with np.errstate(all="ignore"):
            out[:, :, k] = np.nanmax(rolling24(acc)[-last:], -1)
    return out


@lru_cache(maxsize=8)
def _bands(sig) -> tuple[np.ndarray, np.ndarray]:
    """Altitudes evaluated at every grid node — those of its district (band_alts of its
    representative top; 1000 m outside the districts) — as (3, ny, nx) with NaN where a
    district has fewer bands, and their orange/red thresholds (2, 3, ny, nx)."""
    import shapely
    lat0, nlat, dlat, lon0, nlon, dlon = sig
    lats, lons = lat0 + np.arange(nlat) * dlat, lon0 + np.arange(nlon) * dlon
    LO, LA = np.meshgrid(lons, lats)
    zb = np.full((3,) + LA.shape, np.nan)
    zb[0] = 1000.0
    for code, top in TOPS.items():
        g = district_geometry(code)
        if g is None:
            continue
        m = shapely.contains_xy(g, LO, LA)
        for i, z in enumerate(band_alts(top)):
            zb[i][m] = z
        for i in range(len(band_alts(top)), 3):
            zb[i][m] = np.nan
    th = np.full((2,) + zb.shape, np.nan)
    for i in range(3):
        for j, lvl in enumerate(("laranja", "vermelho")):
            th[j, i] = np.where(np.isfinite(zb[i]),
                                np.vectorize(lambda z: C.snow(z)[lvl] if np.isfinite(z) else np.nan)(zb[i]),
                                np.nan)
    return zb, th


def _sig(lats, lons):
    return (round(float(lats[0]), 4), len(lats), round(float(lats[1] - lats[0]), 5),
            round(float(lons[0]), 4), len(lons), round(float(lons[1] - lons[0]), 5))


def band_grids(lats, lons) -> tuple[np.ndarray, np.ndarray]:
    return _bands(_sig(lats, lons))


def grid_snow(P, T2, T8, zn, lats, lons) -> tuple[np.ndarray, np.ndarray]:
    """Grids for the maps (…, ny, nx): the largest 24-h snow of the day at the representative
    top of each district (1000 m outside), and the level reached (0/1/2) at any of the
    district's altitudes. Sea (orography < 30 m) is NaN / 0."""
    zb, th = band_grids(lats, lons)
    top = lev = None
    for i in range(3):
        ok = np.isfinite(zb[i])
        if not ok.any():
            continue
        a = np.max(rolling24(accumulate(P, T2, T8, zn, np.where(ok, zb[i], 0.0)))[-4:], 0)
        li = ((a >= th[0, i]).astype(np.int8) + (a >= th[1, i]).astype(np.int8)) * ok
        lev = li if lev is None else np.maximum(lev, li)
        top = np.where(ok, a, np.nan) if top is None else np.where(ok, a, top)
    sea = zn < 30.0                      # sea (the IFS orography is not exactly 0 offshore)
    top = np.where(sea, np.nan, top).astype(np.float32)
    lev = np.where(sea, 0, lev).astype(np.int8)
    return top, lev


# ---------------------------------------------------------------- orography
async def orography(mid: str):
    """(lats, lons, metres) of the orography a model's 2 m temperature refers to (AIFS and
    the ensemble: the IFS one). Cached in ALERTS_DIR."""
    src = "ecmwf" if mid in ("ecmwf", "aifs", "ens") else mid
    p = ALERTS_DIR / f"orog_{src}.npz"
    if p.exists():
        z = np.load(p)
        return z["lats"], z["lons"], z["values"]
    from .detect import DOMAIN
    m = MODELS[src]
    for run in (await m.list_runs())[:3]:
        try:
            f = (await m.fetch(run, 0, [("orog", 0)], DOMAIN))[("orog", 0)]
            break
        except (FieldUnavailable, KeyError):
            continue
    else:
        raise FieldUnavailable(f"{src}: orografia indisponível")
    vals = f.values.astype(np.float32)
    if src == "ecmwf" and np.nanmax(vals) > 20000:      # still geopotential
        vals = vals / 9.80665
    np.savez_compressed(p, lats=f.lats, lons=f.lons, values=vals)
    return f.lats, f.lons, vals


def on_grid(src_lats, src_lons, vals, lats, lons) -> np.ndarray:
    """A field on a grid with the same spacing (e.g. the IFS orography on the ENS crop)."""
    j = np.clip(np.searchsorted(src_lats, np.asarray(lats) - 1e-6), 0, len(src_lats) - 1)
    i = np.clip(np.searchsorted(src_lons, np.asarray(lons) - 1e-6), 0, len(src_lons) - 1)
    return vals[np.ix_(j, i)]


# ---------------------------------------------------------------- windows of a day
def bounds(run: dt.datetime, day: dt.date) -> list[int] | None:
    """Steps every 6 h from 18 h before the day (or the run, if later) to its end, so the snow
    of the 24 h before each of 06, 12, 18 and 24 UTC can be added up. An overnight snowfall
    (IPMA warnings often run 18 h–09 h) is not split in two by midnight. None if the day
    starts before the run."""
    t0 = dt.datetime(day.year, day.month, day.day)
    s = [int((t0 + k * WINDOW * H - run) / H) for k in range(-3, 5)]
    s = [x for x in s if x >= 0]
    return s if len(s) >= 5 else None


# ---------------------------------------------------------------- ensemble members
async def ens_daily(run: dt.datetime, days: list[dt.date], log) -> dict:
    """Per day: the largest 24-h snow ending on the day for every member at every snow point
    (and at each of the day's four 6-h steps, for the plume), and on the grid at the map
    altitudes of each district (maps)."""
    from .detect import _members_many, sampler_for
    need = sorted({s for d in days if (b := bounds(run, d)) for s in b})
    log(f"  neve: {len(days)} dia(s), {len(need)} passos × 50 membros (precipitação, T2m, T850)")
    tp = await _members_many(run, "tp", [s for s in need if s > 0])
    t2 = await _members_many(run, "t2m", need)
    t8 = await _members_many(run, "t850", need)
    if not t2:
        return {}
    ref = t2[min(t2)]
    lats, lons = ref["lats"], ref["lons"]
    sm = sampler_for(lats, lons)
    ola, olo, ov = await orography("ens")
    zn_grid = on_grid(ola, olo, ov, lats, lons)
    zn = sm.points(zn_grid)
    out = {}
    for d in days:
        b = bounds(run, d)
        if not b or any(s not in t2 or s not in t8 for s in b) or any(s > 0 and s not in tp for s in b):
            continue
        acc = [tp[s]["values"] if s > 0 else np.zeros_like(t2[s]["values"]) for s in b]
        P = np.stack([np.maximum(acc[k + 1] - acc[k], 0.0) for k in range(len(b) - 1)])  # (W, 50, ny, nx)
        T2 = np.stack([t2[s]["values"] for s in b])                                     # (W+1, 50, ny, nx)
        T8 = np.stack([t8[s]["values"] for s in b])
        series = points_from_nodes(sm.points(P), sm.points(T2), sm.points(T8), zn)       # (4, 50, npts)
        grid, levgrid = grid_snow(P, T2, T8, zn_grid, lats, lons)                      # (50, ny, nx)
        out[d] = {"pts": np.nanmax(series, 0), "series": series, "steps": b[-4:], "grid": grid,
                  "levgrid": levgrid, "lats": lats, "lons": lons}
    return out


# ---------------------------------------------------------------- deterministic models
DET = ("ecmwf", "aifs", "gfs", "icon_eu")
KEYS = [("tp", 0), ("t2m", 0), ("t", 850)]


async def _fields(m, run, steps):
    from .detect import DOMAIN
    sem = asyncio.Semaphore(4)
    got: dict = {}

    async def one(s):
        async with sem:
            try:
                got[s] = await m.fetch(run, s, KEYS, DOMAIN)
            except FieldUnavailable:
                pass

    await asyncio.gather(*(one(s) for s in steps))
    return got


def _day_arrays(got, b, sample):
    """(P (W, …), T2 (W+1, …), T8 (W+1, …)) in mm / °C, or None if a step is missing."""
    if any(s not in got or any(k not in got[s] for k in KEYS) for s in b):
        return None
    tp = [sample(got[s][("tp", 0)].values) for s in b]
    P = np.stack([np.maximum(tp[k + 1] - tp[k], 0.0) for k in range(len(b) - 1)])
    T2 = np.stack([sample(got[s][("t2m", 0)].values) - 273.15 for s in b])
    T8 = np.stack([sample(got[s][("t", 850)].values) - 273.15 for s in b])
    return P, T2, T8


async def det_daily(mid: str, run: dt.datetime, days: list[dt.date]) -> dict:
    """{day: (n_points,) snow (cm)} of one deterministic model."""
    from .detect import sampler_for
    m = MODELS[mid]
    steps = await m.list_steps(run)
    if not steps:
        return {}
    want = {d: b for d in days if (b := bounds(run, d)) and b[-1] <= steps[-1]
            and all(s in steps for s in b)}
    if not want:
        return {}
    got = await _fields(m, run, sorted({s for b in want.values() for s in b}))
    if not got:
        return {}
    f0 = got[next(iter(got))][("t2m", 0)]
    sm = sampler_for(f0.lats, f0.lons)
    ola, olo, ov = await orography(mid)
    zn = sampler_for(ola, olo).points(ov)
    out = {}
    for d, b in want.items():
        arr = _day_arrays(got, b, sm.points)
        if arr is None:
            continue
        P, T2, T8 = (x[:, None, :] for x in arr)                # one "member"
        out[d] = np.nanmax(points_from_nodes(P, T2, T8, zn)[:, 0], 0)
    return out


async def deterministic(days: list[dt.date], log) -> dict:
    res = {}
    for mid in DET:
        m = MODELS[mid]
        try:
            from .detect import det_run
            run = await det_run(m)
            vals = await det_daily(mid, run, days)
        except Exception as e:  # noqa: BLE001 — a model that fails must not stop the alerts
            log(f"  {mid} (neve): {e}")
            continue
        if vals:
            res[mid] = {"run": run.isoformat(), "name": m.name, "days": vals}
    return res


async def det_grid(mid: str, day: dt.date):
    """Snow of the day on a deterministic model's grid at the map altitude (figures)."""
    from .detect import det_run
    m = MODELS[mid]
    run = await det_run(m)
    b = bounds(run, day)
    steps = await m.list_steps(run)
    if not b or not steps or b[-1] > steps[-1] or any(s not in steps for s in b):
        return None
    got = await _fields(m, run, b)
    arr = _day_arrays(got, b, lambda v: v)
    if arr is None:
        return None
    f0 = got[b[0]][("t2m", 0)]
    ola, olo, ov = await orography(mid)
    zn = on_grid(ola, olo, ov, f0.lats, f0.lons)
    P, T2, T8 = arr
    grid, _ = grid_snow(P, T2, T8, zn, f0.lats, f0.lons)
    return f0.lats, f0.lons, grid, run


# ---------------------------------------------------------------- screening (IFS)
async def screen(run: dt.datetime, days: list[dt.date], out: dict, log) -> list[dt.date]:
    """Days worth the ensemble members: the IFS estimate reaches ≥ 30 % of the orange threshold
    at some point (± one day, for timing errors). A cheap temperature test first: if even at
    the top of every district the IFS minimum temperature stays above 3 °C, no snow."""
    from ..daily import daily_fields
    from .detect import DOMAIN, sampler_for
    ifs = MODELS["ecmwf"]
    irun = await ifs.best_run()
    dd = await daily_fields(ifs, irun, DOMAIN, kinds=("tmin",))
    sm = sampler_for(dd["grid"].lats, dd["grid"].lons)
    ola, olo, ov = await orography("ecmwf")
    zn = sampler_for(ola, olo).points(ov)
    lp = land_points()
    tops = np.array([TOPS.get(c, np.nan) for c in lp.codes], float)
    cold = []
    for d in days:
        if d not in dd["tmin"]:
            continue
        t_top = sm.points(dd["tmin"][d]) - GAMMA * (tops - zn)          # °C at the top
        if np.nanmin(t_top) <= 3.0:
            cold.append(d)
    ratio = {d.isoformat(): 0.0 for d in days}
    if cold:
        vals = await det_daily("ecmwf", irun, cold)
        thr = thresholds()[0]
        for d, v in vals.items():
            ratio[d.isoformat()] = round(float(np.nanmax(v / thr)), 2) if np.isfinite(v).any() else 0.0
    out["screen"]["neve"] = ratio
    flagged = set()
    for d in days:
        if ratio.get(d.isoformat(), 0.0) >= SCREEN_RATIO:
            flagged |= {d - dt.timedelta(days=1), d, d + dt.timedelta(days=1)}
    return sorted(x for x in flagged if x in days)


# ---------------------------------------------------------------- summary of an area
def band_summary(code: str, pts: np.ndarray, lev: np.ndarray, thr: np.ndarray) -> tuple[int, list]:
    """Per altitude band of an area: probabilities and member statistics; and the index of the
    decisive band (highest chance of orange, then of red, then closest to the threshold)."""
    sp = snow_points()
    sl = sp.slices[code]
    rows, keys = [], []
    for k in range(sl.start, sl.stop):
        v = pts[:, k]
        q = np.nanpercentile(v, [10, 50, 90])
        po, pr = float(np.mean(lev[:, k] >= 1)), float(np.mean(lev[:, k] >= 2))
        rows.append({"z": int(round(sp.alt[k])), "p_o": round(po, 3), "p_r": round(pr, 3),
                     "p10": round(float(q[0]), 1), "p50": round(float(q[1]), 1),
                     "p90": round(float(q[2]), 1), "max": round(float(np.nanmax(v)), 1),
                     "thr_o": float(thr[0, k]), "thr_r": float(thr[1, k])})
        keys.append((po, pr, float(q[1] / thr[0, k])))
    best = max(range(len(rows)), key=lambda i: keys[i])
    return sl.start + best, rows
