"""Mean wind in the Azores («velocidade média do vento», km/h): IPMA orange 75–87 km/h, red
> 87 km/h, besides the gusts (orange 101–130, red > 130). Only the Azores have it.

It is part of the wind warning, as IPMA issues it: in an Azores area a member reaches a level
by its gust or by its mean wind, whichever is higher. The value of a member at a point is the
largest 10-m wind of the UTC day at the 3-hourly steps of the ECMWF ENS (6-hourly after
+144 h): the model's 10-m wind is the mean wind the criterion refers to (10-minute mean at the
stations). In practice the gust decides: in the three Azores storms checked (19–20/01/2025,
11/03/2025, 23–24/02/2026) the chance of orange by the mean wind was always below the one by
the gusts (e.g. 58 % against 94 % in the Western group on 20/01/2025).

Downloads. The 10-m wind components of a member cost ~1.7 MB per step, so they are fetched only
where they can change the level. The model wind at a step never exceeds the maximum gust of the
window that ends at that step (the gust is the wind plus a turbulent part); over the Azores the
smallest ratio seen was 1.09, and a mean wind ≥ 75 km/h always came with gusts ≥ 93 km/h at the
same point (4973 member-points with wind ≥ 55 km/h in those storms). So the components are fetched
for the members whose gust reaches GUST_PRE at an Azores point at that step, and only on the days
when at least WATCH_P of the members do so in some group (otherwise not even the combined level
could reach the watch probability).
"""
from __future__ import annotations

import asyncio
import datetime as dt
from functools import lru_cache

import numpy as np

from . import criteria as C
from .areas import land_points

GUST_PRE = 82.0            # km/h — 75 km/h × the smallest gust/wind ratio seen (1.09)
GUST_FACTOR = 1.3          # typical gust / mean wind over the sea (estimates for the figures only)


@lru_cache(maxsize=1)
def az_index() -> np.ndarray:
    """Indices of the Azores points in land_points()."""
    return np.array([i for i, g in enumerate(land_points().groups) if g == "acores"])


def thresholds() -> np.ndarray:
    n = len(az_index())
    t = C.MEAN_WIND_AZORES
    return np.stack([np.full(n, t["laranja"]), np.full(n, t["vermelho"])]).astype(np.float32)


async def add_mean(run: dt.datetime, per_day: dict, gust: dict, log) -> None:
    """Add to each day of ``per_day`` (detect._ens_wind) "vm": (50, n_az) the largest mean wind
    of every member at the Azores points (NaN where it was not needed: below the orange
    threshold) and "vm_est": the same with an estimate there (figures, member statistics)."""
    from . import ens
    from .detect import WATCH_P, sampler_for
    if not gust:
        return
    ref = next(iter(gust.values()))
    sm = sampler_for(ref["lats"], ref["lons"])
    iaz = az_index()
    groups = {}
    for k, c in enumerate(np.array(land_points().codes)[iaz]):
        groups.setdefault(c, []).append(k)
    gpts = {s: sm.points(g["values"])[:, iaz] for s, g in gust.items()}          # (50, n_az)
    want, days_on = {}, []
    for d, v in per_day.items():
        st = [s for s in v["steps"] if s in gpts]
        hit = np.stack([gpts[s] >= GUST_PRE for s in st]).any(0)                   # (50, n_az)
        frac = max(float(np.mean(hit[:, ix].any(1))) for ix in groups.values())
        if frac >= WATCH_P:
            days_on.append(d)
            for s in st:
                w = (gpts[s] >= GUST_PRE).any(1)
                want[s] = want.get(s, np.zeros(50, bool)) | w
    need = [s for s in sorted(want) if want[s].any()]
    if need:
        log(f"  vento médio (Açores): {len(days_on)} dia(s), u/v a 10 m para "
            f"{sum(int(want[s].sum()) for s in need)} membro×passo(s)")
    sem = asyncio.Semaphore(4)

    async def one(s, nm):
        async with sem:
            return s, nm, await ens.members_some(run, nm, s, want[s])

    res = await asyncio.gather(*(one(s, nm) for s in need for nm in ("u10", "v10")))
    uv: dict = {}
    for s, nm, r in res:
        if r is not None:
            uv.setdefault(s, {})[nm] = sm.points(r["values"])[:, iaz]
    cap = C.MEAN_WIND_AZORES["laranja"] - 0.1
    for d, v in per_day.items():
        st = [s for s in v["steps"] if s in gpts]
        true, est = [], []
        for s in st:
            w = (np.hypot(uv[s]["u10"], uv[s]["v10"]) * 3.6 if s in uv and len(uv[s]) == 2
                 else np.full_like(gpts[s], np.nan))
            true.append(w)
            est.append(np.where(np.isfinite(w), w, np.minimum(gpts[s] / GUST_FACTOR, cap)))
        with np.errstate(all="ignore"):
            v["vm"] = np.nanmax(np.stack(true), 0)
            v["vm_est"] = np.nanmax(np.stack(est), 0)


def merge_levels(lev: np.ndarray, vm: np.ndarray) -> np.ndarray:
    """(50, land points) levels by the gust -> the higher of that and the level by the mean
    wind at the Azores points."""
    from .detect import levels
    out = lev.copy()
    iaz = az_index()
    out[:, iaz] = np.maximum(out[:, iaz], levels(vm, thresholds(), "vento"))
    return out


def annotate(rows: dict, vm_est: np.ndarray, vm: np.ndarray) -> None:
    """Mean-wind statistics of each Azores area of one day: rows[code]["vm"]."""
    iaz = az_index()
    codes = np.array(land_points().codes)[iaz]
    t = C.MEAN_WIND_AZORES
    for code in set(codes):
        if code not in rows:
            continue
        ix = np.nonzero(codes == code)[0]
        with np.errstate(all="ignore"):
            top = np.nanmax(vm_est[:, ix], 1)
            true = np.nanmax(vm[:, ix], 1)
        q = np.nanpercentile(top, [10, 50, 90])
        rows[code]["vm"] = {"p_o": round(float(np.mean(true >= t["laranja"])), 3),
                            "p_r": round(float(np.mean(true >= t["vermelho"])), 3),
                            "p10": round(float(q[0]), 1), "p50": round(float(q[1]), 1),
                            "p90": round(float(q[2]), 1), "max": round(float(np.nanmax(top)), 1),
                            "thr_o": t["laranja"], "thr_r": t["vermelho"]}
