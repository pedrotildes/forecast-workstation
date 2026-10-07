"""Daily maximum / minimum 2 m temperature from model output.

Day windows (UTC): Tmax(D) uses periods centred in [D 06Z, D 18Z];
Tmin(D) uses periods centred in [D−1 18Z, D 06Z]. Models provide:

* GFS        TMAX/TMIN over 6-h windows (steps multiple of 6)
* ECMWF IFS  mx2t3/mn2t3 (3-h windows) to +144 h, mx2t6/mn2t6 (6-h) afterwards
* ECMWF AIFS no extremes in open data → instantaneous 2 m temperature every 6 h
  (slightly underestimates Tmax / overestimates Tmin)
"""
from __future__ import annotations

import asyncio
import datetime as dt

import numpy as np

from .grib import Field
from .models import Domain, FieldUnavailable, Model

H = dt.timedelta(hours=1)


def window_hours(model: Model, step: int) -> int | None:
    """Length of the max/min window ending at ``step`` (0 = instantaneous sample)."""
    if not model.has("tmax"):
        return 0
    if model.id == "gfs":
        return 6 if step % 6 == 0 and step >= 6 else None
    if model.global_download:  # ECMWF IFS
        if step == 0:
            return None
        return 3 if step <= 144 else 6
    return None


def daily_plan(model: Model, run: dt.datetime, steps: list[int]) -> dict:
    """{date: {"tmax": [steps], "tmin": [steps], "tmax_ok": bool, "tmin_ok": bool}}"""
    plan: dict[dt.date, dict] = {}
    for s in steps:
        w = window_hours(model, s)
        if w is None:
            continue
        end = run + s * H
        mid = end - (w / 2) * H
        # Tmax: centre within [D 06Z, D 18Z]
        d = mid.date()
        if dt.datetime(d.year, d.month, d.day, 6) <= mid <= dt.datetime(d.year, d.month, d.day, 18):
            p = plan.setdefault(d, {"tmax": [], "tmin": [], "cov_x": 0, "cov_n": 0})
            p["tmax"].append(s)
            p["cov_x"] += w if w else 1
        # Tmin: centre within [D-1 18Z, D 06Z]
        dmin = (mid + 6 * H).date()
        lo = dt.datetime(dmin.year, dmin.month, dmin.day) - 6 * H
        hi = dt.datetime(dmin.year, dmin.month, dmin.day, 6)
        if lo <= mid <= hi:
            p = plan.setdefault(dmin, {"tmax": [], "tmin": [], "cov_x": 0, "cov_n": 0})
            p["tmin"].append(s)
            p["cov_n"] += w if w else 1
    inst = not model.has("tmax")
    for p in plan.values():
        need = 3 if inst else 12
        p["tmax_ok"] = p["cov_x"] >= need
        p["tmin_ok"] = p["cov_n"] >= need
    return dict(sorted(plan.items()))


def keys_for(model: Model, kind: str) -> tuple[str, int]:
    return (kind, 0) if model.has(kind) else ("t2m", 0)


def complete_days(plan: dict, kinds=("tmax", "tmin")) -> list[dt.date]:
    return [d for d, p in plan.items() if all(p[f"{k}_ok"] for k in kinds)]


async def daily_fields(model: Model, run: dt.datetime, domain: Domain,
                       kinds=("tmax", "tmin"), progress: dict | None = None,
                       max_days: int | None = None) -> dict:
    """Daily Tmax/Tmin grids (°C) for every complete day of the run."""
    steps = await model.list_steps(run)
    plan = daily_plan(model, run, steps)
    days = [d for d, p in plan.items() if all(p[f"{k}_ok"] for k in kinds)]
    if max_days:
        days = days[:max_days]
    if not days:
        raise FieldUnavailable(f"{model.name}: sem dias completos nesta run")
    need: dict[int, set] = {}
    for d in days:
        for k in kinds:
            for s in plan[d][k]:
                need.setdefault(s, set()).add(keys_for(model, k))

    got: dict[int, dict] = {}
    sem = asyncio.Semaphore(3 if model.global_download else 4)
    if progress is not None:
        progress.update(done=0, total=len(need), model=model.name)

    async def one(s, ks):
        async with sem:
            try:
                got[s] = await model.fetch(run, s, list(ks), domain)
            except FieldUnavailable:
                got[s] = {}
            if progress is not None:
                progress["done"] += 1

    await asyncio.gather(*(one(s, ks) for s, ks in need.items()))

    grid: Field | None = None
    out = {"days": [], "grid": None}
    for k in kinds:
        out[k] = {}
    key_t = {k: keys_for(model, k) for k in kinds}
    for d in days:
        okday = True
        vals = {}
        for k in kinds:
            arrs = [got[s][key_t[k]].values for s in plan[d][k] if key_t[k] in got.get(s, {})]
            if not arrs:
                okday = False
                break
            st = np.stack(arrs)
            vals[k] = (np.nanmax(st, 0) if k == "tmax" else np.nanmin(st, 0)) - 273.15
            if grid is None:
                grid = next(f for s in plan[d][k] for kk, f in got.get(s, {}).items())
        if okday:
            out["days"].append(d)
            for k in kinds:
                out[k][d] = vals[k].astype(np.float32)
    out["grid"] = grid
    return out
