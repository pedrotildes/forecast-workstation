"""Point time series across all forecast steps of a run (meteogram data)."""
from __future__ import annotations

import asyncio
import datetime as dt
import time

import numpy as np

from .models import Domain, FieldUnavailable, Model
from .models.base import TTLCache

SURFACE = ["t2m", "d2m", "u10", "v10", "gust", "msl", "sp", "tp", "tcc", "lcc", "mcc", "hcc",
           "cape", "mucape", "mlcape", "orog"]
UPPER_COMMON = [("t", 850), ("gh", 500)]
TH_LEVELS = [1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400,
             350, 300, 250, 200]

PROGRESS: dict[str, dict] = {}
_SERIES = TTLCache(1800)


def series_steps(model: Model, steps: list[int], hours: int) -> list[int]:
    """Meteogram time sampling: dense early, coarser later (keeps downloads sane)."""
    out = []
    for s in steps:
        if s > hours:
            break
        if model.id == "gfs":
            ok = s % 3 == 0 if s <= 240 else s % 6 == 0
        elif model.id == "ecmwf":
            ok = s % 3 == 0 if s <= 72 else s % 6 == 0
        elif model.id == "icon_eu":
            ok = s % 3 == 0
        else:
            ok = True
        if ok:
            out.append(s)
    return out


def has_profile(model: Model) -> bool:
    """Time-height sections are offered where the point download is cheap (GFS tiles)."""
    return not model.global_download


async def resolve_run(model: Model, valid: dt.datetime) -> tuple[dt.datetime, int] | None:
    """Newest run whose available steps contain ``valid``."""
    for run in await model.list_runs():
        s = (valid - run).total_seconds() / 3600
        if s < 0 or s != int(s):
            continue
        if int(s) in await model.list_steps(run):
            return run, int(s)
    return None


async def point_series(model: Model, run: dt.datetime, lat: float, lon: float, hours: int,
                       profile: bool = False, job: str | None = None) -> dict:
    ck = (model.id, run, round(lat, 3), round(lon, 3), hours, profile)
    cached = _SERIES.get(ck)
    if cached is not None:
        return cached
    if not model.covers(lat, lon):
        raise FieldUnavailable(model.outside_msg())
    steps = series_steps(model, await model.list_steps(run), hours)
    if not steps:
        raise FieldUnavailable(f"{model.name}: sem passos disponíveis para {run:%Y%m%d%H}")

    keys = [(n, 0) for n in SURFACE if model.has(n) and (profile or n != "sp")]
    keys += [k for k in UPPER_COMMON if model.has(*k)]
    hum = "r" if model.has("r") else "q"
    lv = [p for p in TH_LEVELS if p in model.levels]
    if profile:
        keys += [(n, p) for p in lv for n in ("t", hum, "u", "v")]
    keys = list(dict.fromkeys(keys))
    dom = Domain.tile(lat, lon)

    prog = PROGRESS.setdefault(job or "_", {})
    prog.update(done=0, total=len(steps), model=model.name, t0=time.time())
    vals: dict[int, dict] = {}
    sem = asyncio.Semaphore(3 if model.global_download else 4)

    async def one(st):
        async with sem:
            ks = keys if st == steps[0] else [k for k in keys if k[0] != "orog"]
            try:
                f = await model.fetch(run, st, ks, dom)
                vals[st] = {k: fld.at(lat, lon) for k, fld in f.items()}
            except FieldUnavailable:
                vals[st] = {}
            prog["done"] += 1

    await asyncio.gather(*(one(s) for s in steps))

    def arr(k):
        return np.array([vals[s].get(k, np.nan) for s in steps], dtype=float)

    out = {
        "model": model.id, "model_name": model.name, "run": run, "lat": lat, "lon": lon,
        "steps": steps, "valid": [run + dt.timedelta(hours=s) for s in steps],
        "s": {k[0] if k[1] == 0 else f"{k[0]}{k[1]}": arr(k) for k in keys if not
              (profile and k[1] and k not in UPPER_COMMON)},
    }
    if model.has("orog") and ("orog", 0) in keys:
        o = vals[steps[0]].get(("orog", 0))
        out["orog"] = float(o) if o is not None and np.isfinite(o) else None
        out["s"].pop("orog", None)
    if profile:
        out["levels"] = lv
        prof = {n: np.array([[vals[s].get((n, p), np.nan) for s in steps] for p in lv])
                for n in ("t", hum, "u", "v")}
        if hum == "q":
            from .products import rh_from_q
            prof["r"] = rh_from_q(prof["t"], prof["q"], np.array(lv)[:, None])
        out["profile"] = prof
    _SERIES.put(ck, out)
    return out
