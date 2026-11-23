"""Thunderstorms («trovoada»): conditions for severe convection in the ECMWF ENS members.

IPMA criteria (revised in August 2026), «fenómenos expectáveis por nível de aviso»:
orange = conditions favourable to *severe* localised phenomena (thunderstorm with 20–40 mm
in 1 h, hail ≥ 2 cm, gusts ≥ 90 km/h); red = *extreme* localised phenomena with potential
for high impact (≥ 40 mm in 1 h, large hail, very destructive gusts). These are the European
definitions of severe (hail ≥ 2 cm, gusts ≥ 25 m/s) and of extremely severe convective storms.

A global model at 9 km makes neither hail nor convective gusts, so — as forecasters do — the
warning is read from the environment of each member:

    WMAXSHEAR = √(2 · MUCAPE) × |V(500 hPa) − V(925 hPa)|          (m²/s²)

the largest updraught the instability allows times the deep-layer wind shear, which
organises storms into multicells and supercells. In European sounding and ERA5 studies
(Taszarek et al. 2017, MWR; 2020, J. Climate) it is the parameter that best separates
ordinary from severe thunderstorms and severe from extremely severe ones; ECMWF's
operational «CAPE-shear» is the same product without the √2 (same 925–500 hPa shear).

CAPE that is not released makes no storm: the index counts only where the member itself
precipitates (≥ GATE_MM in the 6 h around the time), i.e. where it starts convection
(ECMWF: «check CAPE and CAPE-shear values that are associated with forecast precipitation»).
It also needs MUCAPE ≥ criteria.STORM_CAPE: in winter fronts the shear alone (30–40 m/s
with 200–400 J/kg) would reach the threshold; those squall lines are left to the wind and
rain warnings, as IPMA does.
Total, not convective, precipitation: the convective one is missing from the 2024–2026
archive used to calibrate the thresholds.

Value of a day: the largest gated index at the 3-hourly steps of the UTC day (instantaneous
CAPE and winds; the shear is read every 6 h and averaged in between). Thresholds:
criteria.STORM (calibrated on past IPMA orange warnings, see the README).

Downloads: the days are screened with the IFS (the same index from the deterministic run
≥ 60 % of the orange threshold, ± one day); a screened day costs the member CAPE every 3 h
and the precipitation (shared with the rain warnings). The winds at 925 and 500 hPa, two
global fields per member and level every 6 h, are fetched only for the members with rain
and CAPE above the floor at the areas' points near that time: a few on a winter day, all
of them on a day of widespread storms (~0.5 GB).
"""
from __future__ import annotations

import asyncio
import datetime as dt

import numpy as np

from ..models import MODELS, FieldUnavailable
from . import criteria as C

H = dt.timedelta(hours=1)
GATE_MM = 1.0            # mm in the 6 h around the time: the member is precipitating there
SCREEN_RATIO = 0.6       # IFS index ≥ 60 % of the orange threshold -> fetch the members
WINDOW = 3               # h between the steps evaluated


def wmax(cape):
    """√(2·MUCAPE) (m/s), zero below the CAPE floor."""
    cape = np.asarray(cape, np.float32)
    return np.where(cape >= C.STORM_CAPE, np.sqrt(2.0 * np.maximum(cape, 0.0)), 0.0)


def index(cape, u9, v9, u5, v5):
    """WMAXSHEAR (m²/s²) from MUCAPE (J/kg) and the winds at 925 / 500 hPa (m/s)."""
    return wmax(cape) * np.hypot(u5 - u9, v5 - v9)


def shear(u9, v9, u5, v5):
    return np.hypot(u5 - u9, v5 - v9)


def day_steps(run: dt.datetime, day: dt.date) -> list[int]:
    """3-hourly steps (> 0) whose valid time falls in (00 UTC, 24 UTC] of the day."""
    t0 = dt.datetime(day.year, day.month, day.day)
    out = []
    for k in range(1, 9):
        s = int((t0 + k * WINDOW * H - run) / H)
        if s > 0:
            out.append(s)
    return out


def _wind_steps(s: int) -> list[int]:
    """6-hourly steps whose shear stands for step s (itself, or the two around it)."""
    return [s] if s % 6 == 0 else [s - 3, s + 3]


def _tp_steps(s: int) -> tuple[int, int]:
    return max(s - 3, 0), s + 3


# ---------------------------------------------------------------- ensemble members
WIND = ("u925", "v925", "u500", "v500")


async def _members_some(run: dt.datetime, name: str, step: int, want: np.ndarray) -> dict | None:
    """Only the members ``want`` ((50,) bool) of one pressure-level field at one step, cropped
    like ens.members; the others are NaN. Cached member by member (``have``), so a later call
    that needs more members fetches just those."""
    import httpx

    from ..config import HTTP_TIMEOUT, USER_AGENT
    from . import ens
    p = ens._cache_path(run, f"{name}m", step)
    have = np.zeros(50, bool)
    vals = lats = lons = None
    if p.exists():
        z = np.load(p)
        have, vals, lats, lons = z["have"].copy(), z["values"].astype(np.float32), z["lats"], z["lons"]
    miss = want & ~have
    if miss.any():
        prm, lev = ens.MEMBER_PARAMS[name][0], ens.MEMBER_LEVEL[name]
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT}) as c:
            try:
                url, entries = await ens.MEMBERS._index(c, run, step)
            except FieldUnavailable:
                return None
            todo = [e for e in entries if e.get("param") == prm and e.get("type") == "pf"
                    and e.get("levtype") == "pl" and str(e.get("levelist")) == str(lev)
                    and 1 <= int(e.get("number", 0)) <= 50 and miss[int(e["number"]) - 1]]
            blobs = await asyncio.gather(*(ens._get(c, url, e) for e in todo))
        fields = await asyncio.to_thread(lambda: [ens._decode_crop(b) for b in blobs])
        if fields:
            if vals is None:
                lats, lons = fields[0].lats, fields[0].lons
                vals = np.full((50, len(lats), len(lons)), np.nan, np.float32)
            for e, f in zip(todo, fields):
                k = int(e["number"]) - 1
                vals[k], have[k] = f.values, True
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".part.npz")
            np.savez_compressed(tmp, lats=lats, lons=lons, values=vals.astype(np.float16), have=have)
            tmp.replace(p)
    if vals is None:
        return None
    return {"lats": lats, "lons": lons, "values": vals, "have": have}


async def ens_daily(run: dt.datetime, days: list[dt.date], log, prune: bool = True) -> dict:
    """Per day: the largest gated index of every member at every land point (and at each step,
    for the plume), and on the grid (maps).

    The winds aloft, the bulk of the download, are fetched only for the members that have
    rain and CAPE above the floor at some sample point of the areas at a nearby time: the
    others cannot reach any threshold there, and their index is 0 (on the grid too, so the
    maps may miss a member outside the areas)."""
    from .detect import _members_many, sampler_for
    from .areas import land_points
    steps = sorted({s for d in days for s in day_steps(run, d)})
    tps = sorted({x for s in steps for x in _tp_steps(s) if x > 0})
    log(f"  trovoada: {len(days)} dia(s), {len(steps)} passos × 50 membros (CAPE, precipitação)")
    cape = await _members_many(run, "mucape", steps)
    tp = await _members_many(run, "tp", tps)
    if not cape:
        return {}
    ref = cape[min(cape)]
    lats, lons = ref["lats"], ref["lons"]
    sm = sampler_for(lats, lons)
    zero = np.zeros_like(ref["values"])

    def acc(s):
        return zero if s == 0 else (tp[s]["values"] if s in tp else None)

    gate, wmx = {}, {}
    need: dict[int, np.ndarray] = {}            # wind step -> members (50,) that need it
    for s in steps:
        a, b = _tp_steps(s)
        pa, pb = acc(a), acc(b)
        if s not in cape or pa is None or pb is None:
            continue
        gate[s] = (pb - pa) >= GATE_MM
        wmx[s] = wmax(cape[s]["values"])
        pot = sm.points((gate[s] & (wmx[s] > 0)).astype(np.float32))        # (50, npts)
        mem = np.nan_to_num(pot).max(-1) > 0 if prune else np.ones(50, bool)
        if mem.any():
            for w in _wind_steps(s):
                need[w] = need.get(w, np.zeros(50, bool)) | mem
    n = sum(int(m.sum()) for m in need.values())
    log(f"  trovoada: vento a 925/500 hPa para {n} membro×passo(s) em {len(need)} passo(s)")
    wind = {}
    sem = asyncio.Semaphore(4)

    async def one(w, name):
        async with sem:
            return w, name, await _members_some(run, name, w, need[w])

    res = await asyncio.gather(*(one(w, nm) for w in need for nm in WIND))
    got: dict = {}
    for w, nm, r in res:
        if r is not None:
            got.setdefault(w, {})[nm] = r["values"]
    for w, f in got.items():
        if len(f) == 4:
            wind[w] = shear(*(f[nm] for nm in WIND))                       # NaN: member not fetched
    lp = land_points()
    out = {}
    for d in days:
        grids, used = [], []
        for s in day_steps(run, d):
            if s not in gate:
                continue
            ws = [wind[w] for w in _wind_steps(s) if w in wind]
            if ws:
                g = wmx[s] * (np.mean(ws, 0) if len(ws) > 1 else ws[0])
            else:                 # no member with rain and CAPE at the areas
                g = np.zeros_like(wmx[s])
            grids.append(np.nan_to_num(np.where(gate[s], g, 0.0)).astype(np.float32))
            used.append(s)
        if not grids:
            continue
        st = np.stack(grids)                                            # (steps, 50, ny, nx)
        pts = np.stack([sm.points(g) for g in st])                      # (steps, 50, npts)
        out[d] = {"pts": np.nanmax(pts, 0), "series": pts, "grid": np.nanmax(st, 0),
                  "steps": used, "lats": lats, "lons": lons}
    assert all(v["pts"].shape[-1] == len(lp.codes) for v in out.values())
    return out


# ---------------------------------------------------------------- deterministic models
DET = ("ecmwf", "gfs", "icon_eu")


def _keys(m) -> list:
    cape = "mucape" if m.has("mucape") else "mlcape"      # ICON-EU: mixed-layer CAPE only
    return [(cape, 0), ("tp", 0), ("u", 925), ("v", 925), ("u", 500), ("v", 500)]


async def _fields(m, run, steps, keys):
    from .detect import DOMAIN
    sem = asyncio.Semaphore(4)
    got: dict = {}

    async def one(s):
        async with sem:
            try:
                f = await m.fetch(run, s, keys, DOMAIN)
            except FieldUnavailable:
                return
            if all(k in f for k in keys):
                got[s] = f

    await asyncio.gather(*(one(s) for s in steps))
    return got


def _det_index(got, s, keys, sample):
    """Gated index of one deterministic step on ``sample``'s points, or None."""
    a, b = _tp_steps(s)
    if s not in got or b not in got or (a > 0 and a not in got):
        return None
    tpa = sample(got[a][("tp", 0)].values) if a > 0 else 0.0
    rain = sample(got[b][("tp", 0)].values) - tpa
    f = got[s]
    v = index(*(sample(f[k].values) for k in (keys[0], ("u", 925), ("v", 925), ("u", 500), ("v", 500))))
    return np.where(rain >= GATE_MM, v, 0.0)


async def det_daily(mid: str, run: dt.datetime, days: list[dt.date], grid: bool = False) -> dict:
    """{day: (n_points,) daily index} of one deterministic model (grid=True: on its grid)."""
    from .detect import sampler_for
    m = MODELS[mid]
    steps = await m.list_steps(run)
    if not steps:
        return {}
    keys = _keys(m)
    want = {}
    for d in days:
        ss = [s for s in day_steps(run, d) if s in steps and _tp_steps(s)[1] in steps]
        if ss:
            want[d] = ss
    if not want:
        return {}
    need = sorted({x for ss in want.values() for s in ss for x in (s, *_tp_steps(s)) if x > 0})
    got = await _fields(m, run, need, keys)
    if not got:
        return {}
    f0 = got[next(iter(got))][keys[0]]
    sample = (lambda v: v) if grid else sampler_for(f0.lats, f0.lons).points
    out = {}
    for d, ss in want.items():
        vals = [x for s in ss if (x := _det_index(got, s, keys, sample)) is not None]
        if vals:
            out[d] = np.nanmax(np.stack(vals), 0)
    if grid:
        return {"lats": f0.lats, "lons": f0.lons, "days": out}
    return out


async def deterministic(days: list[dt.date], log) -> dict:
    from .detect import det_run
    res = {}
    for mid in DET:
        m = MODELS[mid]
        try:
            run = await det_run(m)
            vals = await det_daily(mid, run, days)
        except Exception as e:  # noqa: BLE001 — a model that fails must not stop the alerts
            log(f"  {mid} (trovoada): {e}")
            continue
        if vals:
            res[mid] = {"run": run.isoformat(), "name": m.name, "days": vals}
    return res


async def det_grid(mid: str, day: dt.date):
    """Index of the day on a deterministic model's grid (figures)."""
    from .detect import det_run
    run = await det_run(MODELS[mid])
    r = await det_daily(mid, run, [day], grid=True)
    if not r or day not in r["days"]:
        return None
    return r["lats"], r["lons"], r["days"][day], run


# ---------------------------------------------------------------- screening (IFS)
async def screen(run: dt.datetime, days: list[dt.date], out: dict, log) -> list[dt.date]:
    """Days worth the ensemble members: the IFS index reaches ≥ SCREEN_RATIO of the orange
    threshold at some land point (± one day, for timing errors)."""
    from .detect import det_run
    ifs = MODELS["ecmwf"]
    irun = await det_run(ifs)
    vals = await det_daily("ecmwf", irun, days)
    thr = C.STORM["laranja"]
    ratio = {d.isoformat(): 0.0 for d in days}
    for d, v in vals.items():
        if np.isfinite(v).any():
            ratio[d.isoformat()] = round(float(np.nanmax(v) / thr), 2)
    out["screen"]["trovoada"] = ratio
    flagged = set()
    for d in days:
        if ratio.get(d.isoformat(), 0.0) >= SCREEN_RATIO:
            flagged |= {d - dt.timedelta(days=1), d, d + dt.timedelta(days=1)}
    return sorted(x for x in flagged if x in days)
