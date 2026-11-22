"""Detector: from one ECMWF ENS run (and the latest deterministic runs) to the probability
of reaching the IPMA orange / red criteria in each warning area and day.

For every member, the field is read at every sample point of every area, compared with
the threshold of that point (wind: higher in the highlands), and the area takes the worst
point: the probability of an area-day is the fraction of the 50 members that reach the
level somewhere in the area. Heat and cold also need two consecutive days (≥ 48 h).

To keep downloads small, member fields are fetched only where they can matter: wind and
rain where the ENS probability products ("ep") already show some chance of ≥ 90 km/h
gusts / ≥ 25 mm in 24 h, heat and cold where the IFS forecast comes within a few degrees
of the orange threshold. The deterministic models (IFS, AIFS, GFS, ICON-EU) are read only
for the area-days the ensemble flags.

Rain in 1 h is the exception: the ECMWF ENS has no hourly precipitation, so it comes from
the 40 members of the DWD ICON-EU-EPS (rain1h.py), analysed together with each ECMWF run.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
from functools import lru_cache

import numpy as np

from ..config import ALERTS_DIR
from ..models import MODELS, Domain, FieldUnavailable
from . import criteria as C
from . import ens
from .areas import BOX, PointSet, Sampler, areas, land_points

H = dt.timedelta(hours=1)
RUNS_DIR = ALERTS_DIR / "runs"
DOMAIN = Domain(BOX[0], BOX[1], BOX[2], BOX[3])

HORIZON = {"vento": 5, "chuva": 5, "calor": 7, "frio": 7, "mar": 5, "neve": 5,
           "chuva1h": 2}        # days analysed after the run (1-h rain: hourly ICON-EU-EPS to +48 h)
SCREEN = 5.0            # % — ep probability below which member fields are not fetched
TEMP_MARGIN = 4.0       # °C — IFS within this of the orange threshold -> fetch members
WATCH_P = 0.2           # area-day probability (orange) that puts it "em vigilância"
DET_MARGIN = {"vento": 10.0, "chuva": 10.0, "calor": 1.0, "frio": 1.0, "mar": 1.0,
              "neve": 0.4,      # snow: a fraction of the threshold (6, 11 or 26 cm by altitude)
              "chuva1h": 8.0}
DET_MODELS = {"vento": ("ecmwf", "gfs", "icon_eu"),
              "chuva": ("ecmwf", "aifs", "gfs", "icon_eu"),
              "calor": ("ecmwf", "aifs", "gfs", "icon_eu"),
              "frio": ("ecmwf", "aifs", "gfs", "icon_eu")}
_HIGH_SIGN = {"vento": 1, "chuva": 1, "calor": 1, "frio": -1, "mar": 1, "neve": 1, "chuva1h": 1}


# ---------------------------------------------------------------- helpers
@lru_cache(maxsize=8)
def _sampler(sig) -> Sampler:
    lat0, nlat, dlat, lon0, nlon, dlon = sig
    return Sampler(lat0 + np.arange(nlat) * dlat, lon0 + np.arange(nlon) * dlon)


def sampler_for(lats, lons) -> Sampler:
    sig = (round(float(lats[0]), 4), len(lats), round(float(lats[1] - lats[0]), 5),
           round(float(lons[0]), 4), len(lons), round(float(lons[1] - lons[0]), 5))
    return _sampler(sig)


def point_meta() -> tuple[list[str], list[str]]:
    """Area code and group of every sample point, in Sampler order."""
    codes, groups = [], []
    for a in areas():
        codes += [a.code] * len(a.points)
        groups += [a.group] * len(a.points)
    return codes, groups


async def highland_points() -> np.ndarray:
    """True for mainland sample points whose (GFS 0.25°) orography is ≥ HIGHLAND_M."""
    p = ALERTS_DIR / "orog.npz"
    if not p.exists():
        gfs = MODELS["gfs"]
        run = (await gfs.list_runs())[0]
        f = (await gfs.fetch(run, 0, [("orog", 0)], DOMAIN))[("orog", 0)]
        np.savez_compressed(p, lats=f.lats, lons=f.lons, values=f.values)
    z = np.load(p)
    pts = sampler_for(z["lats"], z["lons"]).points(z["values"])
    codes, groups = point_meta()
    return np.array([g == "continente" and h >= C.HIGHLAND_M for g, h in zip(groups, pts)])


def pointset(phen: str) -> PointSet:
    """Sample points of a phenomenon: offshore for the sea state, the land points otherwise."""
    if phen == "mar":
        from .waves import sea_points
        return sea_points()
    if phen == "neve":
        from .snow import snow_points
        return snow_points()
    if phen == "chuva1h":
        from .rain1h import rain_points
        return rain_points()
    return land_points()


def point_thresholds(phen: str, highland: np.ndarray) -> np.ndarray:
    """(2, n_points): orange and red threshold at every sample point."""
    if phen == "mar":
        from .waves import thresholds
        return thresholds()
    if phen == "neve":
        from .snow import thresholds
        return thresholds()
    if phen == "chuva1h":
        from .rain1h import thresholds
        return thresholds()
    codes, groups = point_meta()
    out = np.zeros((2, len(codes)), np.float32)
    for i, (c, g) in enumerate(zip(codes, groups)):
        t = C.gust(c, g, bool(highland[i])) if phen == "vento" else C.thresholds(phen, c, g)
        out[0, i], out[1, i] = t["laranja"], t["vermelho"]
    return out


def levels(values: np.ndarray, thr: np.ndarray, phen: str) -> np.ndarray:
    """values (..., n_points) -> level 0/1/2 (none, orange, red) per point."""
    if _HIGH_SIGN[phen] > 0:
        return (values >= thr[0]).astype(np.int8) + (values >= thr[1]).astype(np.int8)
    return (values <= thr[0]).astype(np.int8) + (values <= thr[1]).astype(np.int8)


def by_area(arr: np.ndarray, how: str) -> dict[str, np.ndarray]:
    """(..., n_points) -> {code: (...)} worst value over the area's points."""
    out, k = {}, 0
    for a in areas():
        n = len(a.points)
        sub = arr[..., k:k + n]
        with np.errstate(all="ignore"):
            out[a.code] = (np.nanmax(sub, -1) if how == "max" else np.nanmin(sub, -1))
        k += n
    return out


def first_day(run: dt.datetime) -> dt.date:
    return run.date() if run.hour == 0 else run.date() + dt.timedelta(days=1)


def _steps_in(run: dt.datetime, day: dt.date, every_le144=3) -> list[int]:
    """ENS steps whose 3-h (6-h after +144 h) window ends inside the UTC day."""
    t0 = dt.datetime(day.year, day.month, day.day)
    out = []
    for s in ens.MEMBERS.nominal_steps(run):
        if s == 0:
            continue
        v = run + s * H
        if t0 < v <= t0 + 24 * H and (s > 144 or s % every_le144 == 0):
            out.append(s)
    return out


async def _members_many(run, name, steps, conc=3) -> dict[int, dict]:
    sem = asyncio.Semaphore(conc)
    got: dict[int, dict] = {}

    async def one(s):
        async with sem:
            m = await ens.members(run, name, s)
            if m is not None:
                got[s] = m

    await asyncio.gather(*(one(s) for s in steps))
    return got


# ---------------------------------------------------------------- ENS per phenomenon
async def _ens_wind(run, days, log) -> dict:
    steps = sorted({s for d in days for s in _steps_in(run, d)})
    log(f"  vento: {len(days)} dia(s), {len(steps)} passos × 50 membros")
    got = await _members_many(run, "gust", steps)
    out = {}
    for d in days:
        arrs = [got[s] for s in _steps_in(run, d) if s in got]
        if not arrs:
            continue
        sm = sampler_for(arrs[0]["lats"], arrs[0]["lons"])
        pts = np.stack([sm.points(a["values"]) for a in arrs])          # (steps, 50, npts)
        out[d] = {"pts": np.nanmax(pts, 0), "series": pts,
                  "grid": np.nanmax(np.stack([a["values"] for a in arrs]), 0),
                  "steps": [s for s in _steps_in(run, d) if s in got],
                  "lats": arrs[0]["lats"], "lons": arrs[0]["lons"]}
    return out


async def _ens_rain(run, days, log) -> dict:
    """Largest 6-h accumulation of each day (windows every 3 h up to +144 h)."""
    ends = sorted({s for d in days for s in _steps_in(run, d)})
    need = sorted({s for e in ends for s in (e, e - 6) if s >= 0})
    log(f"  chuva: {len(days)} dia(s), {len(need)} passos × 50 membros")
    got = await _members_many(run, "tp", [s for s in need if s > 0])
    out = {}
    for d in days:
        acc, grids, used = [], [], []
        for e in _steps_in(run, d):
            if e not in got or (e - 6 > 0 and e - 6 not in got):
                continue
            a = got[e]["values"] - (got[e - 6]["values"] if e - 6 > 0 else 0.0)
            a = np.maximum(a, 0.0)
            grids.append(a)
            used.append(e)
        if not grids:
            continue
        ref = got[used[0]]
        sm = sampler_for(ref["lats"], ref["lons"])
        pts = np.stack([sm.points(g) for g in grids])
        out[d] = {"pts": np.nanmax(pts, 0), "series": pts, "grid": np.nanmax(np.stack(grids), 0),
                  "steps": used, "lats": ref["lats"], "lons": ref["lons"]}
    return out


def _temp_steps(run, day, kind) -> list[int]:
    """Member max/min windows that represent Tmax (12–18 UTC) / Tmin (03–09 UTC) of a day."""
    t0 = dt.datetime(day.year, day.month, day.day)
    ends = [t0 + 15 * H, t0 + 18 * H] if kind == "tmax" else [t0 + 6 * H, t0 + 9 * H]
    out = []
    for e in ends:
        s = int((e - run) / H)
        if s <= 0:
            continue
        if s <= 144:
            out.append(s)
        elif s % 6 == 0:
            out.append(s)
        elif kind == "tmin" and (s + 3) % 6 == 0:   # 6-h windows: 06–12 holds the morning
            out.append(s + 3)
    return sorted(set(out))


async def _ens_temp(run, days, kind, log) -> dict:
    steps = sorted({s for d in days for s in _temp_steps(run, d, kind)})
    log(f"  {'calor' if kind == 'tmax' else 'frio'}: {len(days)} dia(s), {len(steps)} passos × 50 membros")
    got = await _members_many(run, kind, steps)
    out = {}
    for d in days:
        arrs = [got[s] for s in _temp_steps(run, d, kind) if s in got]
        if not arrs:
            continue
        st = np.stack([a["values"] for a in arrs])
        grid = np.nanmax(st, 0) if kind == "tmax" else np.nanmin(st, 0)
        sm = sampler_for(arrs[0]["lats"], arrs[0]["lons"])
        out[d] = {"pts": sm.points(grid), "grid": grid, "lats": arrs[0]["lats"],
                  "lons": arrs[0]["lons"]}
    return out


# ---------------------------------------------------------------- deterministic models
async def _det_daily(model, run, phen, days) -> dict:
    """{day: (n_points,) daily value} of one deterministic model (missing days skipped)."""
    out: dict = {}
    if phen in ("calor", "frio"):
        from ..daily import daily_fields
        kind = "tmax" if phen == "calor" else "tmin"
        try:
            d = await daily_fields(model, run, DOMAIN, kinds=(kind,))
        except FieldUnavailable:
            return out
        g = d["grid"]
        sm = sampler_for(g.lats, g.lons)
        for day in days:
            if day in d[kind]:
                out[day] = sm.points(d[kind][day])
        return out
    steps = await model.list_steps(run)
    key = ("gust", 0) if phen == "vento" else ("tp", 0)
    if not model.has(key[0]):
        return out
    for day in days:
        t0 = dt.datetime(day.year, day.month, day.day)
        lo, hi = int((t0 - run) / H), int((t0 + 24 * H - run) / H)
        if lo < 0 or hi > steps[-1]:
            continue
        if phen == "vento":
            use = [s for s in steps if lo < s <= hi and s % 3 == 0]
        else:
            use = [s for s in steps if lo - 6 <= s <= hi and s % 3 == 0 and s >= 0]
        if not use:
            continue
        sem = asyncio.Semaphore(4)
        got = {}

        async def one(s):
            async with sem:
                try:
                    got[s] = (await model.fetch(run, s, [key], DOMAIN))[key]
                except FieldUnavailable:
                    pass

        await asyncio.gather(*(one(s) for s in use))
        if not got:
            continue
        f0 = next(iter(got.values()))
        sm = sampler_for(f0.lats, f0.lons)
        if phen == "vento":
            vals = [sm.points(got[s].values) * 3.6 for s in sorted(got) if s > lo]
        else:
            vals = [np.maximum(sm.points(got[s].values) - sm.points(got[s - 6].values), 0)
                    for s in sorted(got) if s > lo and s - 6 in got]
        if vals:
            out[day] = np.nanmax(np.stack(vals), 0)
    return out


async def det_run(m) -> dt.datetime:
    """Run of a deterministic model for the alerts: its newest complete run, except that for
    ICON-EU only the long runs count (00/06/12/18Z reach +120 h; 03/09/15/21Z stop at +48 h
    and would leave days 3–4 without the model)."""
    if m.id == "icon_eu":
        for r in (await m.list_runs())[:6]:
            if r.hour % 6 == 0:
                steps = await m.list_steps(r)
                if steps and steps[-1] >= m.last_nominal_step(r):
                    return r
    return await m.best_run()


async def deterministic(phen: str, days: list, log) -> dict:
    """{model id: {"run": iso, "days": {day: (n_points,)}}} for the models that cover it."""
    res = {}
    for mid in DET_MODELS[phen]:
        m = MODELS[mid]
        try:
            run = await det_run(m)
            vals = await _det_daily(m, run, phen, days)
        except Exception as e:  # noqa: BLE001 — a model that fails must not stop the alerts
            log(f"  {mid}: {e}")
            continue
        if vals:
            res[mid] = {"run": run.isoformat(), "name": m.name, "days": vals}
    return res


# ---------------------------------------------------------------- analysis of one run
def _summary(phen, d, pts, lev_pts, thr_pts, det, codes_by_area) -> dict:
    """Per-area numbers of one day: probabilities, member statistics, deterministic support."""
    how = "max" if _HIGH_SIGN[phen] > 0 else "min"
    ps = pointset(phen)
    lev_area = ps.by_area(lev_pts.astype(np.float32), "max")       # (50,) per area
    val_area = ps.by_area(pts, how)                                 # (50,) per area
    out = {}
    for code, lev in lev_area.items():
        v = val_area[code]
        if not np.isfinite(v).any():
            continue
        po, pr = float(np.mean(lev >= 1)), float(np.mean(lev >= 2))
        sl = codes_by_area[code]
        thr_o = float(np.min(thr_pts[0, sl]) if _HIGH_SIGN[phen] > 0 else np.max(thr_pts[0, sl]))
        thr_r = float(np.min(thr_pts[1, sl]) if _HIGH_SIGN[phen] > 0 else np.max(thr_pts[1, sl]))
        q = np.nanpercentile(v, [10, 50, 90])
        row = {"p_o": round(po, 3), "p_r": round(pr, 3), "p10": round(float(q[0]), 1),
               "p50": round(float(q[1]), 1), "p90": round(float(q[2]), 1),
               "max": round(float(np.nanmax(v) if how == "max" else np.nanmin(v)), 1),
               "thr_o": thr_o, "thr_r": thr_r, "det": {}}
        for mid, dd in det.items():
            if d not in dd["days"]:
                continue
            mv = dd["days"][d][sl]
            ml = levels(mv, thr_pts[:, sl], phen)
            m = DET_MARGIN[phen] * (thr_pts[:, sl] if phen == "neve" else 1.0)
            m0, m1 = (m[0], m[1]) if np.ndim(m) else (m, m)
            if _HIGH_SIGN[phen] > 0:
                near, near_r = mv >= thr_pts[0, sl] - m0, mv >= thr_pts[1, sl] - m1
            else:
                near, near_r = mv <= thr_pts[0, sl] + m0, mv <= thr_pts[1, sl] + m1
            with np.errstate(all="ignore"):
                worst = float(np.nanmax(mv) if how == "max" else np.nanmin(mv))
            if np.isfinite(worst):
                row["det"][mid] = {"v": round(worst, 1), "lev": int(ml.max()), "near": bool(near.any()),
                                   "near_r": bool(near_r.any())}
        if phen == "neve":           # report the decisive altitude band, not a mix of bands
            from .snow import band_summary
            k, bands = band_summary(code, pts, lev_pts, thr_pts)
            b = bands[k - sl.start]
            row.update({x: b[x] for x in ("p10", "p50", "p90", "max", "thr_o", "thr_r")})
            row["z"], row["bands"] = b["z"], bands
            for mid, dd in det.items():
                if mid in row["det"] and d in dd["days"] and np.isfinite(dd["days"][d][k]):
                    row["det"][mid]["v"] = round(float(dd["days"][d][k]), 1)
        out[code] = row
    return out


async def analyse(run: dt.datetime, log=print) -> dict:
    """Signals of one ENS run: {phen: {day iso: {area: row}}} plus metadata. Saves the
    member arrays used by the figures in RUNS_DIR/<run>/."""
    highland = await highland_points()
    codes, _ = point_meta()
    slices, k = {}, 0
    for a in areas():
        slices[a.code] = slice(k, k + len(a.points))
        k += len(a.points)
    d0 = first_day(run)
    out = {"run": run.isoformat(), "analysed": dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"),
           "signals": {}, "screen": {}}
    rdir = RUNS_DIR / run.strftime("%Y%m%d%H")
    rdir.mkdir(parents=True, exist_ok=True)

    # ---- screening with the ENS probability products
    ep = await ens.ep_fields(run, ["p_gust90", "p_tp25"])
    sm_ep = sampler_for(ep["lats"], ep["lons"])

    def screened(name, ndays):
        days = []
        for i in range(ndays):
            d = d0 + dt.timedelta(days=i)
            t0 = dt.datetime(d.year, d.month, d.day)
            best = 0.0
            for (a, b), arr in zip(ep[name]["windows"], ep[name]["values"]):
                wa, wb = run + a * H, run + b * H
                if wa < t0 + 24 * H and wb > t0:
                    best = max(best, float(np.nanmax(sm_ep.points(arr))))
            out["screen"].setdefault(name, {})[d.isoformat()] = round(best, 1)
            if best >= SCREEN:
                days.append(d)
        return days

    wind_days = screened("p_gust90", HORIZON["vento"])
    rain_days = screened("p_tp25", HORIZON["chuva"])

    # heat / cold: IFS within TEMP_MARGIN of the orange threshold (and the neighbours, for 48 h)
    temp_days = {"calor": [], "frio": []}
    try:
        from ..daily import daily_fields
        ifs = MODELS["ecmwf"]
        dd = await daily_fields(ifs, await ifs.best_run(), DOMAIN)
        sm = sampler_for(dd["grid"].lats, dd["grid"].lons)
        for phen, kind in (("calor", "tmax"), ("frio", "tmin")):
            thr = point_thresholds(phen, highland)
            flagged = set()
            for day, arr in dd[kind].items():
                if not (d0 <= day < d0 + dt.timedelta(days=HORIZON[phen])):
                    continue
                v = sm.points(arr)
                hit = (v >= thr[0] - TEMP_MARGIN) if phen == "calor" else (v <= thr[0] + TEMP_MARGIN)
                if hit.any():
                    flagged |= {day - dt.timedelta(days=1), day, day + dt.timedelta(days=1)}
            temp_days[phen] = sorted(x for x in flagged if x >= d0)
            out["screen"][phen] = [x.isoformat() for x in temp_days[phen]]
    except Exception as e:  # noqa: BLE001
        log(f"  triagem de temperatura (IFS) falhou: {e}")

    # sea state: ep probabilities of the wave ensemble
    out["failed"] = []
    sea_days = []
    try:
        from . import waves
        sea_days = await waves.screen(run, [d0 + dt.timedelta(days=i) for i in range(HORIZON["mar"])],
                                      out, SCREEN)
    except Exception as e:  # noqa: BLE001
        log(f"  triagem do mar falhou: {e}")
        out["failed"].append("mar")

    # snow: IFS estimate at the altitudes of each district
    snow_days = []
    try:
        from . import snow
        snow_days = await snow.screen(run, [d0 + dt.timedelta(days=i) for i in range(HORIZON["neve"])],
                                      out, log)
    except Exception as e:  # noqa: BLE001
        log(f"  triagem da neve falhou: {e}")
        out["failed"].append("neve")

    # ---- members (a phenomenon that fails is marked, so its events are not cancelled)
    from . import rain1h
    ens_data = {}
    jobs = [("vento", wind_days, lambda: _ens_wind(run, wind_days, log)),
            ("chuva", rain_days, lambda: _ens_rain(run, rain_days, log)),
            ("calor", temp_days["calor"], lambda: _ens_temp(run, temp_days["calor"], "tmax", log)),
            ("frio", temp_days["frio"], lambda: _ens_temp(run, temp_days["frio"], "tmin", log)),
            ("mar", sea_days, lambda: waves.ens_daily(run, sea_days, log)),
            ("neve", snow_days, lambda: snow.ens_daily(run, snow_days, log)),
            # 1-h rain: hourly ICON-EU-EPS, which screens its own days (exactly, see rain1h.py)
            ("chuva1h", [run], lambda: rain1h.ens_daily(run, out, log))]
    if "calor" not in out["screen"]:        # the temperature screening (IFS) failed
        out["failed"] += ["calor", "frio"]
    for phen, days, job in jobs:
        if not days:
            continue
        try:
            ens_data[phen] = await job()
        except Exception as e:  # noqa: BLE001
            log(f"  {phen}: membros do {'ICON-EU-EPS' if phen == 'chuva1h' else 'ENS'} falharam ({e})")
            out["failed"].append(phen)

    # ---- levels, probabilities, deterministic support
    for phen, per_day in ens_data.items():
        if not per_day:
            continue
        try:
            await _finish_async(phen, per_day, highland, slices, out, rdir, log)
        except Exception as e:  # noqa: BLE001
            log(f"  {phen}: análise falhou ({e})")
            out["failed"].append(phen)
    with open(rdir / "signals.json", "w") as fh:
        json.dump(out, fh, ensure_ascii=False)
    return out


async def _finish_async(phen, per_day, highland, slices, out, rdir, log):
    thr = point_thresholds(phen, highland)
    ps = pointset(phen)
    slices = ps.slices
    days = sorted(per_day)
    # sea state: levels already computed per member (south-easterly swell rule in the Algarve)
    lev = {d: per_day[d]["lev"] if "lev" in per_day[d] else levels(per_day[d]["pts"], thr, phen)
           for d in days}                                                   # (50, npts)
    if phen in ("calor", "frio"):          # ≥ 48 h: the level must hold on two consecutive days
        pers = {}
        for d in days:
            prev, nxt = lev.get(d - dt.timedelta(days=1)), lev.get(d + dt.timedelta(days=1))
            cand = [np.minimum(lev[d], x) for x in (prev, nxt) if x is not None]
            pers[d] = np.maximum.reduce(cand) if cand else np.zeros_like(lev[d])
        lev = pers
    # deterministic models only where the ensemble sees something
    def max_p(levarr):
        la = ps.by_area(levarr.astype(np.float32), "max")
        return max((float(np.mean(v >= 1)) for v in la.values()), default=0.0)

    flag_days = [d for d in days if max_p(lev[d]) >= WATCH_P]
    if phen == "mar":
        from . import waves
        det = await waves.deterministic(flag_days, log) if flag_days else {}
    elif phen == "neve":
        from . import snow
        det = await snow.deterministic(flag_days, log) if flag_days else {}
    elif phen == "chuva1h":
        from . import rain1h
        det = await rain1h.deterministic(flag_days, log) if flag_days else {}
    else:
        det = await deterministic(phen, flag_days, log) if flag_days else {}
    sig = {}
    for d in days:
        sig[d.isoformat()] = _summary(phen, d, per_day[d]["pts"], lev[d], per_day[d].get("thr", thr),
                                      det, slices)
    out["signals"][phen] = sig
    out.setdefault("det_runs", {})[phen] = {m: v["run"] for m, v in det.items()}
    # arrays for the figures
    save = {"days": np.array([d.isoformat() for d in days])}
    for d in days:
        k = d.isoformat()
        g = per_day[d]["grid"]
        save[f"grid_{k}"] = g.astype(np.float16)
        save[f"pts_{k}"] = per_day[d]["pts"].astype(np.float16)
        save[f"lev_{k}"] = lev[d]
        if "thr" in per_day[d]:                 # sea state: thresholds of the day (SE swell rule)
            save[f"thr_{k}"] = per_day[d]["thr"]
        if "levgrid" in per_day[d]:             # snow: level on the grid (any altitude band)
            save[f"levgrid_{k}"] = per_day[d]["levgrid"]
        if "series" in per_day[d]:
            save[f"series_{k}"] = per_day[d]["series"].astype(np.float16)
            save[f"steps_{k}"] = np.array(per_day[d]["steps"])
    ref = per_day[days[0]]
    save["lats"], save["lons"] = ref["lats"], ref["lons"]
    if "src_run" in ref:                        # 1-h rain: the ICON-EU-EPS run used
        save["src_run"] = np.array(ref["src_run"])
    np.savez_compressed(rdir / f"{phen}.npz", **save)
    with open(rdir / f"{phen}_det.json", "w") as fh:
        json.dump({m: {"run": v["run"], "name": v["name"],
                       "days": {d.isoformat(): [None if not np.isfinite(x) else round(float(x), 1)
                                                for x in arr] for d, arr in v["days"].items()}}
                   for m, v in det.items()}, fh)
