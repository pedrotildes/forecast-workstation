"""Fog («nevoeiro»): IPMA warns on its persistence — yellow ≥ 48 h, orange ≥ 72 h, red ≥ 96 h
(«Visibilidade», duration; the same in all regions).

What the models can see. The open ECMWF data have no visibility, and the 0.25° grid does not
see the valleys where persistent fog lives: in the orange episode of 26/12/2024–02/01/2025
(Bragança, Viseu, Guarda, Vila Real: fog «em especial em zonas de vale», with rime) the IFS
saturated every night but warmed and dried a little every afternoon. What it does show,
clearly, is the regime that keeps the fog: saturated air at dawn, an afternoon that stays
humid under the inversion, little wind, no rain. So the duration here is counted in days of
that regime.

Fog day of a point (2-m temperature T and dew point Td):
* T − Td ≤ DD_SAT at 06 UTC (saturated at dawn);
* T − Td ≤ DD_AFT at 15 UTC (the afternoon stays humid: the fog or the low cloud does not clear);
* 10-m wind ≤ WIND_MAX at 15 UTC; precipitation in the UTC day ≤ RAIN_MAX (rain is not fog).
Fog day of an area: some of its points. Duration on day D: 24 h × the consecutive fog days up
to D (orange from the third day, as IPMA did on 28/12/2024). The days before the run come from
the IFS (the 00 UTC run of each day, the same test), up to PRIOR days back.
Red: four consecutive days of fog that does not lift even in the afternoon (T − Td ≤ DD_DENSE
at 15 UTC). It never happened in the two winters checked, and IPMA kept the 2024 episode at
orange for eight days.

Calibration (IFS, 00 UTC run of each day, 240 winter days Nov–Feb 2024/25 and 2025/26): the only
sequences of ≥ 2 fog days were the two IPMA fog episodes — Dec 2024 (orange): Bragança 5 days,
Viseu 4, Guarda 4, Vila Real 4; Dec 2025 (yellow, 48 h): Bragança, Guarda and Vila Real 2 —
plus two other 2-day ones (Setúbal 9/1/2025, Azores Eastern group 2/12/2024); no sequence of
≥ 3 days outside the 2024 episode. Without the wind test Setúbal became 3 days.

Downloads. The ENS is read only when the IFS shows ≥ 2 consecutive fog days in some area (the
past days included). Then, first the 06 and 15 UTC temperature and dew point of all members
(≈ 130 MB a day); only the members that can reach 72 h (consecutive days saturated at dawn with
a humid afternoon, the past days included) get the rest — the 15 UTC wind and the day's
precipitation. The others keep that count, an upper bound below 72 h.

Deterministic support: IFS, GFS, ICON-EU (the AIFS has no 15 UTC step).
"""
from __future__ import annotations

import asyncio
import datetime as dt
from functools import lru_cache

import numpy as np

from ..models import MODELS, FieldUnavailable
from . import criteria as C
from .areas import PointSet, areas, land_points

H = dt.timedelta(hours=1)
DD_SAT = 0.5          # K, 06 UTC
DD_AFT = 3.5          # K, 15 UTC
DD_DENSE = 1.0        # K, 15 UTC (red)
WIND_MAX = 3.0        # m/s, 15 UTC
RAIN_MAX = 1.0        # mm in the day
PRIOR = 4             # days before the run read from the IFS
SCREEN_DAYS = 3       # IFS consecutive fog days that call the ensemble …
SCREEN_DD_AFT = 4.5   # … with a more tolerant afternoon (K at 15 UTC)
DET = ("ecmwf", "gfs", "icon_eu")
KEYS = [("t2m", 0), ("d2m", 0), ("u10", 0), ("v10", 0)]


# ---------------------------------------------------------------- one value per area
class FogPoints(PointSet):
    """One point per area (its label): the fog day is decided over the area's land points."""

    def __init__(self):
        lp = land_points()
        super().__init__([(a.code, a.group, *(a.label if a.group == "continente" else lp.label(a.code)))
                          for a in areas() if a.code in lp.slices], "fog")


@lru_cache(maxsize=1)
def fog_points() -> FogPoints:
    return FogPoints()


def thresholds() -> np.ndarray:
    n = len(fog_points())
    return np.stack([np.full(n, C.FOG["laranja"]), np.full(n, C.FOG["vermelho"])]).astype(np.float32)


def area_any(flags: np.ndarray) -> np.ndarray:
    """(..., land points) bool -> (..., areas) bool: some point of the area."""
    lp = land_points()
    return np.stack([flags[..., lp.slices[c]].any(-1) for c in fog_points().codes], -1)


def day_flags(t06, d06, t15, d15, u15, v15, rain, dd_aft: float = DD_AFT):
    """Fog day and dense-fog day (bool arrays) from the fields at the points or on a grid
    (temperatures in °C or K, wind m/s, rain mm). NaN anywhere gives False."""
    with np.errstate(invalid="ignore"):
        dd15 = t15 - d15
        ok = ((t06 - d06) <= DD_SAT) & (dd15 <= dd_aft) & (np.hypot(u15, v15) <= WIND_MAX) \
            & (rain <= RAIN_MAX)
        return ok, ok & (dd15 <= DD_DENSE)


def counts(days: np.ndarray, prior) -> np.ndarray:
    """(D, …) bool fog days -> (D, …) consecutive fog days up to each day (``prior``: the
    count on the day before the first, broadcast to the trailing shape)."""
    out = np.zeros(days.shape, np.int16)
    c = np.broadcast_to(np.asarray(prior, np.int16), days.shape[1:]).copy()
    for i in range(days.shape[0]):
        c = np.where(days[i], c + 1, 0).astype(np.int16)
        out[i] = c
    return out


def day_steps(run: dt.datetime, day: dt.date) -> dict[str, int]:
    t0 = dt.datetime(day.year, day.month, day.day)
    return {k: int((t0 + h * H - run) / H) for k, h in (("s00", 0), ("s06", 6), ("s15", 15), ("s24", 24))}


# ---------------------------------------------------------------- deterministic models
async def _model_day(m, run: dt.datetime, day: dt.date, steps_avail=None, dd_aft: float = DD_AFT):
    """(fog, dense) bool at the land points for one model day; None if not covered."""
    from .detect import DOMAIN, sampler_for
    st = day_steps(run, day)
    steps_avail = steps_avail if steps_avail is not None else await m.list_steps(run)
    if st["s06"] < 0 or st["s24"] > steps_avail[-1] or any(st[k] not in steps_avail for k in ("s06", "s15")):
        return None
    s00 = max(st["s00"], 0)
    try:
        f06, f15, p0, p1 = await asyncio.gather(
            m.fetch(run, st["s06"], KEYS[:2], DOMAIN), m.fetch(run, st["s15"], KEYS, DOMAIN),
            m.fetch(run, s00, [("tp", 0)], DOMAIN) if s00 > 0 else asyncio.sleep(0, None),
            m.fetch(run, st["s24"], [("tp", 0)], DOMAIN))
    except FieldUnavailable:
        return None
    ref = f15[("t2m", 0)]
    sm = sampler_for(ref.lats, ref.lons)
    P = (lambda f, k: sm.points(f[k].values))
    rain = P(p1, ("tp", 0)) - (P(p0, ("tp", 0)) if p0 is not None else 0.0)
    return day_flags(P(f06, ("t2m", 0)), P(f06, ("d2m", 0)), P(f15, ("t2m", 0)), P(f15, ("d2m", 0)),
                     P(f15, ("u10", 0)), P(f15, ("v10", 0)), rain, dd_aft)


async def prior_days(d0: dt.date) -> tuple[np.ndarray, np.ndarray]:
    """(n_areas,) consecutive fog days / dense-fog days before ``d0``, from the IFS 00 UTC run
    of each previous day (its own day: the closest thing to an analysis in the open data)."""
    ifs = MODELS["ecmwf"]
    na = len(fog_points())
    fog = np.zeros((PRIOR, na), bool)
    dense = np.zeros((PRIOR, na), bool)
    alive = np.ones(na, bool)
    for k in range(1, PRIOR + 1):
        day = d0 - dt.timedelta(days=k)
        try:
            r = await _model_day(ifs, dt.datetime(day.year, day.month, day.day), day,
                                 list(range(0, 25, 3)))
        except Exception:  # noqa: BLE001 — a missing past run just ends the count
            r = None
        if r is None:
            break
        fog[k - 1], dense[k - 1] = area_any(r[0]), area_any(r[1])
        alive &= fog[k - 1]
        if not alive.any():
            break
    def run_len(x):                                   # consecutive True from the start
        return np.argmin(np.vstack([x, np.zeros((1, na), bool)]), 0)
    return run_len(fog).astype(np.int16), run_len(dense).astype(np.int16)


async def screen(run: dt.datetime, days: list[dt.date], out: dict, log) -> list[dt.date]:
    """Days for the ensemble: all of them if the newest IFS shows ≥ SCREEN_DAYS consecutive fog
    days in some area (the past days included; afternoon test SCREEN_DD_AFT), none otherwise."""
    from .detect import first_day
    ifs = MODELS["ecmwf"]
    irun = await ifs.best_run()
    steps = await ifs.list_steps(irun)
    flags = []
    for d in days:
        r = await _model_day(ifs, irun, d, steps, SCREEN_DD_AFT)
        flags.append(area_any(r[0]) if r is not None else np.zeros(len(fog_points()), bool))
    prior, _ = await prior_days(first_day(run))
    c = counts(np.stack(flags), prior)
    out["screen"]["nevoeiro"] = {d.isoformat(): int(c[i].max()) for i, d in enumerate(days)}
    out.setdefault("prior", {})["nevoeiro"] = {code: int(p) for code, p in zip(fog_points().codes, prior) if p}
    if c.max() >= SCREEN_DAYS:
        return list(days)
    return []


async def _det_counts(m, run: dt.datetime, days: list[dt.date], d0: dt.date, prior) -> dict:
    """{day: (n_areas,) hours} of one model, counting from d0 (days it cannot cover break the count)."""
    steps = await m.list_steps(run)
    last = max(days)
    flags, ds = [], []
    d = d0
    while d <= last:
        r = await _model_day(m, run, d, steps)
        flags.append(area_any(r[0]) if r is not None else None)
        ds.append(d)
        d += dt.timedelta(days=1)
    out = {}
    na = len(fog_points())
    c = np.asarray(prior, np.int16)
    for d, f in zip(ds, flags):
        if f is None:
            c = np.zeros(na, np.int16)
            continue
        c = np.where(f, c + 1, 0).astype(np.int16)
        if d in days:
            out[d] = 24.0 * c.astype(np.float32)
    return out


async def deterministic(days: list[dt.date], log, d0: dt.date | None = None) -> dict:
    """{model: {"run", "name", "days": {day: (n_areas,) hours of fog}}}."""
    from .detect import det_run
    d0 = d0 or min(days)
    prior, _ = await prior_days(d0)
    res = {}
    for mid in DET:
        m = MODELS[mid]
        try:
            run = await det_run(m)
            vals = await _det_counts(m, run, days, d0, prior)
        except Exception as e:  # noqa: BLE001 — a model that fails must not stop the alerts
            log(f"  {mid} (nevoeiro): {e}")
            continue
        if vals:
            res[mid] = {"run": run.isoformat(), "name": m.name, "days": vals}
    return res


# ---------------------------------------------------------------- ensemble
async def _all(run, name, steps):
    from .detect import _members_many
    return await _members_many(run, name, [s for s in steps if s > 0])


async def ens_daily(run: dt.datetime, days: list[dt.date], log) -> dict:
    """Per day: hours of fog of every member in every area ("pts", "lev") and on the grid."""
    from . import ens
    from .detect import first_day, sampler_for
    d0 = first_day(run)
    days = [d for d in days if 0 < day_steps(run, d)["s06"] and day_steps(run, d)["s24"] <= 144]
    if not days:
        return {}
    st = {d: day_steps(run, d) for d in days}
    # 1) 06 and 15 UTC temperature and dew point of all members
    steps = sorted({st[d][k] for d in days for k in ("s06", "s15")})
    t2, d2 = await asyncio.gather(_all(run, "t2m", steps), _all(run, "d2m", steps))
    ok_days = [d for d in days if all(st[d][k] in t2 and st[d][k] in d2 for k in ("s06", "s15"))]
    if not ok_days:
        return {}
    ref = t2[st[ok_days[0]]["s15"]]
    sm = sampler_for(ref["lats"], ref["lons"])

    def dd(s, grid=False):
        a = t2[s]["values"] - d2[s]["values"]
        return a if grid else sm.points(a)

    cand = np.stack([area_any((dd(st[d]["s06"]) <= DD_SAT) & (dd(st[d]["s15"]) <= DD_AFT)) for d in ok_days])
    prior, prior_d = await prior_days(d0)
    # consecutive days must start at d0 to carry the past days
    contiguous = ok_days[0] == d0 and all((b - a).days == 1 for a, b in zip(ok_days, ok_days[1:]))
    ub = counts(cand, prior if contiguous else 0)                             # (D, 50, areas)
    need_m = (ub >= C.FOG["laranja"] / 24).any((0, 2))                        # (50,)
    log(f"  nevoeiro: {len(ok_days)} dia(s), madrugada saturada e tarde húmida em "
        f"{int(cand.any((0, 2)).sum())} membros, {int(need_m.sum())} podem chegar às 72 h")
    fog = np.zeros(cand.shape, bool)
    dense = np.zeros(cand.shape, bool)
    gfog = None
    if need_m.any():
        # 2) for those members only: 15 UTC wind and the day's precipitation
        sem = asyncio.Semaphore(4)

        # only on the days when that member is saturated at dawn with a humid afternoon somewhere
        want: dict[tuple[str, int], np.ndarray] = {}
        for i, d in enumerate(ok_days):
            w = need_m & cand[i].any(-1)
            for name, s in (("u10", st[d]["s15"]), ("v10", st[d]["s15"]), ("tp", st[d]["s00"]),
                            ("tp", st[d]["s24"])):
                want[(name, s)] = want.get((name, s), np.zeros(50, bool)) | w

        async def some(name, s):
            if s <= 0 or not want[(name, s)].any():
                return name, s, None
            async with sem:
                return name, s, await ens.members_some(run, name, s, want[(name, s)])

        jobs = [some(name, s) for name, s in want]
        got = {}
        for name, s, r in await asyncio.gather(*jobs):
            if r is not None:
                got[(name, s)] = r["values"]
        ny, nx = ref["values"].shape[1:]
        gfog = np.zeros((len(ok_days), 50, ny, nx), bool)
        zero = np.zeros((50, ny, nx), np.float32)
        for i, d in enumerate(ok_days):
            k = st[d]
            fields = [t2[k["s06"]]["values"], d2[k["s06"]]["values"], t2[k["s15"]]["values"],
                      d2[k["s15"]]["values"], got.get(("u10", k["s15"])), got.get(("v10", k["s15"])),
                      got.get(("tp", k["s24"]))]
            if any(f is None for f in fields):          # no member needed that day
                continue
            rain = fields[6] - (got.get(("tp", k["s00"]), zero) if k["s00"] > 0 else zero)
            f_g, _ = day_flags(*fields[:6], rain)
            gfog[i] = f_g & need_m[:, None, None]
            fp, dp = day_flags(*(sm.points(x) for x in fields[:6]), sm.points(rain))
            fog[i], dense[i] = area_any(fp) & need_m[:, None], area_any(dp) & need_m[:, None]
    cnt = counts(fog, prior if contiguous else 0)
    dcnt = counts(dense, prior_d if contiguous else 0)
    cnt = np.where(need_m[None, :, None], cnt, ub)                         # the others: upper bound < 3
    out = {}
    for i, d in enumerate(ok_days):
        lev = ((cnt[i] >= C.FOG["laranja"] / 24).astype(np.int8)
               + (dcnt[i] >= C.FOG["vermelho"] / 24).astype(np.int8))
        grid = (24.0 * counts(gfog[:i + 1], 0)[-1]).astype(np.float32) if gfog is not None else \
            np.zeros(ref["values"].shape, np.float32)
        out[d] = {"pts": 24.0 * cnt[i].astype(np.float32), "lev": lev, "grid": grid,
                  "lats": ref["lats"], "lons": ref["lons"]}
    return out


async def det_grid(mid: str, day: dt.date, d0: dt.date | None = None):
    """Hours of fog of a model on its grid on ``day``, counting its days from ``d0`` (figures)."""
    from .detect import DOMAIN, det_run
    m = MODELS[mid]
    run = await det_run(m)
    steps = await m.list_steps(run)
    d = d0 or day
    c = None
    lats = lons = None
    while d <= day:
        st = day_steps(run, d)
        if st["s06"] < 0 or st["s24"] > steps[-1]:
            return None
        try:
            f06 = await m.fetch(run, st["s06"], KEYS[:2], DOMAIN)
            f15 = await m.fetch(run, st["s15"], KEYS, DOMAIN)
            p1 = await m.fetch(run, st["s24"], [("tp", 0)], DOMAIN)
            p0 = await m.fetch(run, st["s00"], [("tp", 0)], DOMAIN) if st["s00"] > 0 else None
        except FieldUnavailable:
            return None
        g = lambda f, k: f[k].values  # noqa: E731
        rain = g(p1, ("tp", 0)) - (g(p0, ("tp", 0)) if p0 is not None else 0.0)
        f, _ = day_flags(g(f06, ("t2m", 0)), g(f06, ("d2m", 0)), g(f15, ("t2m", 0)), g(f15, ("d2m", 0)),
                         g(f15, ("u10", 0)), g(f15, ("v10", 0)), rain)
        c = np.where(f, (c if c is not None else 0) + 1, 0)
        lats, lons = f15[("t2m", 0)].lats, f15[("t2m", 0)].lons
        d += dt.timedelta(days=1)
    return lats, lons, (24.0 * c).astype(np.float32), run
