"""Case studies of archived events: cyclone tracks, central pressure, event totals, synoptic
sequences and point series — all built from the event archive (and live data if present).

Definitions
-----------
* **Analysis sequence** — for every time ``t`` (3-hourly; 6-hourly for AIFS) the field from
  the run started at ``t`` (step 0) or from the run started 3 h earlier (step 3).
* **Short-range chain** — the first 6 h of every 6-hourly run, chained: precipitation over
  [r, r+6 h] from run r, gusts as the maximum over its 0–6 h steps. Event totals and point
  series use it (hourly for GFS and ICON-EU, 3-hourly for ECMWF, 6-hourly for AIFS).
* **Tracking** — starting from a seed (name, position, time: e.g. taken from the IPMA
  bulletin) the centre is followed forwards and backwards as the minimum of the smoothed
  MSLP within a search radius that grows with the time step.
"""
from __future__ import annotations

import asyncio
import datetime as dt

import numpy as np
from scipy.ndimage import gaussian_filter

from . import archive
from .grib import Field
from .models import MODELS, FieldUnavailable
from .regions import REGIONS

H = dt.timedelta(hours=1)
R_EARTH = 6371.0

# Default systems (seeds from the IPMA bulletins) for the events already created
DEFAULT_SYSTEMS = {
    "bruna": [{"name": "Bruna", "lat": 38.0, "lon": -22.0, "time": "2026-11-07T18:00"}],
    "charles": [{"name": "Charles", "lat": 34.0, "lon": -21.0, "time": "2026-11-11T00:00"}],
}


def systems(ev: dict) -> list[dict]:
    return ev.get("systems") or DEFAULT_SYSTEMS.get(ev["id"], [])


def _domain(model=None):
    """Event area (clipped to the model's coverage, which is how the archive stored it)."""
    from .models.base import Domain
    d = REGIONS["nea"].download_domain
    lo0, lo1, la0, la1 = d.lon0, d.lon1, d.lat0, d.lat1
    if model is not None and model.coverage:
        c = model.coverage
        lo0, lo1, la0, la1 = max(lo0, c[0]), min(lo1, c[1]), max(la0, c[2]), min(la1, c[3])
    return Domain(lo0, lo1, la0, la1)


async def _steps(model, run) -> list[int]:
    """Steps of a run: the event archive's when it holds the run (no server round trips for
    runs long gone from the servers), else the model's live listing."""
    st = archive.archived_steps(model.id, run)
    return sorted(st) if st else await model.list_steps(run)


async def _field(model, run, step, key):
    if run > archive._utcnow():
        return None                       # not run yet
    try:
        f = await model.fetch(run, step, [key], _domain(model))
    except FieldUnavailable:
        return None
    return f.get(key)


# ------------------------------------------------------------------ analysis sequence
def analysis_times(model, ev) -> list[dt.datetime]:
    t0, t1 = archive.parse_t(ev["start"]), archive.parse_t(ev["end"])
    t1 = min(t1, archive._utcnow())
    stepH = 6 if model.id == "aifs" else 3
    t = t0.replace(minute=0, second=0, microsecond=0)
    t = t.replace(hour=(t.hour // stepH) * stepH)
    out = []
    while t <= t1:
        out.append(t)
        t += stepH * H
    return out


async def analysis_field(model, t: dt.datetime, key):
    """Field valid at t from the most recent run (step 0, else step 3 of the run 3 h earlier)."""
    r = t.replace(hour=(t.hour // 6) * 6)
    step = int((t - r).total_seconds() // 3600)
    f = await _field(model, r, step, key)
    if f is None and step == 0:
        f = await _field(model, r - 6 * H, 6, key)
    return f


# ------------------------------------------------------------------ tracking
def _dist_km(lat0, lon0, lats, lons):
    la0, lo0 = np.radians(lat0), np.radians(lon0)
    la, lo = np.radians(lats), np.radians(lons)
    a = np.sin((la - la0) / 2) ** 2 + np.cos(la0) * np.cos(la) * np.sin((lo - lo0) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def find_centre(f: Field, lat0, lon0, radius_km):
    """Minimum of the smoothed MSLP within radius_km of (lat0, lon0) -> (lat, lon, hPa) or None."""
    res = abs(float(f.lats[1] - f.lats[0])) if len(f.lats) > 1 else 0.25
    v = np.asarray(f.values, dtype=float)
    if np.isnan(v).all():
        return None
    v = np.where(np.isnan(v), np.nanmax(v), v)
    sm = gaussian_filter(v, max(1.0, 0.4 / res))        # ~0.4° smoothing on any grid
    LA, LO = np.meshgrid(f.lats, f.lons, indexing="ij")
    d = _dist_km(lat0, lon0, LA, LO)
    inside = d <= radius_km
    if not inside.any():
        return None
    j, i = np.unravel_index(np.argmin(np.where(inside, sm, np.inf)), sm.shape)
    if d[j, i] > 0.92 * radius_km:
        return None                                      # minimum on the rim: no closed low
    e = max(2, int(0.75 / res))
    if j < e or i < e or j >= sm.shape[0] - e or i >= sm.shape[1] - e:
        return None                                      # at the edge of the model's data
    # must be a real local minimum of the smoothed field (± ~1.5°)
    k = max(2, int(1.5 / res))
    win = sm[max(0, j - k):j + k + 1, max(0, i - k):i + k + 1]
    if sm[j, i] > win.min() + 1e-6:
        return None
    # central pressure: raw minimum within ~60 km of the smoothed centre
    near = _dist_km(LA[j, i], LO[j, i], LA, LO) <= 60
    return float(LA[j, i]), float(LO[j, i]), float(np.min(v[near]) / 100.0)


def _speed_radius(hours: float) -> float:
    return max(350.0, 110.0 * hours)         # up to ~110 km/h displacement (Charles: ~90 km/h)


def _local_minima(f: Field, lat0, lon0, radius_km):
    """Closed lows of the smoothed MSLP within radius_km: [(lat, lon, hPa, km)]."""
    from scipy.ndimage import minimum_filter
    res = abs(float(f.lats[1] - f.lats[0])) if len(f.lats) > 1 else 0.25
    v = np.asarray(f.values, dtype=float)
    if np.isnan(v).all():
        return []
    v = np.where(np.isnan(v), np.nanmax(v), v)
    sm = gaussian_filter(v, max(1.0, 0.4 / res))
    k = max(2, int(1.5 / res))
    LA, LO = np.meshgrid(f.lats, f.lons, indexing="ij")
    d = _dist_km(lat0, lon0, LA, LO)
    mask = (sm == minimum_filter(sm, size=2 * k + 1, mode="nearest")) & (d <= 0.92 * radius_km)
    e = max(2, int(0.75 / res))
    mask[:e], mask[-e:], mask[:, :e], mask[:, -e:] = False, False, False, False   # data edge
    out = []
    for j, i in zip(*np.nonzero(mask)):
        near = _dist_km(LA[j, i], LO[j, i], LA, LO) <= 60
        out.append((float(LA[j, i]), float(LO[j, i]), float(np.min(v[near]) / 100.0), float(d[j, i])))
    return out


def _jump(f, lat, lon, hrs, last_p, motion):
    """The centre may re-form ahead of the old one (or move very fast): look farther for a
    closed low roughly along the recent direction of motion (±75°) with a similar central
    pressure; the nearest such low is taken (a deeper neighbouring system is not grabbed)."""
    if motion is None or np.hypot(*motion) < 0.3:
        return None
    best = None
    for la, lo, p, d in _local_minima(f, lat, lon, min(950.0, 2.2 * _speed_radius(hrs))):
        if abs(p - last_p) > max(8.0, 2.0 * hrs):
            continue
        dy, dx = la - lat, (lo - lon) * np.cos(np.radians(lat))
        cosang = (dx * motion[1] + dy * motion[0]) / (np.hypot(dx, dy) * np.hypot(*motion) + 1e-9)
        if cosang >= np.cos(np.radians(75)) and (best is None or d < best[3]):
            best = (la, lo, p, d)
    return best[:3] if best else None


def _next_centre(f, lat, lon, hrs, last_p, motion, last_hrs):
    """Next position of a tracked low: the closed low nearest to where its recent motion takes
    it (within the speed radius) — not simply the lowest pressure in the search circle, which
    can be a deeper neighbour; failing that, a centre re-formed farther along the motion."""
    elat, elon = lat, lon
    if motion is not None and last_hrs:
        k = hrs / last_hrs
        elat = lat + motion[0] * k
        elon = lon + motion[1] * k / max(0.2, float(np.cos(np.radians(lat))))
    dp = max(12.0, 3.0 * hrs)            # explosive deepening reaches ~8 hPa in 6 h
    cands = [c for c in _local_minima(f, lat, lon, _speed_radius(hrs)) if abs(c[2] - last_p) <= dp]
    if cands:
        return min(cands, key=lambda c: float(_dist_km(elat, elon, c[0], c[1])))[:3]
    return _jump(f, lat, lon, hrs, last_p, motion)


def _motion(prev, cur):
    if prev is None:
        return None
    return (cur[0] - prev[0], (cur[1] - prev[1]) * np.cos(np.radians(cur[0])))


def _ref_pos(ref, t):
    """Position of a reference track at the time nearest to t (within 3 h)."""
    best = min(ref, key=lambda p: abs(archive.parse_t(p["time"]) - t), default=None)
    if best and abs(archive.parse_t(best["time"]) - t) <= 3 * H:
        return best["lat"], best["lon"]
    return None


async def track_analysis(model, ev, system, ref: list[dict] | None = None) -> list[dict]:
    """Track in this model's analyses; ``ref`` (another model's track) seeds the search when
    this model has no data at the seed time (e.g. ICON-EU, kept only for the last 24 h)."""
    times = analysis_times(model, ev)
    if not times:
        return []
    seed_t = archive.parse_t(system["time"])
    i0 = int(np.argmin([abs((t - seed_t).total_seconds()) for t in times]))
    fields = await asyncio.gather(*(analysis_field(model, t, ("msl", 0)) for t in times))
    track = {}

    def follow(order, motion0=None, hrs0=None):
        lat, lon = system["lat"], system["lon"]
        last_t, last_p, motion, last_hrs = None, None, None, None
        for k in order:
            f = fields[k]
            if f is None:
                continue
            if last_t is None and ref:
                rp = _ref_pos(ref, times[k])
                if rp is None:
                    continue
                lat, lon = rp
            hrs = abs((times[k] - last_t).total_seconds()) / 3600 if last_t else 6
            if last_t is None:
                c = find_centre(f, lat, lon, 300.0 if ref else 500.0)
            else:
                c = _next_centre(f, lat, lon, hrs, last_p, motion, last_hrs)
            if c is None:
                if last_t is None:
                    continue
                break
            motion = _motion((lat, lon) if last_t else None, c)
            if motion is None and motion0 is not None:
                motion, hrs = motion0, hrs0          # backward pass: reversed forward motion
            lat, lon, p = c
            track[k] = {"time": times[k].isoformat(), "lat": lat, "lon": lon, "p": p}
            last_t, last_p, last_hrs = times[k], p, hrs

    if ref and fields[i0] is None:
        # no data at the seed time: start where this model's data begin
        i0 = next((k for k, f in enumerate(fields) if f is not None and _ref_pos(ref, times[k])), i0)
    follow(range(i0, len(times)))
    ks = sorted(k for k in track if k >= i0)
    m0 = h0 = None
    if len(ks) >= 2:
        a, b = track[ks[0]], track[ks[1]]
        m0 = _motion((b["lat"], b["lon"]), (a["lat"], a["lon"]))          # b → a: backwards
        h0 = (times[ks[1]] - times[ks[0]]).total_seconds() / 3600
    follow(range(i0, -1, -1), m0, h0)
    return [track[k] for k in sorted(track)]


async def track_forecast(model, run, ev, start: dict) -> list[dict]:
    """Follow the low in one forecast run, starting from an analysed position."""
    t1 = archive.parse_t(ev["end"])
    st0 = int((archive.parse_t(start["time"]) - run).total_seconds() // 3600)
    if st0 < 0:
        return []
    steps = [s for s in await _steps(model, run) if s >= st0 and (run + s * H) <= t1]
    steps = [s for s in steps if s % 6 == st0 % 6] or steps
    out, lat, lon, last, last_p, motion, last_hrs = [], start["lat"], start["lon"], None, None, None, None
    for s in steps:
        f = await _field(model, run, s, ("msl", 0))
        if f is None:
            continue
        hrs = (s - last) if last is not None else 0
        if last is None:
            c = find_centre(f, lat, lon, 300.0)
        else:
            c = _next_centre(f, lat, lon, hrs, last_p, motion, last_hrs)
        if c is None:
            break
        motion = _motion((lat, lon) if last is not None else None, c)
        lat, lon, p = c
        last_p, last_hrs = p, hrs
        out.append({"time": (run + s * H).isoformat(), "lat": lat, "lon": lon, "p": p, "step": s})
        last = s
    return out


def focus_time(system: dict, ref: list[dict]):
    """Moment the forecasts are verified against: the system's ``focus`` (e.g. landfall)
    when set, else the deepest analysed point."""
    if system.get("focus"):
        return archive.parse_t(system["focus"])
    return archive.parse_t(min(ref, key=lambda p: p["p"])["time"]) if ref else None


def select_forecasts(fcs: dict, ref: list[dict], leads=(24, 48, 72, 96), focus=None) -> dict:
    """Keep the runs closest to 24/48/72/96 h before the focus time (default: the deepest
    analysed point)."""
    if not fcs or not ref:
        return {}
    peak = focus or archive.parse_t(min(ref, key=lambda p: p["p"])["time"])
    out = {}
    for L in leads:
        target = peak - L * H
        best = min(fcs, key=lambda r: abs(archive.parse_t(r) - target))
        lead = int((peak - archive.parse_t(best)).total_seconds() // 3600)
        if abs(lead - L) <= 12 and best not in out and lead > 0:
            out[best] = (fcs[best], lead)
    return out


async def forecast_runs(model, ev, analysis_track: list[dict]) -> dict:
    """{run iso: track} for the archived 00/12Z forecast runs started before the low ends."""
    if not analysis_track:
        return {}
    t_last = archive.parse_t(analysis_track[-1]["time"])
    runs = [r for r in archive.archived_runs(model.id) if r.hour in (0, 12) and r < t_last
            and r >= archive.parse_t(ev["start"]) - dt.timedelta(days=float(ev.get("lead_days", 2)) + 0.3)]
    out = {}
    for r in sorted(runs):
        # first analysed point at or after the run start
        start = next((p for p in analysis_track if archive.parse_t(p["time"]) >= r), None)
        if start is None:
            continue
        tr = await track_forecast(model, r, ev, start)
        if len(tr) >= 2:
            out[r.isoformat()] = tr
    return out


# ------------------------------------------------------------------ short-range chain
def chain_runs(ev) -> list[dt.datetime]:
    """6-hourly runs whose first 6 h fall inside the event window."""
    t0, t1 = archive.parse_t(ev["start"]), archive.parse_t(ev["end"])
    r = t0.replace(minute=0, second=0, microsecond=0)
    r = r.replace(hour=(r.hour // 6) * 6)
    if r < t0:
        r += 6 * H
    out = []
    while r + 6 * H <= t1:
        out.append(r)
        r += 6 * H
    return out


async def event_totals(model, ev, kind: str) -> tuple[Field | None, dict]:
    """kind 'precip': total precipitation (mm) from the chained 0–6 h forecasts;
    'gust': maximum gust (km/h) over the chained steps."""
    runs = chain_runs(ev)
    acc, grid, used, missing = None, None, 0, 0
    for r in runs:
        if kind == "precip":
            f = await _field(model, r, 6, ("tp", 0))
            arr = None if f is None else np.clip(f.values, 0, None)
        else:
            if not model.has("gust"):
                return None, {"error": f"{model.name} não fornece rajadas"}
            parts = []
            for s in sorted(set(archive.archived_steps(model.id, r)) & set(range(1, 7))):
                g = await _field(model, r, s, ("gust", 0))
                if g is not None:
                    parts.append(g.values * 3.6)
                    f = g
            arr = np.nanmax(np.stack(parts), 0) if parts else None
        if arr is None:
            missing += 1
            continue
        if grid is None:
            grid = f
            acc = arr.astype(np.float64)
        elif arr.shape == acc.shape:
            acc = acc + arr if kind == "precip" else np.fmax(acc, arr)
        used += 1
    if grid is None:
        return None, {"used": 0, "missing": missing}
    return Field(grid.lats, grid.lons, acc.astype(np.float32)), {"used": used, "missing": missing,
                                                                 "runs": len(runs)}


async def point_chain(model, ev, lat, lon) -> dict:
    """Analysis-chain series at a point: state variables from steps 0–5 of every 6-hourly run,
    precipitation per interval from the run's cumulative total (steps 1–6)."""
    from .models.base import Domain
    if not model.covers(lat, lon):
        raise FieldUnavailable(model.outside_msg())
    dom = Domain.tile(lat, lon)
    state, precip = {}, {}
    keys = [(n, 0) for n in ("msl", "u10", "v10", "gust", "tp", "t2m") if model.has(n)]
    for r in chain_runs(ev):
        steps = sorted(set(await _steps(model, r)) & set(range(0, 7)))
        prev_s, prev_tp = 0, 0.0
        for s in steps:
            try:
                f = await model.fetch(r, s, keys if s < 6 else [("tp", 0)], dom)
            except FieldUnavailable:
                continue
            t = r + s * H
            if s < 6:
                state[t] = {k[0]: fld.at(lat, lon) for k, fld in f.items()}
                if s == 0:
                    state[t].pop("gust", None)     # gusts are maxima over the previous interval
            if s > 0 and ("tp", 0) in f:
                tp = f[("tp", 0)].at(lat, lon)
                if np.isfinite(tp):
                    precip[t] = (max(0.0, tp - prev_tp), s - prev_s)
                    prev_s, prev_tp = s, tp
    times = sorted(set(state) | set(precip))
    g = lambda t, k: state.get(t, {}).get(k, np.nan)  # noqa: E731
    return {
        "time": times,
        "msl": [g(t, "msl") / 100.0 for t in times],
        "wind": [float(np.hypot(g(t, "u10"), g(t, "v10")) * 3.6) for t in times],
        "gust": [g(t, "gust") * 3.6 for t in times],
        "t2m": [g(t, "t2m") - 273.15 for t in times],
        "precip": [precip.get(t, (np.nan, 0))[0] for t in times],
        "precip_h": [precip.get(t, (np.nan, 0))[1] for t in times],
    }


async def all_tracks(ev, models: list[str]) -> dict:
    """{system: {model: track}} — ECMWF (else GFS) first, used as reference for the others."""
    out = {}
    order = [m for m in ("ecmwf", "gfs", "aifs", "icon_eu") if m in models]
    for sysd in systems(ev):
        tracks, ref = {}, None
        for mid in order:
            tr = await track_analysis(MODELS[mid], ev, sysd, ref)
            tracks[mid] = tr
            if ref is None and len(tr) >= 3:
                ref = tr
        out[sysd["name"]] = tracks
    return out


# ================================================================== chart jobs
def _fmt(t: dt.datetime) -> str:
    from .maps.compose import fmt_time
    return fmt_time(t)


def _label(t_iso: str) -> str:
    t = archive.parse_t(t_iso)
    return f"{t.day:02d}/{t:%H}Z"


TRACK_STYLE = {"ecmwf": ("#dc2626", "o"), "gfs": ("#2563eb", "s"), "aifs": ("#16a34a", "D"),
               "icon_eu": ("#9333ea", "^")}


async def build_track_job(ev, models: list[str], fc_model: str | None, region="nea") -> dict:
    """Analysed tracks of every system (one colour per model) over the event's minimum MSLP;
    optionally the forecast tracks of fc_model's 00/12Z runs (thin, lighter = older)."""
    from .maps.styles import Fill, _interp
    tracks = await all_tracks(ev, models)
    # background: minimum MSLP over the event (reference model analyses)
    refm = MODELS["ecmwf" if "ecmwf" in models else models[0]]
    mins, grid = None, None
    for t in analysis_times(refm, ev):
        f = await analysis_field(refm, t, ("msl", 0))
        if f is None:
            continue
        v = f.values / 100.0
        if mins is None:
            mins, grid = v.copy(), f
        elif v.shape == mins.shape:
            mins = np.fmin(mins, v)
    if grid is None:
        raise FieldUnavailable("sem análises arquivadas para este evento")
    lv = [960, 965, 970, 975, 980, 985, 990, 995, 1000, 1005, 1010]
    fill = Fill(lv, _interp(["#3b0764", "#6d28d9", "#2563eb", "#0ea5e9", "#67e8f9", "#d1fae5",
                             "#f8fafc", "#f8fafc", "#f8fafc", "#ffffff"], len(lv) - 1),
                extend="min", under="#1e0636", transparent_under=False)
    fill.colors[-4:] = [(1, 1, 1, 0)] * 4          # >= 990 hPa transparent: land/sea visible
    layers = [{"type": "fill", "var": "msl", "data": mins.astype(np.float32), "fill_style": fill,
               "label": f"Pressão mínima (MSL) atingida durante o evento — {refm.name} (hPa)"}]
    for sysname, per in tracks.items():
        for mid, tr in per.items():
            if not tr:
                continue
            col, mk = TRACK_STYLE.get(mid, ("#111", "o"))
            pts = []
            for k, p in enumerate(tr):
                t = archive.parse_t(p["time"])
                lab = f"{_label(p['time'])} {p['p']:.0f}" if t.hour in (0, 12) and mid == refm.id else None
                pts.append((p["lon"], p["lat"], lab))
            layers.append({"type": "track", "points": pts, "color": col, "marker": mk,
                           "label": f"{sysname} · {MODELS[mid].name} (análises)", "lw": 2.0})
    if fc_model and fc_model in MODELS:
        fm = MODELS[fc_model]
        for sysname, per in tracks.items():
            ref = per.get(fc_model) or next((v for v in per.values() if v), [])
            sysd = next((x for x in systems(ev) if x["name"] == sysname), {})
            fz = focus_time(sysd, ref)
            fcs = select_forecasts(await forecast_runs(fm, ev, ref), ref, focus=fz)
            greys = ["#7f1d1d", "#b91c1c", "#f97316", "#facc15"]   # recent -> older
            for k, (r, (tr, lead)) in enumerate(sorted(fcs.items(), reverse=True)):
                layers.append({"type": "track", "points": [(p["lon"], p["lat"], None) for p in tr],
                               "color": greys[k % 4], "ls": (0, (4, 2)), "lw": 1.4, "ms": 2.5,
                               "label": f"{sysname} · previsão {fm.name} {_label(r)} (−{lead} h de {_label(fz.isoformat())})"})
    names = " e ".join(tracks)
    return {"region": region, "lats": grid.lats, "lons": grid.lons, "layers": layers,
            "header": {"left1": f"Trajetórias: {names}",
                       "left2": f"Evento «{ev['name']}» · {_fmt(archive.parse_t(ev['start']))} → {_fmt(archive.parse_t(ev['end']))}",
                       "layers": "Centro = mínimo da pressão MSL suavizada em cada análise (3/3 h; AIFS 6/6 h); "
                                 "etiquetas: dia/hora e pressão central (00 e 12Z)"
                                 + (f" · tracejado: previsões {MODELS[fc_model].name}" if fc_model else ""),
                       "right1": "Análise de caso", "right2": REGIONS[region].name},
            "footer": "Dados: arquivo Forecast Workstation — " + " · ".join(sorted({MODELS[m].source for m in models}))}, tracks


async def build_totals_job(ev, model_id: str, kind: str, region: str) -> dict:
    from .maps.styles import GUST_COLORS, GUST_LEVELS, Fill, _interp
    m = MODELS[model_id]
    f, info = await event_totals(m, ev, kind)
    if f is None:
        raise FieldUnavailable(info.get("error") or f"{m.name}: sem dados arquivados para os totais")
    if kind == "precip":
        lv = [1, 5, 10, 20, 30, 40, 50, 75, 100, 125, 150, 200, 250, 300, 400]
        fill = Fill(lv, _interp(["#c6e9fb", "#6cb6ef", "#1f5fc7", "#14a33a", "#a5dc2b", "#f2ec2c",
                                 "#f99d1c", "#e2231a", "#8b0a50", "#b02fa3", "#f0c4f2"], len(lv) - 1),
                    extend="max", over="#ffffff", transparent_under=True)
        title, unit = "Precipitação total no evento", "mm"
        how = "soma das previsões 0–6 h de cada run de 6 em 6 h (cadeia de curto prazo)"
    else:
        fill = Fill(GUST_LEVELS, GUST_COLORS, extend="max", over="#ffe0ff", transparent_under=True)
        title, unit = "Rajada máxima no evento", "km/h"
        how = ("máximo das rajadas nos passos 0–6 h de cada run (GFS/ICON: horário; ECMWF: máx. 3 h)")
    tracks = await all_tracks(ev, [model_id])
    layers = [{"type": "fill", "var": "t2m", "data": f.values, "fill_style": fill,
               "label": f"{title} — {m.name} ({unit})"},
              {"type": "values", "var": "t2m", "data": f.values, "fmt": "%.0f"}]
    for sysname, per in tracks.items():
        tr = per.get(model_id) or []
        if tr:
            layers.append({"type": "track", "points": [(p["lon"], p["lat"], None) for p in tr],
                           "color": "#111111", "lw": 1.4, "ms": 3, "label": f"Trajetória {sysname}"})
    miss = f" · {info['missing']} de {info['runs']} runs em falta" if info.get("missing") else ""
    return {"region": region, "lats": f.lats, "lons": f.lons, "layers": layers,
            "header": {"left1": f"{title} · {m.name}",
                       "left2": f"Evento «{ev['name']}» · {_fmt(archive.parse_t(ev['start']))} → {_fmt(archive.parse_t(ev['end']))}{miss}",
                       "layers": how, "right1": "Análise de caso", "right2": REGIONS[region].name},
            "footer": f"Dados: arquivo Forecast Workstation — {m.source}"}


async def sequence_data(ev, model_id: str, every: int, region: str) -> dict:
    """Panels of MSLP + 6-h precipitation every `every` hours (analysis chain)."""
    m = MODELS[model_id]
    t0, t1 = archive.parse_t(ev["start"]), min(archive.parse_t(ev["end"]), archive._utcnow())
    t = t0.replace(hour=(t0.hour // 6) * 6, minute=0, second=0, microsecond=0)
    if t < t0:
        t += 6 * H
    panels = []
    tracks = await all_tracks(ev, [model_id])
    while t <= t1 and len(panels) < 24:
        msl = await analysis_field(m, t, ("msl", 0))
        pr = await _field(m, t - 6 * H, 6, ("tp", 0))       # precipitation in the 6 h before t
        if msl is not None:
            centres = []
            for sysname, per in tracks.items():
                for p in per.get(model_id) or []:
                    if archive.parse_t(p["time"]) == t:
                        centres.append((p["lon"], p["lat"], sysname, p["p"]))
            panels.append({"time": t, "lats": msl.lats, "lons": msl.lons,
                           "msl": (msl.values / 100.0).astype(np.float32),
                           "pr": None if pr is None else np.clip(pr.values, 0, None).astype(np.float32),
                           "pr_lats": None if pr is None else pr.lats,
                           "pr_lons": None if pr is None else pr.lons,
                           "centres": centres})
        t += every * H
    return {"model": m.name, "source": m.source, "region": region, "event": ev["name"],
            "panels": panels}


# ================================================================== soundings & meteogram
async def case_sounding_items(ev, t: dt.datetime, lat: float, lon: float, sid: str | None,
                              models: list[str]) -> list[dict]:
    """Observed radiosonde (if any) + every model's analysis profile at time t."""
    from . import obs
    from .sounding.profile import get_profile
    items = []
    if sid:
        prof = await obs.get_obs(sid, t)
        if prof:
            snd = obs.obs_as_sounding(prof)
            items.append({"model": "obs", "name": f"Radiossonda {prof['name']}", "short": "Observada",
                          "run": t, "step": 0, "snd": snd, "obs": True, "source": prof["source"]})
    for mid in models:
        m = MODELS[mid]
        if not m.covers(lat, lon):
            continue
        try:
            snd = await get_profile(m, t, 0, lat, lon)       # analysis = step 0 of the run at t
        except FieldUnavailable:
            continue
        if len(snd["profile"]["p"]) < 5:
            continue
        items.append({"model": mid, "name": f"{m.name} (análise)", "short": m.name.replace("ECMWF ", ""),
                      "run": t, "step": 0, "snd": snd})
    return items


async def case_meteogram_data(ev, model_id: str, lat: float, lon: float, name: str) -> dict:
    """Time-height section from the 6-hourly analyses + surface panels from the chain."""
    from . import obs
    from .models.base import Domain
    m = MODELS[model_id]
    if not m.covers(lat, lon):
        raise FieldUnavailable(m.outside_msg())
    t0, t1 = archive.parse_t(ev["start"]), min(archive.parse_t(ev["end"]), archive._utcnow())
    levels = [p for p in m.levels if p >= 200]
    hum = "r" if m.has("r") else "q"
    dom = Domain.tile(lat, lon)
    r = t0.replace(hour=(t0.hour // 6) * 6, minute=0, second=0, microsecond=0)
    if r < t0:
        r += 6 * H
    times, cols = [], []
    while r <= t1:
        keys = [(n, p) for p in levels for n in ("t", hum, "u", "v")] + [("sp", 0)]
        try:
            f = await m.fetch(r, 0, keys, dom)
        except FieldUnavailable:
            f = {}
        if f and ("t", levels[0]) in f:
            col = {(n, p): f[(n, p)].at(lat, lon) if (n, p) in f else np.nan
                   for p in levels for n in ("t", hum, "u", "v")}
            col["sp"] = f[("sp", 0)].at(lat, lon) / 100 if ("sp", 0) in f else np.nan
            times.append(r)
            cols.append(col)
        r += 6 * H
    prof = None
    if times:
        A = lambda n: np.array([[c[(n, p)] for c in cols] for p in levels])  # noqa: E731
        prof = {"t": A("t"), "u": A("u"), "v": A("v"),
                "r": A("r") if hum == "r" else None, "q": A("q") if hum == "q" else None,
                "sp": np.array([c["sp"] for c in cols])}
        if hum == "q":
            from .products import rh_from_q
            prof["r"] = rh_from_q(prof["t"], prof["q"], np.array(levels)[:, None])
    chain = await point_chain(m, ev, lat, lon)
    sid, dkm = obs.nearest_station(lat, lon)
    obs_times = []
    if sid:
        for t in obs.event_obs_times(ev):
            if obs._path(sid, t).exists():
                obs_times.append(t)
    return {"event": ev["name"], "model": m.name, "model_id": model_id, "source": m.source,
            "place": name, "lat": lat, "lon": lon, "levels": levels, "times": times,
            "profile": prof, "chain": chain,
            "obs": {"station": obs.station_info(sid)["name"] if sid else None,
                    "km": round(dkm), "times": obs_times}}
