"""Heavy rain in 1 hour («precipitação forte», mm/1 h): IPMA orange 21–40 mm, red > 40 mm.

Ensemble. The ECMWF ENS publishes precipitation only every 3 h, so here the ensemble is
the DWD ICON-EU-EPS: 40 members on an icosahedral grid (cell centres ≈ 10 km apart), with
hourly precipitation up to +48 h, runs every 6 h (opendata.dwd.de keeps only the latest
run of each cycle, ≈ 24 h). A file holds one step of all 40 members for all of Europe
(≈ 10 MB, bzip2): it is read whole and only the cells around Portugal are kept.

Coverage: the mainland and Madeira. The Azores are outside the ICON-EU domain (not
evaluated).

Sample points: the ICON cells whose centres lie inside each district (≈ 50 per district,
against ≈ 4 on the 0.25° lattice of the other phenomena) and, in Madeira, the cell
nearest each town of the area (the island has only ~10 land cells).

Screening, exact and cheap. Precipitation is accumulated from the start of the run, so a
member with ≥ 21 mm in one hour at a point also has ≥ 21 mm in the 6 h and the 24 h that
contain that hour. First the day totals, then the 6-h totals; only the 6-h windows where
some member reaches the orange threshold at some point get their hourly files. The hours
outside those windows stay unknown (all < 21 mm), so in the figures values below the
orange threshold are a lower bound; probabilities and values above it are exact.

Deterministic support: the models with hourly precipitation, ICON-EU (0.0625°) and GFS
(0.25°). IFS (3-hourly) and AIFS (6-hourly) cannot tell one hour.

Limitation. A model cell (10–28 km) gets the rain averaged over the cell: a thunderstorm
that pours 30–50 mm/h on a few km² gives the cell much less. The probabilities therefore
underestimate the downpours of isolated thunderstorms; what the models can see is
organised heavy rain (squall lines, fronts with embedded convection, cut-off lows).
"""
from __future__ import annotations

import asyncio
import bz2
import datetime as dt
import re
import shutil
from functools import lru_cache

import eccodes
import httpx
import numpy as np

from ..config import CACHE_DIR, HTTP_TIMEOUT, USER_AGENT
from ..models import MODELS, FieldUnavailable
from . import criteria as C
from .areas import PointSet, areas, district_geometry

H = dt.timedelta(hours=1)
BASE = "https://opendata.dwd.de/weather/nwp/icon-eu-eps/grib"
CACHE = CACHE_DIR / "alerts" / "iconeps"
KEEP = (-18.5, -5.5, 31.8, 42.8)          # lon0, lon1, lat0, lat1: mainland and Madeira
GRID_STEP = 0.0625                         # regular grid of the maps (nearest cell)
HOURLY_TO = 48                             # hourly steps up to +48 h (then 3 h, 6 h)
WINDOW = 6
N_MEMBERS = 40
DET = ("icon_eu", "gfs")                   # models with hourly precipitation
SOURCE = "DWD ICON-EU-EPS"
_SEM = asyncio.Semaphore(6)


# ---------------------------------------------------------------- files on opendata.dwd.de
def _url(run: dt.datetime, name: str, step: int | None = None) -> str:
    r = run.strftime("%Y%m%d%H")
    if step is None:
        return f"{BASE}/{run:%H}/{name}/icon-eu-eps_europe_icosahedral_time-invariant_{r}_{name}.grib2.bz2"
    return f"{BASE}/{run:%H}/{name}/icon-eu-eps_europe_icosahedral_single-level_{r}_{step:03d}_{name}.grib2.bz2"


async def _get(c: httpx.AsyncClient, url: str) -> bytes | None:
    async with _SEM:
        for attempt in range(3):
            try:
                r = await c.get(url)
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                return r.content
            except httpx.HTTPError:
                if attempt == 2:
                    raise
                await asyncio.sleep(2 + 3 * attempt)
    return None


def _decode(blob: bytes) -> list[tuple[int, np.ndarray]]:
    """[(member number, values)] of a bzip2 GRIB2 file (0 for a field without members)."""
    data = bz2.decompress(blob)
    out, i = [], 0
    while (j := data.find(b"GRIB", i)) >= 0:
        n = int.from_bytes(data[j + 8:j + 16], "big")
        h = eccodes.codes_new_from_message(data[j:j + n])
        try:
            num = int(eccodes.codes_get(h, "perturbationNumber")) if eccodes.codes_is_defined(
                h, "perturbationNumber") else 0
            out.append((num, eccodes.codes_get_values(h)))
        finally:
            eccodes.codes_release(h)
        i = j + n
    return out


async def runs_available() -> list[tuple[dt.datetime, list[int]]]:
    """[(run, steps)] on the server, newest first."""
    async def cycle(c, hh):
        try:
            r = await c.get(f"{BASE}/{hh:02d}/tot_prec/")
        except httpx.HTTPError:
            return None
        if r.status_code != 200:
            return None
        found = re.findall(r"single-level_(\d{10})_(\d{3})_tot_prec", r.text)
        if not found:
            return None
        run = max(x[0] for x in found)
        return (dt.datetime.strptime(run, "%Y%m%d%H"), sorted({int(s) for rr, s in found if rr == run}))

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT}) as c:
        res = await asyncio.gather(*(cycle(c, h) for h in (0, 6, 12, 18)))
    return sorted((x for x in res if x), key=lambda x: x[0], reverse=True)


async def latest_run() -> dt.datetime | None:
    """Newest run whose hourly steps are all published (up to +48 h)."""
    for run, steps in await runs_available():
        if all(s in steps for s in range(0, HOURLY_TO + 1)):
            return run
    return None


# ---------------------------------------------------------------- grid, points, map grid
class Grid:
    """The ICON cells kept (inside KEEP), the sample points and the map grid."""

    def __init__(self, z):
        self.n_total = int(z["n_total"])
        self.keep = z["keep"]                    # indices of the kept cells in the full grid
        self.lat, self.lon = z["lat"], z["lon"]  # of the kept cells
        self.pt_cell = z["pt_cell"]              # sample point -> kept cell
        self.pt_code = [str(x) for x in z["pt_code"]]
        self.glats, self.glons = z["glats"], z["glons"]
        self.gcell = z["gcell"]                  # (ny, nx) map node -> kept cell (-1: none)


async def grid() -> Grid:
    p = CACHE / "grid.npz"
    if not p.exists():
        await _build_grid(p)
    return Grid(np.load(p))


async def _build_grid(p) -> None:
    run = await latest_run()
    if run is None:
        raise FieldUnavailable("ICON-EU-EPS: nenhuma run completa no servidor")
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT}) as c:
        blat, blon = await asyncio.gather(_get(c, _url(run, "clat")), _get(c, _url(run, "clon")))
    if not blat or not blon:
        raise FieldUnavailable("ICON-EU-EPS: coordenadas das células em falta")
    clat = (await asyncio.to_thread(_decode, blat))[0][1]
    clon = (await asyncio.to_thread(_decode, blon))[0][1]
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **await asyncio.to_thread(_layout, clat, clon))


def _layout(clat: np.ndarray, clon: np.ndarray) -> dict:
    import shapely
    lon0, lon1, lat0, lat1 = KEEP
    keep = np.nonzero((clat >= lat0) & (clat <= lat1) & (clon >= lon0) & (clon <= lon1))[0]
    lat, lon = clat[keep], clon[keep]
    pt_cell, pt_code = [], []
    for a in areas():
        if a.group == "acores":
            continue
        g = district_geometry(a.code)
        if g is not None:
            idx = np.nonzero(shapely.contains_xy(g, lon, lat))[0]
        else:                                            # Madeira: the cell nearest each town
            idx = []
            for _, la, lo in a.towns:
                d = np.hypot(lat - la, (lon - lo) * np.cos(np.radians(la)))
                k = int(np.argmin(d))
                if k not in idx:
                    idx.append(k)
        pt_cell += [int(k) for k in idx]
        pt_code += [a.code] * len(idx)
    glats = np.arange(lat0, lat1 + 1e-6, GRID_STEP)
    glons = np.arange(lon0, lon1 + 1e-6, GRID_STEP)
    LO, LA = np.meshgrid(glons, glats)
    gcell = np.full(LA.size, -1, np.int32)
    flat_la, flat_lo = LA.ravel(), LO.ravel()
    coslat = np.cos(np.radians(lat))
    for s in range(0, LA.size, 2000):                    # nearest cell, by chunks
        la, lo = flat_la[s:s + 2000, None], flat_lo[s:s + 2000, None]
        d = (la - lat[None]) ** 2 + ((lo - lon[None]) * coslat[None]) ** 2
        k = d.argmin(1)
        ok = d[np.arange(len(k)), k] <= 0.12 ** 2
        gcell[s:s + 2000] = np.where(ok, k, -1)
    return {"n_total": np.int64(clat.size), "keep": keep.astype(np.int32), "lat": lat, "lon": lon,
            "pt_cell": np.array(pt_cell, np.int32), "pt_code": np.array(pt_code),
            "glats": glats, "glons": glons, "gcell": gcell.reshape(LA.shape)}


class RainPoints(PointSet):
    """The ICON cells of every area (see the module docstring)."""

    def __init__(self, g: Grid):
        grp = {a.code: a.group for a in areas()}
        super().__init__([(c, grp[c], float(g.lat[k]), float(g.lon[k]))
                          for c, k in zip(g.pt_code, g.pt_cell)], "rain1h")


_POINTS: RainPoints | None = None


def rain_points() -> RainPoints:
    """Sample points (needs the grid file: built on the first analysis)."""
    global _POINTS
    if _POINTS is None:
        p = CACHE / "grid.npz"
        if not p.exists():
            raise FieldUnavailable("ICON-EU-EPS: grelha ainda não descarregada")
        _POINTS = RainPoints(Grid(np.load(p)))
    return _POINTS


def thresholds() -> np.ndarray:
    n = len(rain_points())
    return np.stack([np.full(n, C.RAIN1["laranja"]), np.full(n, C.RAIN1["vermelho"])]).astype(np.float32)


# ---------------------------------------------------------------- member fields
def _cache(run: dt.datetime, step: int):
    return CACHE / run.strftime("%Y%m%d%H") / f"tp_{step:03d}.npz"


async def _step(c: httpx.AsyncClient, run: dt.datetime, step: int, g: Grid) -> np.ndarray | None:
    """(40, kept cells) precipitation accumulated from the start of the run (mm)."""
    if step == 0:
        return np.zeros((N_MEMBERS, len(g.keep)), np.float32)
    p = _cache(run, step)
    if p.exists():
        return np.load(p)["v"].astype(np.float32)
    blob = await _get(c, _url(run, "tot_prec", step))
    if blob is None:
        return None
    msgs = await asyncio.to_thread(_decode, blob)
    msgs.sort(key=lambda x: x[0])
    if len(msgs) != N_MEMBERS or msgs[0][1].size != g.n_total:
        raise FieldUnavailable(f"ICON-EU-EPS +{step} h: {len(msgs)} membros / grelha diferente")
    v = np.stack([m[1][g.keep] for m in msgs]).astype(np.float32)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".part.npz")
    np.savez_compressed(tmp, v=v)
    tmp.replace(p)
    return v


async def _steps(run: dt.datetime, steps, g: Grid) -> dict[int, np.ndarray]:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT}) as c:
        vals = await asyncio.gather(*(_step(c, run, s, g) for s in steps))
    return {s: v for s, v in zip(steps, vals) if v is not None}


def day_hours(run: dt.datetime, day: dt.date) -> tuple[int, int] | None:
    """Steps (a, b] of the run inside the UTC day, if they are all hourly; None otherwise."""
    t0 = dt.datetime(day.year, day.month, day.day)
    a, b = max(0, int((t0 - run) / H)), int((t0 + 24 * H - run) / H)
    if b <= a or b > HOURLY_TO:
        return None
    return a, b


async def ens_daily(ens_run: dt.datetime, out: dict, log) -> dict:
    """Per day: the largest 1-h precipitation of every member at every point ("pts"), the
    hourly series (NaN in the hours not fetched), the map grid. ``ens_run`` is the ECMWF
    run being analysed: the days start on its first day, steps are given relative to it."""
    from .detect import first_day
    run = await latest_run()
    if run is None:
        raise FieldUnavailable("ICON-EU-EPS: nenhuma run completa no servidor")
    g = await grid()
    out.setdefault("src_runs", {})["chuva1h"] = run.isoformat()
    d0 = first_day(ens_run)
    days = [d for d in (d0 + dt.timedelta(days=i) for i in range(3)) if day_hours(run, d)]
    pc = g.pt_cell
    thr = C.RAIN1["laranja"]
    # 1) day totals and 2) 6-h totals: which windows can hold an hour ≥ the orange threshold
    bounds = {d: day_hours(run, d) for d in days}
    tot = await _steps(run, sorted({x for a, b in bounds.values() for x in (a, b)}), g)
    screen, windows = {}, {}
    for d, (a, b) in bounds.items():
        if a not in tot or b not in tot:
            continue
        mx = float(np.max(tot[b][:, pc] - tot[a][:, pc]))
        screen[d.isoformat()] = round(mx, 1)
        if mx >= thr:
            edges = list(range(a, b, WINDOW)) + [b]
            windows[d] = list(zip(edges[:-1], edges[1:]))
    out["screen"]["chuva1h"] = screen
    if not windows:
        log(f"  chuva em 1 h: ICON-EU-EPS {run:%d %Hh}, nenhum membro com ≥ {thr + 0.5:.0f} mm "
            f"num dia ({', '.join(f'{k[8:]}: {v}' for k, v in screen.items())} mm)")
        return {}
    w6 = await _steps(run, sorted({x for ws in windows.values() for w in ws for x in w}), g)
    hot = {}
    for d, ws in windows.items():
        hot[d] = [(a, b) for a, b in ws if a in w6 and b in w6
                  and float(np.max(w6[b][:, pc] - w6[a][:, pc])) >= thr]
    need = sorted({s for ws in hot.values() for a, b in ws for s in range(a, b + 1)})
    log(f"  chuva em 1 h: ICON-EU-EPS {run:%d %Hh}, {sum(len(w) for w in hot.values())} janela(s) "
        f"de 6 h com ≥ {thr + 0.5:.0f} mm, {len(need)} passos × 40 membros")
    hr = await _steps(run, need, g)
    shift = int((run - ens_run) / H)
    res = {}
    for d, ws in hot.items():
        if not ws:
            continue
        a, b = bounds[d]
        series = np.full((b - a, N_MEMBERS, len(pc)), np.nan, np.float32)
        gmax = np.full((N_MEMBERS, len(g.keep)), np.nan, np.float32)
        for wa, wb in ws:
            for s in range(wa + 1, wb + 1):
                if s in hr and s - 1 in hr:
                    one = np.maximum(hr[s] - hr[s - 1], 0.0)
                    series[s - a - 1] = one[:, pc]
                    gmax = np.fmax(gmax, one)
        if not np.isfinite(series).any():
            continue
        with np.errstate(all="ignore"):
            pts = np.nanmax(series, 0)
        gridv = np.where(g.gcell >= 0, gmax[:, np.clip(g.gcell, 0, None)], np.nan)
        res[d] = {"pts": pts, "series": series, "steps": [shift + s for s in range(a + 1, b + 1)],
                  "grid": gridv.astype(np.float32), "lats": g.glats, "lons": g.glons,
                  "src_run": run.isoformat()}
    return res


def cleanup(keep: int = 4) -> None:
    if not CACHE.exists():
        return
    runs = sorted(d for d in CACHE.iterdir() if d.is_dir())
    for d in runs[:-keep]:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- deterministic models
class _Bilinear:
    """Bilinear weights of arbitrary points on a regular grid (lats/lons ascending)."""

    def __init__(self, glat, glon, plat, plon):
        glat, glon = np.asarray(glat, float), np.asarray(glon, float)
        j = np.clip(np.searchsorted(glat, plat) - 1, 0, len(glat) - 2)
        i = np.clip(np.searchsorted(glon, plon) - 1, 0, len(glon) - 2)
        ty = np.clip((plat - glat[j]) / (glat[j + 1] - glat[j]), 0, 1)
        tx = np.clip((plon - glon[i]) / (glon[i + 1] - glon[i]), 0, 1)
        nx = len(glon)
        self.flat = np.stack([j * nx + i, j * nx + i + 1, (j + 1) * nx + i, (j + 1) * nx + i + 1], 1)
        self.w = np.stack([(1 - ty) * (1 - tx), (1 - ty) * tx, ty * (1 - tx), ty * tx], 1).astype(np.float32)
        out = (plat < glat[0]) | (plat > glat[-1]) | (plon < glon[0]) | (plon > glon[-1])
        self.w[out] = np.nan

    def __call__(self, arr):
        a = np.asarray(arr, np.float32)
        return (a.reshape(a.shape[:-2] + (-1,))[..., self.flat] * self.w).sum(-1)


async def _det_hours(m, run: dt.datetime, day: dt.date):
    """(hours, ny, nx) 1-h precipitation of a model in the day, the grid and the run; None
    if the model has no hourly precipitation for the whole day."""
    from .detect import DOMAIN
    steps = await m.list_steps(run)
    t0 = dt.datetime(day.year, day.month, day.day)
    lo, hi = int((t0 - run) / H), int((t0 + 24 * H - run) / H)
    if hi <= 0 or not steps or hi > steps[-1]:
        return None
    lo = max(lo, 0)
    need = list(range(lo, hi + 1))
    if any(s not in steps for s in need if s > 0):
        return None
    sem = asyncio.Semaphore(4)
    got = {}

    async def one(s):
        async with sem:
            try:
                got[s] = (await m.fetch(run, s, [("tp", 0)], DOMAIN))[("tp", 0)]
            except FieldUnavailable:
                pass

    await asyncio.gather(*(one(s) for s in need))
    if any(s not in got for s in need):
        return None
    f0 = got[need[-1]]
    hours = np.stack([np.maximum(got[s].values - got[s - 1].values, 0.0) for s in need[1:]])
    return hours.astype(np.float32), f0.lats, f0.lons


async def deterministic(days: list[dt.date], log) -> dict:
    """{model: {"run", "name", "days": {day: (n_points,) largest 1-h precipitation}}}."""
    from .detect import det_run
    rp = rain_points()
    res = {}
    for mid in DET:
        m = MODELS[mid]
        try:
            run = await det_run(m)
            vals = {}
            for d in days:
                h = await _det_hours(m, run, d)
                if h is None:
                    continue
                hours, la, lo = h
                vals[d] = np.nanmax(_Bilinear(la, lo, rp.lats, rp.lons)(hours), 0)
        except Exception as e:  # noqa: BLE001 — a model that fails must not stop the alerts
            log(f"  {mid} (chuva em 1 h): {e}")
            continue
        if vals:
            res[mid] = {"run": run.isoformat(), "name": m.name, "days": vals}
    return res


async def det_grid(mid: str, day: dt.date):
    """Largest 1-h precipitation of the day on a model's grid (figures)."""
    from .detect import det_run
    m = MODELS[mid]
    run = await det_run(m)
    h = await _det_hours(m, run, day)
    if h is None:
        return None
    hours, la, lo = h
    return la, lo, hours.max(0), run
