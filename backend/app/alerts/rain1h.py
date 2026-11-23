"""Heavy rain in 1 hour («precipitação forte», mm/1 h): IPMA orange 21–40 mm, red > 40 mm.

Ensemble. The ECMWF ENS publishes precipitation only every 3 h, so here the ensemble is
the DWD ICON-EU-EPS: 40 members on an icosahedral grid (cell centres ≈ 10 km apart), with
hourly precipitation up to +48 h, runs every 6 h (opendata.dwd.de keeps only the latest
run of each cycle, ≈ 24 h). A file holds one step of all 40 members for all of Europe
(≈ 10 MB, bzip2): it is read whole and only the cells around Portugal are kept.

Coverage: the mainland and Madeira from the ICON-EU-EPS. The Azores lie outside the ICON-EU
domain: there the ensemble is the global ICON-EPS of the same run (40 members, cells ≈ 20 km
apart, hourly precipitation to +48 h in the 00 and 12 UTC runs only). A global file is
≈ 32 MB per step (a rainy day can take 10–15 of them), so it is read only when the ECMWF ENS
gives ≥ 20 % chance of 25 mm in 24 h at an Azores point on one of the days (the watch level:
a downpour that a fifth of the ICON members put in one hour hardly comes without it); and the
run is chosen so that both ensembles come from the same one (normally the 00 or 12 UTC run,
the newest at the ECMWF analysis).

Sample points: the ICON cells whose centres lie inside each district (≈ 50 per district,
against ≈ 4 on the 0.25° lattice of the other phenomena) and, in Madeira and the Azores,
the cell nearest each town of the area (Madeira has only ~10 land cells; in the global
ICON every town is within 14 km of a cell centre).

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
BASE_G = "https://opendata.dwd.de/weather/nwp/icon-eps/grib"     # global ICON-EPS (the Azores)
CACHE = CACHE_DIR / "alerts" / "iconeps"
KEEP = (-18.5, -5.5, 31.8, 42.8)          # lon0, lon1, lat0, lat1: mainland and Madeira
KEEP_G = (-32.0, -24.5, 36.5, 40.2)       # the Azores, from the global ICON-EPS
MAP = (-32.0, -5.5, 31.8, 42.8)           # the map grid covers both
AZ_SCREEN = 20.0                          # % — ECMWF P(≥ 25 mm in 24 h) to read the global files
GRID_STEP = 0.0625                         # regular grid of the maps (nearest cell)
HOURLY_TO = 48                             # hourly steps up to +48 h (then 3 h, 6 h)
WINDOW = 6
N_MEMBERS = 40
DET = ("icon_eu", "gfs")                   # models with hourly precipitation
SOURCE = "DWD ICON-EU-EPS"
SOURCE_G = "DWD ICON-EPS"
_SEM = asyncio.Semaphore(6)


# ---------------------------------------------------------------- files on opendata.dwd.de
def _url(run: dt.datetime, name: str, step: int | None = None, glob: bool = False) -> str:
    r = run.strftime("%Y%m%d%H")
    base, model, dom = (BASE_G, "icon-eps", "global") if glob else (BASE, "icon-eu-eps", "europe")
    if step is None:
        return f"{base}/{run:%H}/{name}/{model}_{dom}_icosahedral_time-invariant_{r}_{name}.grib2.bz2"
    return f"{base}/{run:%H}/{name}/{model}_{dom}_icosahedral_single-level_{r}_{step:03d}_{name}.grib2.bz2"


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


async def runs_available(glob: bool = False) -> list[tuple[dt.datetime, list[int]]]:
    """[(run, steps)] on the server, newest first (``glob``: the global ICON-EPS, whose hourly
    steps exist only in the 00 and 12 UTC runs)."""
    async def cycle(c, hh):
        try:
            r = await c.get(f"{BASE_G if glob else BASE}/{hh:02d}/tot_prec/")
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
        res = await asyncio.gather(*(cycle(c, h) for h in ((0, 12) if glob else (0, 6, 12, 18))))
    return sorted((x for x in res if x), key=lambda x: x[0], reverse=True)


def _complete(steps) -> bool:
    return all(s in steps for s in range(0, HOURLY_TO + 1))


async def latest_run() -> dt.datetime | None:
    """Newest ICON-EU-EPS run whose hourly steps are all published (up to +48 h)."""
    for run, steps in await runs_available():
        if _complete(steps):
            return run
    return None


async def pick_run() -> tuple[dt.datetime | None, bool]:
    """(run, the global ICON-EPS has it too): the newest complete ICON-EU-EPS run, or the one
    before it when only that one also has the global run (so that the Azores and the rest come
    from the same run)."""
    eu = [r for r, st in await runs_available() if _complete(st)]
    if not eu:
        return None, False
    try:
        gl = {r for r, st in await runs_available(glob=True) if _complete(st)}
    except Exception:  # noqa: BLE001 — without the global ICON-EPS the Azores are just left out
        gl = set()
    for r in eu[:2]:
        if r in gl:
            return r, True
    return eu[0], False


# ---------------------------------------------------------------- grid, points, map grid
class Grid:
    """The ICON cells kept — those of the ICON-EU-EPS inside KEEP, then those of the global
    ICON-EPS inside KEEP_G — the sample points and the map grid."""

    def __init__(self, z):
        self.n_total = int(z["n_total"])         # cells of the full ICON-EU grid
        self.keep = z["keep"]                    # indices of its kept cells
        self.n_total_g = int(z["n_total_g"])     # cells of the full global grid
        self.keep_g = z["keep_g"]                # indices of its kept cells (the Azores)
        self.n_eu = len(self.keep)               # kept cells: [ICON-EU …, global …]
        self.lat, self.lon = z["lat"], z["lon"]  # of the kept cells
        self.pt_cell = z["pt_cell"]              # sample point -> kept cell
        self.pt_code = [str(x) for x in z["pt_code"]]
        self.glats, self.glons = z["glats"], z["glons"]
        self.gcell = z["gcell"]                  # (ny, nx) map node -> kept cell (-1: none)

    @property
    def n_cells(self) -> int:
        return self.n_eu + len(self.keep_g)

    def pt_glob(self) -> np.ndarray:
        """True for the sample points read from the global ICON-EPS (the Azores)."""
        return self.pt_cell >= self.n_eu


GRID_FILE = "grid2.npz"                      # (grid.npz: the layout before the Azores)


async def grid() -> Grid:
    p = CACHE / GRID_FILE
    if not p.exists():
        await _build_grid(p)
    return Grid(np.load(p))


async def _coords(c, run, glob):
    blat, blon = await asyncio.gather(_get(c, _url(run, "clat", glob=glob)),
                                      _get(c, _url(run, "clon", glob=glob)))
    if not blat or not blon:
        raise FieldUnavailable(f"{SOURCE_G if glob else SOURCE}: coordenadas das células em falta")
    clat = (await asyncio.to_thread(_decode, blat))[0][1]
    clon = (await asyncio.to_thread(_decode, blon))[0][1]
    return clat, clon


async def _build_grid(p) -> None:
    run = await latest_run()
    grun = next((r for r, st in await runs_available(glob=True)), None)
    if run is None or grun is None:
        raise FieldUnavailable("ICON-EU-EPS / ICON-EPS: nenhuma run no servidor")
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT}) as c:
        (clat, clon), (glat, glon) = await asyncio.gather(_coords(c, run, False), _coords(c, grun, True))
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **await asyncio.to_thread(_layout, clat, clon, glat, glon))


def _inside(lat, lon, box):
    lon0, lon1, lat0, lat1 = box
    return np.nonzero((lat >= lat0) & (lat <= lat1) & (lon >= lon0) & (lon <= lon1))[0]


def _nearest(la, lo, lat, lon):
    return int(np.argmin(np.hypot(lat - la, (lon - lo) * np.cos(np.radians(la)))))


def _map_cells(glats, glons, lat, lon, box, max_deg):
    """Map nodes inside ``box`` -> nearest cell (within max_deg), -1 elsewhere."""
    LO, LA = np.meshgrid(glons, glats)
    out = np.full(LA.size, -1, np.int32)
    lon0, lon1, lat0, lat1 = box
    flat_la, flat_lo = LA.ravel(), LO.ravel()
    sel = np.nonzero((flat_la >= lat0) & (flat_la <= lat1) & (flat_lo >= lon0) & (flat_lo <= lon1))[0]
    coslat = np.cos(np.radians(lat))
    for s in range(0, len(sel), 2000):                   # nearest cell, by chunks
        ix = sel[s:s + 2000]
        la, lo = flat_la[ix, None], flat_lo[ix, None]
        d = (la - lat[None]) ** 2 + ((lo - lon[None]) * coslat[None]) ** 2
        k = d.argmin(1)
        ok = d[np.arange(len(k)), k] <= max_deg ** 2
        out[ix] = np.where(ok, k, -1)
    return out.reshape(LA.shape)


def _layout(clat, clon, glat, glon) -> dict:
    import shapely
    keep = _inside(clat, clon, KEEP)
    keep_g = _inside(glat, glon, KEEP_G)
    lat_e, lon_e = clat[keep], clon[keep]
    lat_g, lon_g = glat[keep_g], glon[keep_g]
    n_eu = len(keep)
    pt_cell, pt_code = [], []
    for a in areas():
        g = district_geometry(a.code)
        if g is not None:
            idx = [int(k) for k in np.nonzero(shapely.contains_xy(g, lon_e, lat_e))[0]]
        else:                                    # islands: the cell nearest each town
            lat_s, lon_s, off = (lat_g, lon_g, n_eu) if a.group == "acores" else (lat_e, lon_e, 0)
            idx = []
            for _, la, lo in a.towns:
                k = off + _nearest(la, lo, lat_s, lon_s)
                if k not in idx:
                    idx.append(k)
        pt_cell += idx
        pt_code += [a.code] * len(idx)
    lon0, lon1, lat0, lat1 = MAP
    glats = np.arange(lat0, lat1 + 1e-6, GRID_STEP)
    glons = np.arange(lon0, lon1 + 1e-6, GRID_STEP)
    ge = _map_cells(glats, glons, lat_e, lon_e, KEEP, 0.12)
    gg = _map_cells(glats, glons, lat_g, lon_g, KEEP_G, 0.25)
    gcell = np.where(ge >= 0, ge, np.where(gg >= 0, gg + n_eu, -1)).astype(np.int32)
    return {"n_total": np.int64(clat.size), "keep": keep.astype(np.int32),
            "n_total_g": np.int64(glat.size), "keep_g": keep_g.astype(np.int32),
            "lat": np.concatenate([lat_e, lat_g]), "lon": np.concatenate([lon_e, lon_g]),
            "pt_cell": np.array(pt_cell, np.int32), "pt_code": np.array(pt_code),
            "glats": glats, "glons": glons, "gcell": gcell}


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
        p = CACHE / GRID_FILE
        if not p.exists():
            raise FieldUnavailable("ICON-EU-EPS: grelha ainda não descarregada")
        _POINTS = RainPoints(Grid(np.load(p)))
    return _POINTS


def thresholds() -> np.ndarray:
    n = len(rain_points())
    return np.stack([np.full(n, C.RAIN1["laranja"]), np.full(n, C.RAIN1["vermelho"])]).astype(np.float32)


# ---------------------------------------------------------------- member fields
def _cache(run: dt.datetime, step: int, glob: bool = False):
    return CACHE / run.strftime("%Y%m%d%H") / f"{'tpg' if glob else 'tp'}_{step:03d}.npz"


async def _step(c: httpx.AsyncClient, run: dt.datetime, step: int, g: Grid,
                glob: bool = False) -> np.ndarray | None:
    """(40, kept cells of one ensemble) precipitation accumulated from the start (mm)."""
    keep, n_total = (g.keep_g, g.n_total_g) if glob else (g.keep, g.n_total)
    if step == 0:
        return np.zeros((N_MEMBERS, len(keep)), np.float32)
    p = _cache(run, step, glob)
    if p.exists():
        return np.load(p)["v"].astype(np.float32)
    blob = await _get(c, _url(run, "tot_prec", step, glob=glob))
    if blob is None:
        return None
    msgs = await asyncio.to_thread(_decode, blob)
    msgs.sort(key=lambda x: x[0])
    if len(msgs) != N_MEMBERS or msgs[0][1].size != n_total:
        raise FieldUnavailable(f"{SOURCE_G if glob else SOURCE} +{step} h: {len(msgs)} membros / "
                               "grelha diferente")
    v = np.stack([m[1][keep] for m in msgs]).astype(np.float32)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".part.npz")
    np.savez_compressed(tmp, v=v)
    tmp.replace(p)
    return v


async def _steps(run: dt.datetime, eu_steps, g: Grid, g_steps=()) -> dict[int, np.ndarray]:
    """{step: (40, kept cells)}: the ICON-EU-EPS cells read for ``eu_steps``, the global cells
    for ``g_steps``; NaN where a part was not read."""
    eu_steps, g_steps = sorted(set(eu_steps)), sorted(set(g_steps))
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT}) as c:
        ve, vg = await asyncio.gather(asyncio.gather(*(_step(c, run, s, g) for s in eu_steps)),
                                      asyncio.gather(*(_step(c, run, s, g, True) for s in g_steps)))
    out: dict[int, np.ndarray] = {}
    for part, steps, vals in ((slice(0, g.n_eu), eu_steps, ve), (slice(g.n_eu, None), g_steps, vg)):
        for s, v in zip(steps, vals):
            if v is not None:
                out.setdefault(s, np.full((N_MEMBERS, g.n_cells), np.nan, np.float32))[:, part] = v
    return out


def day_hours(run: dt.datetime, day: dt.date) -> tuple[int, int] | None:
    """Steps (a, b] of the run inside the UTC day, if they are all hourly; None otherwise."""
    t0 = dt.datetime(day.year, day.month, day.day)
    a, b = max(0, int((t0 - run) / H)), int((t0 + 24 * H - run) / H)
    if b <= a or b > HOURLY_TO:
        return None
    return a, b


async def _azores_wanted(ens_run: dt.datetime, days: list[dt.date]) -> bool:
    """Whether the ECMWF ENS gives ≥ AZ_SCREEN % of 25 mm in 24 h at an Azores point on one
    of the days (the global ICON-EPS files are read only then)."""
    from . import ens
    from .areas import land_points
    from .detect import sampler_for
    ep = await ens.ep_fields(ens_run, ["p_tp25"])
    sm = sampler_for(ep["lats"], ep["lons"])
    iaz = np.array([i for i, gr in enumerate(land_points().groups) if gr == "acores"])
    for (a, b), arr in zip(ep["p_tp25"]["windows"], ep["p_tp25"]["values"]):
        wa, wb = ens_run + a * H, ens_run + b * H
        for d in days:
            t0 = dt.datetime(d.year, d.month, d.day)
            if wa < t0 + 24 * H and wb > t0 and float(np.nanmax(sm.points(arr)[iaz])) >= AZ_SCREEN:
                return True
    return False


async def ens_daily(ens_run: dt.datetime, out: dict, log) -> dict:
    """Per day: the largest 1-h precipitation of every member at every point ("pts"), the
    hourly series (NaN in the hours not fetched), the map grid. ``ens_run`` is the ECMWF
    run being analysed: the days start on its first day, steps are given relative to it."""
    from .detect import first_day
    run, has_g = await pick_run()
    if run is None:
        raise FieldUnavailable("ICON-EU-EPS: nenhuma run completa no servidor")
    g = await grid()
    out.setdefault("src_runs", {})["chuva1h"] = run.isoformat()
    d0 = first_day(ens_run)
    days = [d for d in (d0 + dt.timedelta(days=i) for i in range(3)) if day_hours(run, d)]
    use_g = has_g and await _azores_wanted(ens_run, days)
    pc, pg = g.pt_cell, g.pt_glob()
    thr = C.RAIN1["laranja"]

    def mx(arr, glob):                          # largest value over one ensemble's points
        v = arr[:, pc[pg if glob else ~pg]]
        return float(np.nanmax(v)) if np.isfinite(v).any() else float("nan")

    # 1) day totals and 2) 6-h totals: which windows can hold an hour ≥ the orange threshold
    bounds = {d: day_hours(run, d) for d in days}
    ends = sorted({x for a, b in bounds.values() for x in (a, b)})
    tot = await _steps(run, ends, g, ends if use_g else ())
    screen, win = {}, {False: {}, True: {}}
    for d, (a, b) in bounds.items():
        if a not in tot or b not in tot:
            continue
        acc = tot[b] - tot[a]
        m_eu, m_g = mx(acc, False), mx(acc, True)
        screen[d.isoformat()] = round(float(np.nanmax([m_eu, m_g])), 1)
        edges = list(range(a, b, WINDOW)) + [b]
        for glob, m in ((False, m_eu), (True, m_g)):
            if m >= thr:
                win[glob][d] = list(zip(edges[:-1], edges[1:]))
    out["screen"]["chuva1h"] = screen
    az = " (Açores: ICON-EPS global)" if use_g else (" (Açores: sem chuva forte no ECMWF)" if has_g
                                                    else " (Açores: run global em falta)")
    if not win[False] and not win[True]:
        log(f"  chuva em 1 h: ICON-EU-EPS {run:%d %Hh}{az}, nenhum membro com ≥ {thr + 0.5:.0f} mm "
            f"num dia ({', '.join(f'{k[8:]}: {v}' for k, v in screen.items())} mm)")
        return {}
    w6 = await _steps(run, [x for ws in win[False].values() for w in ws for x in w], g,
                      [x for ws in win[True].values() for w in ws for x in w])
    hot = {False: {}, True: {}}
    for glob in (False, True):
        for d, ws in win[glob].items():
            hot[glob][d] = [(a, b) for a, b in ws if a in w6 and b in w6
                            and mx(w6[b] - w6[a], glob) >= thr]
    need = {glob: sorted({s for ws in hot[glob].values() for a, b in ws for s in range(a, b + 1)})
            for glob in (False, True)}
    n_win = sum(len(w) for h in hot.values() for w in h.values())
    log(f"  chuva em 1 h: ICON-EU-EPS {run:%d %Hh}{az}, {n_win} janela(s) de 6 h com "
        f"≥ {thr + 0.5:.0f} mm, {len(set(need[False]) | set(need[True]))} passos × 40 membros")
    hr = await _steps(run, need[False], g, need[True])
    shift = int((run - ens_run) / H)
    res = {}
    for d in days:
        ws = sorted(set(hot[False].get(d, [])) | set(hot[True].get(d, [])))
        if not ws:
            continue
        a, b = bounds[d]
        series = np.full((b - a, N_MEMBERS, len(pc)), np.nan, np.float32)
        gmax = np.full((N_MEMBERS, g.n_cells), np.nan, np.float32)
        for wa, wb in ws:
            for s in range(wa + 1, wb + 1):
                if s in hr and s - 1 in hr:
                    one = np.maximum(hr[s] - hr[s - 1], 0.0)
                    series[s - a - 1] = np.fmax(series[s - a - 1], one[:, pc])
                    gmax = np.fmax(gmax, one)
        if not np.isfinite(series).any():
            continue
        pts = np.fmax.reduce(series, axis=0)             # NaN: no hour read (other ensemble)
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
