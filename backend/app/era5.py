"""ERA5 reanalysis (ECMWF / Copernicus C3S) for historical events.

Read from the NCAR RDA copy of ERA5 (ds633.0) through OPeNDAP, which lets us ask only for
the event box, the pressure levels and the times we need — no account is required. The
fields go into the permanent event archive as model ``era5``, laid out like the forecast
runs of the other models so that every case-study product works unchanged:

* run r = every 6 h; upper-air fields (27 levels, 1000–100 hPa) at steps 0 and 3
  (3-hourly analyses);
* surface fields hourly at steps 0–5 of each run;
* ``tp`` at steps 1–6 = precipitation accumulated since r (large-scale + convective, mm);
* ``gust`` at steps 1–6 = maximum 10 m gust in the hour ending at r + step.
"""
from __future__ import annotations

import asyncio
import calendar
import datetime as dt
import re

import httpx
import numpy as np

from .config import USER_AGENT
from .grib import Field

DAP = "https://thredds.rda.ucar.edu/thredds/dodsC/files/g/d633000"
ALL_LEVELS = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450,
              500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
L0 = ALL_LEVELS.index(100)
LEVELS = ALL_LEVELS[L0:]                       # 27 levels, 100 → 1000 hPa
G = 9.80665
H = dt.timedelta(hours=1)

# canonical name -> (file code, DAP variable)
PL = {"gh": ("128_129_z.ll025sc", "Z"), "t": ("128_130_t.ll025sc", "T"), "u": ("128_131_u.ll025uv", "U"),
      "v": ("128_132_v.ll025uv", "V"), "r": ("128_157_r.ll025sc", "R"), "w": ("128_135_w.ll025sc", "W")}
SFC = {"msl": ("128_151_msl", "MSL"), "sp": ("128_134_sp", "SP"), "u10": ("128_165_10u", "VAR_10U"),
       "v10": ("128_166_10v", "VAR_10V"), "t2m": ("128_167_2t", "VAR_2T"), "d2m": ("128_168_2d", "VAR_2D"),
       "tcc": ("128_164_tcc", "TCC"), "pwat": ("128_137_tcwv", "TCWV"), "mucape": ("128_059_cape", "CAPE")}
FC = {"lsp": ("accumu", "128_142_lsp", "LSP"), "cp": ("accumu", "128_143_cp", "CP"),
      "gust": ("minmax", "128_049_10fg", "VAR_10FG")}
UNITS = {"gh": "gpm", "t": "K", "u": "m s**-1", "v": "m s**-1", "r": "%", "w": "Pa s**-1", "msl": "Pa",
         "sp": "Pa", "u10": "m s**-1", "v10": "m s**-1", "t2m": "K", "d2m": "K", "tcc": "%", "pwat": "kg m**-2",
         "mucape": "J kg**-1", "tp": "kg m**-2", "gust": "m s**-1"}


# ------------------------------------------------------------------ OPeNDAP
def _parse_dods(b: bytes) -> np.ndarray:
    i = b.find(b"\nData:\n")
    if i < 0:
        raise RuntimeError("resposta OPeNDAP inválida: " + b[:200].decode("utf-8", "replace"))
    dds = b[:i].decode()
    m = re.search(r"Float32 \w+((?:\[\w+ = \d+\])+);", dds)
    if not m:
        raise RuntimeError("variável Float32 não encontrada na resposta OPeNDAP")
    shape = [int(x) for x in re.findall(r"= (\d+)\]", m.group(1))]
    d = b[i + 7:]
    n = int.from_bytes(d[:4], "big")
    a = np.frombuffer(d[8:8 + 4 * n], dtype=">f4").astype(np.float32).reshape(shape)
    a[np.abs(a) > 1e19] = np.nan
    return a


def _lon_pieces(lon0: float, lon1: float):
    """Longitude index ranges (ERA5 grid 0…359.75) and their longitudes (−180…180)."""
    i0, i1 = int(np.floor((lon0 % 360) / 0.25)), int(np.ceil((lon1 % 360) / 0.25))
    if lon0 < 0 <= lon1:
        return [(i0, 1439), (0, i1)]
    return [(i0, i1)]


async def _get(c: httpx.AsyncClient, sem, path: str, var: str, idx: str) -> np.ndarray:
    url = f"{DAP}/{path}.dods?{var}.{var}{idx}"
    async with sem:
        for attempt in range(4):
            try:
                r = await c.get(url.replace("[", "%5B").replace("]", "%5D"))
                if r.status_code == 200:
                    return _parse_dods(r.content)
                err = f"HTTP {r.status_code}"
            except httpx.HTTPError as e:
                err = str(e)
            await asyncio.sleep(5 * (attempt + 1))
    raise RuntimeError(f"ERA5 {path}: {err}")


async def _box(c, sem, path, var, lead: str, j0, j1, pieces) -> np.ndarray:
    """Request [lead][j0:j1][i…] for each longitude piece and join them along longitude."""
    parts = await asyncio.gather(*(_get(c, sem, path, var, f"{lead}[{j0}:{j1}][{a}:{b}]") for a, b in pieces))
    return np.concatenate(parts, axis=-1)


def _axes(j0, j1, pieces):
    lats = 90 - 0.25 * np.arange(j0, j1 + 1)
    lons = np.concatenate([np.arange(a, b + 1) * 0.25 for a, b in pieces])
    lons = np.where(lons >= 180, lons - 360, lons)
    return lats, lons


def _field(a2d, lats, lons, units) -> Field:
    """Field with ascending latitudes (the archive convention)."""
    return Field(lats[::-1].astype(np.float64), lons.astype(np.float64),
                 np.ascontiguousarray(a2d[::-1]).astype(np.float32), units)


def _floor6(t: dt.datetime) -> dt.datetime:
    return t.replace(hour=(t.hour // 6) * 6, minute=0, second=0, microsecond=0)


def _fc_init(v: dt.datetime) -> tuple[dt.datetime, int]:
    """ERA5 short forecast (initialised 06/18 UTC, hours 1–12) whose hour ends at ``v``."""
    x = v - H
    if x.hour >= 18:
        init = x.replace(hour=18)
    elif x.hour >= 6:
        init = x.replace(hour=6)
    else:
        init = (x - 24 * H).replace(hour=18)
    init = init.replace(minute=0, second=0, microsecond=0)
    return init, int((v - init) / H)


def _fc_file(kind: str, code: str, init: dt.datetime) -> tuple[str, int]:
    """(path, index of ``init`` in the file) — forecast files run 01 06 UTC → 16 06 UTC → next 01 06."""
    first = init.replace(day=1, hour=6)
    if init < first:                                   # 1st of month 00–05 h: previous month's file
        prev = first - dt.timedelta(days=1)
        first = prev.replace(day=16, hour=6)
        nxt = (prev.replace(day=28) + dt.timedelta(days=4)).replace(day=1, hour=6)
    elif init >= first.replace(day=16):
        first = first.replace(day=16)
        nxt = (first.replace(day=28) + dt.timedelta(days=4)).replace(day=1, hour=6)
    else:
        nxt = first.replace(day=16)
    name = f"e5.oper.fc.sfc.{kind}.{code}.ll025sc.{first:%Y%m%d%H}_{nxt:%Y%m%d%H}.nc"
    return f"e5.oper.fc.sfc.{kind}/{first:%Y%m}/{name}", int((init - first) / (12 * H))


# ------------------------------------------------------------------ import
async def import_event(ev: dict, box: tuple[float, float, float, float], log=print, concurrency=3) -> dict:
    """Archive ERA5 for the event window over ``box`` = (lon0, lon1, lat0, lat1)."""
    from . import archive
    t0 = _floor6(archive.parse_t(ev["start"]))
    t1 = archive.parse_t(ev["end"])
    lon0, lon1, lat0, lat1 = box
    j0, j1 = int(np.floor((90 - lat1) / 0.25)), int(np.ceil((90 - lat0) / 0.25))
    pieces = _lon_pieces(lon0, lon1)
    lats, lons = _axes(j0, j1, pieces)
    sem = asyncio.Semaphore(concurrency)
    saved = {"upper": 0, "surface": 0, "tp": 0, "gust": 0}
    async with httpx.AsyncClient(timeout=600, headers={"User-Agent": USER_AGENT}) as c:
        # --- pressure levels, 3-hourly (one request per variable and day)
        day = t0.replace(hour=0)
        while day <= t1:
            hours = [h for h in range(0, 24, 3) if t0 <= day + h * H <= t1]
            if hours:
                for name, (code, var) in PL.items():
                    path = (f"e5.oper.an.pl/{day:%Y%m}/e5.oper.an.pl.{code}."
                            f"{day:%Y%m%d}00_{day:%Y%m%d}23.nc")
                    if all(archive._path("era5", _floor6(day + h * H), h % 6, (name, 1000)).exists()
                           for h in hours):
                        continue
                    a = await _box(c, sem, path, var, f"[{hours[0]}:3:{hours[-1]}][{L0}:36]", j0, j1, pieces)
                    for k, h in enumerate(hours):
                        t = day + h * H
                        for li, p in enumerate(LEVELS):
                            v = a[k, li] / G if name == "gh" else a[k, li]
                            archive.save_field("era5", _floor6(t), h % 6, (name, p),
                                               _field(v, lats, lons, UNITS[name]))
                            saved["upper"] += 1
                log(f"[era5] níveis de pressão {day:%Y-%m-%d}: {len(hours)} análises")
            day += 24 * H
        # --- surface analyses, hourly (one request per variable and month)
        month = t0.replace(day=1, hour=0)
        while month <= t1:
            last = month.replace(day=calendar.monthrange(month.year, month.month)[1], hour=23)
            a_t, b_t = max(t0, month), min(t1 + 6 * H, last)
            ia, ib = int((a_t - month) / H), int((b_t - month) / H)
            hrs = [month + (ia + k) * H for k in range(ib - ia + 1)]
            for name, (code, var) in SFC.items():
                if all(archive._path("era5", _floor6(t), int((t - _floor6(t)) / H), (name, 0)).exists() for t in hrs):
                    continue
                path = (f"e5.oper.an.sfc/{month:%Y%m}/e5.oper.an.sfc.{code}.ll025sc."
                        f"{month:%Y%m%d}00_{last:%Y%m%d}23.nc")
                a = await _box(c, sem, path, var, f"[{ia}:{ib}]", j0, j1, pieces)
                for k in range(ib - ia + 1):
                    t = month + (ia + k) * H
                    r = _floor6(t)
                    v = a[k] * 100 if name == "tcc" else a[k]
                    archive.save_field("era5", r, int((t - r) / H), (name, 0), _field(v, lats, lons, UNITS[name]))
                    saved["surface"] += 1
            log(f"[era5] superfície {month:%Y-%m}: {ib - ia + 1} horas")
            month = last.replace(hour=0) + 24 * H
        # --- precipitation and gusts from the short forecasts (initialised 06 and 18 UTC)
        hourly = {}
        v_t = t0 + H
        while v_t <= t1 + 6 * H:
            hourly[v_t] = _fc_init(v_t)
            v_t += H
        runs = [t0 + 6 * k * H for k in range(int((t1 - t0) / (6 * H)) + 1)]
        if all(archive._path("era5", r, 6, ("tp", 0)).exists() for r in runs):
            hourly = {}
        need = sorted({(k, init) for v_t, (init, fh) in hourly.items() for k in FC})
        got = {}
        for k, init in need:
            kind, code, var = FC[k]
            path, ii = _fc_file(kind, code, init)
            got[(k, init)] = (await _box(c, sem, path, var, f"[{ii}:{ii}][0:11]", j0, j1, pieces))[0]
        run = t0
        while run <= t1:
            acc = None
            for s in range(1, 7):
                v_t = run + s * H
                if v_t not in hourly:
                    break
                init, fh = hourly[v_t]
                pr = (got[("lsp", init)][fh - 1] + got[("cp", init)][fh - 1]) * 1000.0
                acc = pr if acc is None else acc + pr
                archive.save_field("era5", run, s, ("tp", 0), _field(np.clip(acc, 0, None), lats, lons, UNITS["tp"]))
                archive.save_field("era5", run, s, ("gust", 0),
                                   _field(got[("gust", init)][fh - 1], lats, lons, UNITS["gust"]))
                saved["tp"] += 1
                saved["gust"] += 1
            run += 6 * H
        log(f"[era5] precipitação e rajadas: {saved['tp']} passos")
    return saved
