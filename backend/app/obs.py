"""Observed radiosonde soundings (for case studies).

Sources, in order: the permanent event archive, University of Wyoming (high-resolution
BUFR, usually within hours) and NOAA IGRA v2 "year-to-date" files (reference archive,
~1–2 days behind). Every sounding obtained is stored in the archive, so a case study keeps
its observations. Profiles use the same layout as model profiles (``p z t td u v``,
pressure descending, wind in knots) so they can be overlaid and analysed alike.
"""
from __future__ import annotations

import asyncio
import csv
import datetime as dt
import io
import json
import math
import re
import time
import zipfile

import httpx
import numpy as np

from .config import CACHE_DIR, USER_AGENT
from .archive import ARCHIVE_DIR

OBS_DIR = ARCHIVE_DIR / "obs"
IGRA_CACHE = CACHE_DIR / "igra"
for _d in (OBS_DIR, IGRA_CACHE):
    _d.mkdir(parents=True, exist_ok=True)

WYOMING = "https://weather.uwyo.edu/wsgi/sounding"
IGRA_Y2D = "https://www.ncei.noaa.gov/pub/data/igra/data/data-y2d"
IGRA_POR = "https://www.ncei.noaa.gov/pub/data/igra/data/data-por"
# older WMO numbers of the same launch site, tried at the University of Wyoming
ALIASES = {"08536": ["08579"]}
KT = 1.943844

# WMO id, IGRA id, name, lat, lon
STATIONS = [
    ("08536", "POM00008536", "Lisboa (Gago Coutinho)", 38.77, -9.13),
    ("08522", "POM00008522", "Funchal (Madeira)", 32.69, -16.77),
    ("08508", "POM00008508", "Lajes (Terceira)", 38.73, -27.07),
    ("08001", "SPM00008001", "A Corunha", 43.37, -8.42),
    ("08023", "SPM00008023", "Santander", 43.49, -3.80),
    ("08221", "SPM00008221", "Madrid (Barajas)", 40.47, -3.58),
    ("08160", "SPM00008160", "Saragoça", 41.66, -1.01),
    ("08430", "SPM00008430", "Múrcia", 38.00, -1.17),
    ("08302", "SPM00008302", "Palma de Maiorca", 39.55, 2.62),
    ("08190", "SPM00008190", "Barcelona", 41.39, 2.12),
    ("60018", "SPM00060018", "Güímar (Tenerife)", 28.32, -16.38),
    ("07510", "FRM00007510", "Bordéus", 44.83, -0.69),
    ("07110", "FRM00007110", "Brest", 48.45, -4.42),
    ("03953", "EIM00003953", "Valentia (Irlanda)", 51.94, -10.25),
    ("03808", "UKM00003808", "Camborne (Cornualha)", 50.22, -5.33),
]
STATION = {s[0]: s for s in STATIONS}


def station_info(sid: str) -> dict:
    s = STATION[sid]
    return {"id": s[0], "name": s[2], "lat": s[3], "lon": s[4]}


def nearest_station(lat: float, lon: float, max_km: float = 150.0):
    best, bd = None, 1e9
    for s in STATIONS:
        d = 111.2 * math.hypot(lat - s[3], (lon - s[4]) * math.cos(math.radians(lat)))
        if d < bd:
            best, bd = s, d
    return (best[0], bd) if best and bd <= max_km else (None, bd)


def _path(sid: str, t: dt.datetime):
    return OBS_DIR / sid / f"{t:%Y%m%d%H}.json"


def _absent_path(sid: str, t: dt.datetime):
    return OBS_DIR / sid / f"{t:%Y%m%d%H}.absent"


def _profile(p, z, t, td, wdir, wspd_ms, src: str, sid: str, when) -> dict | None:
    """Clean, sort (surface first), thin to >= 2 hPa steps; wind to u/v knots."""
    arr = np.array([p, z, t, td, wdir, wspd_ms], dtype=float).T
    arr = arr[np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 2])]
    if len(arr) < 5:
        return None
    arr = arr[np.argsort(-arr[:, 0])]
    keep, last = [], None
    for row in arr:
        if last is None or last - row[0] >= 2.0:
            keep.append(row)
            last = row[0]
    a = np.array(keep)
    p, z, t, td, wd, ws = a.T
    td = np.where(np.isfinite(td), np.minimum(td, t), np.nan)
    # fill missing heights hypsometrically (needed for heights / kinematics)
    if not np.isfinite(z).all():
        zf = z.copy()
        for i in range(len(p)):
            if not np.isfinite(zf[i]) and i > 0 and np.isfinite(zf[i - 1]):
                tm = 273.15 + np.nanmean([t[i - 1], t[i]])
                zf[i] = zf[i - 1] + 287.05 * tm / 9.80665 * math.log(p[i - 1] / p[i])
        z = zf
    rad = np.radians(wd)
    u = -ws * np.sin(rad) * KT
    v = -ws * np.cos(rad) * KT
    s = STATION.get(sid)
    return {"station": sid, "name": s[2] if s else sid, "lat": s[3] if s else None,
            "lon": s[4] if s else None, "time": when.isoformat(), "source": src,
            "p": p.tolist(), "z": z.tolist(), "t": t.tolist(), "td": td.tolist(),
            "u": u.tolist(), "v": v.tolist()}


def _num(x):
    try:
        f = float(str(x).strip())
        return f if math.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


async def _wyoming(c: httpx.AsyncClient, sid: str, when: dt.datetime) -> dict | None:
    """High-resolution BUFR soundings first, then older (TEMP) soundings (src=UNKNOWN); also
    the station's older WMO numbers."""
    for wid in [sid] + ALIASES.get(sid, []):
        for src in ("BUFR", "UNKNOWN"):
            prof = await _wyoming_one(c, sid, wid, src, when)
            if prof:
                return prof
    return None


async def _wyoming_one(c: httpx.AsyncClient, sid: str, wid: str, src: str, when: dt.datetime) -> dict | None:
    try:
        r = await c.get(WYOMING, params={"src": src, "datetime": when.strftime("%Y-%m-%d %H:00:00"),
                                         "id": wid, "type": "TEXT:CSV"})
    except httpx.HTTPError:
        return None
    if r.status_code != 200:
        return None
    lines = r.text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith("time,") and "pressure_hPa" in ln), None)
    if start is None:
        return None
    block = [lines[start]]
    for ln in lines[start + 1:]:
        if not ln.strip() or ln.strip().startswith("<"):
            break
        block.append(ln)
    rows = list(csv.DictReader(io.StringIO("\n".join(block))))
    if not rows:
        return None
    g = lambda k: [_num(r.get(k)) for r in rows]  # noqa: E731
    return _profile(g("pressure_hPa"), g("geopotential height_m"), g("temperature_C"),
                    g("dew point temperature_C"), g("wind direction_degree"), g("wind speed_m/s"),
                    "Univ. Wyoming (BUFR)" if src == "BUFR" else f"Univ. Wyoming ({wid})", sid, when)


_igra_lock = asyncio.Lock()


async def _igra_zip(c: httpx.AsyncClient, igra_id: str, year: int):
    """Cached IGRA zip with the station's soundings: the year-to-date file for the current
    year, the whole period of record (~30 MB) for older years."""
    async with _igra_lock:
        if year >= dt.datetime.utcnow().year:
            idx = IGRA_CACHE / "_index.html"
            if not idx.exists() or time.time() - idx.stat().st_mtime > 6 * 3600:
                try:
                    r = await c.get(IGRA_Y2D + "/")
                    r.raise_for_status()
                    idx.write_text(r.text)
                except httpx.HTTPError:
                    if not idx.exists():
                        return None
            m = re.search(rf"{igra_id}-data-beg\d+\.txt\.zip", idx.read_text())
            if not m:
                return None
            name, url, max_age = m.group(0), f"{IGRA_Y2D}/{m.group(0)}", 6 * 3600
        else:
            name, url, max_age = f"{igra_id}-data.txt.zip", f"{IGRA_POR}/{igra_id}-data.txt.zip", 30 * 86400
        z = IGRA_CACHE / name
        if not z.exists() or time.time() - z.stat().st_mtime > max_age:
            try:
                r = await c.get(url)
                r.raise_for_status()
                z.write_bytes(r.content)
            except httpx.HTTPError:
                if not z.exists():
                    return None
        return z


def _igra_find(z, sid: str, when: dt.datetime) -> dict | None:
    """Stream the IGRA zip until the sounding of ``when``."""
    key = f"{when.year:4d} {when.month:02d} {when.day:02d} {when.hour:02d}"
    try:
        with zipfile.ZipFile(z) as zf, zf.open(zf.namelist()[0]) as fh:
            lines = io.TextIOWrapper(fh, encoding="utf-8", errors="ignore")
            for ln in lines:
                if ln.startswith("#") and ln[13:26] == key:
                    n = int(ln[32:36].strip() or 0)
                    return _igra_parse_body([next(lines) for _ in range(n)], sid, when)
    except (zipfile.BadZipFile, OSError):
        z.unlink(missing_ok=True)
    return None


def _igra_parse_body(body: list[str], sid: str, when: dt.datetime) -> dict | None:
    def num(s, sc=1.0):
        s = s.strip()
        try:
            v = int(s)
        except ValueError:
            return np.nan
        return np.nan if v in (-9999, -8888) else v * sc

    p, z, t, dpd, rh, wd, ws = ([] for _ in range(7))
    for b in body:
        b = b.ljust(51)
        p.append(num(b[9:15], 0.01))
        z.append(num(b[16:21]))
        t.append(num(b[22:27], 0.1))
        rh.append(num(b[28:33], 0.1))
        dpd.append(num(b[34:39], 0.1))
        wd.append(num(b[40:45]))
        ws.append(num(b[46:51], 0.1))
    t, dpd, rh = np.array(t), np.array(dpd), np.array(rh)
    td = t - dpd
    need = ~np.isfinite(td) & np.isfinite(t) & np.isfinite(rh)
    if need.any():
        tt, hh = t[need], np.clip(rh[need], 1, 100)
        g = np.log(hh / 100) + 17.625 * tt / (243.04 + tt)
        td[need] = 243.04 * g / (17.625 - g)
    return _profile(p, z, t, td, wd, ws, "NOAA IGRA v2", sid, when)


async def get_obs(sid: str, when: dt.datetime, retry_absent_h: float = 12) -> dict | None:
    """Observed sounding (archive → Wyoming → IGRA). Stored in the archive when found;
    an absent result is remembered for ``retry_absent_h`` hours."""
    p = _path(sid, when)
    if p.exists():
        return json.loads(p.read_text())
    ab = _absent_path(sid, when)
    if ab.exists() and time.time() - ab.stat().st_mtime < retry_absent_h * 3600:
        return None
    if when > dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) - dt.timedelta(hours=1):
        return None
    async with httpx.AsyncClient(timeout=60, headers={"User-Agent": USER_AGENT},
                                 follow_redirects=True) as c:
        prof = await _wyoming(c, sid, when)
        if prof is None and sid in STATION:
            z = await _igra_zip(c, STATION[sid][1], when.year)
            if z:
                prof = await asyncio.to_thread(_igra_find, z, sid, when)
    p.parent.mkdir(parents=True, exist_ok=True)
    if prof is None:
        ab.write_text("")
        return None
    p.write_text(json.dumps(prof))
    ab.unlink(missing_ok=True)
    return prof


def obs_as_sounding(prof: dict) -> dict:
    """Wrap an observed profile in the structure produced by ``get_profile``."""
    pr = {k: np.array(prof[k], dtype=float) for k in ("p", "z", "t", "td", "u", "v")}
    pr["w"] = np.full(len(pr["p"]), np.nan)
    sfc = {"p": float(pr["p"][0]), "z": float(pr["z"][0]), "t": float(pr["t"][0]),
           "td": float(pr["td"][0]), "u": float(pr["u"][0]), "v": float(pr["v"][0])}
    return {"lat": prof["lat"], "lon": prof["lon"], "profile": pr, "surface": sfc, "model_diag": {}}


def event_obs_times(ev: dict) -> list[dt.datetime]:
    from .archive import parse_t
    t0, t1 = parse_t(ev["start"]), parse_t(ev["end"])
    t = t0.replace(hour=0, minute=0, second=0, microsecond=0)
    out = []
    while t <= t1:
        if t >= t0:
            out.append(t)
        t += dt.timedelta(hours=12)
    return out


async def capture_event_obs(ev: dict, log=print) -> int:
    """Fetch (and archive) every station's 00/12Z soundings inside the event window."""
    from .regions import REGIONS
    d = REGIONS[ev.get("region", "nea")].download_domain
    sids = [s[0] for s in STATIONS if d.lon0 <= s[4] <= d.lon1 and d.lat0 <= s[3] <= d.lat1]
    n = 0
    for t in event_obs_times(ev):
        for sid in sids:
            if _path(sid, t).exists():
                n += 1
                continue
            try:
                if await get_obs(sid, t):
                    n += 1
            except Exception as e:  # noqa: BLE001
                log(f"[obs] {sid} {t:%Y%m%d%H}: {e}")
    return n


def archived_obs(ev: dict) -> dict:
    """{station id: [iso times]} of archived observed soundings in the event window."""
    out = {}
    times = set(event_obs_times(ev))
    for s in STATIONS:
        d = OBS_DIR / s[0]
        if not d.exists():
            continue
        ts = sorted(t.isoformat() for t in times if _path(s[0], t).exists())
        if ts:
            out[s[0]] = ts
    return out
