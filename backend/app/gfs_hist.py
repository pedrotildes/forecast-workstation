"""Historical GFS analyses (0.5°) from the NOAA/NCEI archive, for historical events.

NCEI keeps the GFS analysis files (``gfsanl_4``, every 6 h) for past years, but not the
forecasts. Each file (~60 MB) is downloaded once, the needed fields are cropped to the event
box and stored in the event archive as model ``gfs`` (step 0 of each run).
"""
from __future__ import annotations

import asyncio
import datetime as dt
import tempfile
from pathlib import Path

import eccodes
import httpx

from .config import USER_AGENT
from .grib import decode_message, split_messages

H = dt.timedelta(hours=1)
URLS = ["https://www.ncei.noaa.gov/thredds/fileServer/model-gfs-g4-anl-files-old/{t:%Y%m}/{t:%Y%m%d}/"
        "gfsanl_4_{t:%Y%m%d}_{t:%H}00_000.grb2",
        "https://www.ncei.noaa.gov/thredds/fileServer/model-gfs-g4-anl-files/{t:%Y%m}/{t:%Y%m%d}/"
        "gfsanl_4_{t:%Y%m%d}_{t:%H}00_000.grb2"]
LEVELS = {1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250,
          200, 150, 100}
UPPER = {"gh": "gh", "t": "t", "r": "r", "u": "u", "v": "v", "w": "w"}
SINGLE = {("prmsl", "meanSea"): "msl", ("sp", "surface"): "sp", ("orog", "surface"): "orog",
          ("2t", "heightAboveGround"): "t2m", ("2d", "heightAboveGround"): "d2m",
          ("10u", "heightAboveGround"): "u10", ("10v", "heightAboveGround"): "v10",
          ("gust", "surface"): "gust", ("pwat", "atmosphereSingleLayer"): "pwat",
          ("pwat", "atmosphere"): "pwat", ("cape", "surface"): "cape"}


def _key(msg: bytes):
    h = eccodes.codes_new_from_message(msg)
    try:
        sn = eccodes.codes_get(h, "shortName")
        tl = eccodes.codes_get(h, "typeOfLevel")
        lv = int(eccodes.codes_get(h, "level"))
    finally:
        eccodes.codes_release(h)
    if tl == "isobaricInhPa" and sn in UPPER and lv in LEVELS:
        return (UPPER[sn], lv)
    name = SINGLE.get((sn, tl))
    if name and (tl != "heightAboveGround" or lv in (2, 10)):
        return (name, 0)
    return None


def _store(path: Path, run: dt.datetime, bbox) -> int:
    from . import archive
    data = path.read_bytes()
    n = 0
    seen = set()
    for _, msg in split_messages(data):
        try:
            k = _key(msg)
        except Exception:  # noqa: BLE001
            continue
        if k is None or k in seen:
            continue
        seen.add(k)
        _, f = decode_message(msg, bbox)
        if f.values.size:
            archive.save_field("gfs", run, 0, k, f)
            n += 1
    return n


async def import_event(ev: dict, box: tuple[float, float, float, float], log=print) -> int:
    """Archive the 6-hourly GFS analyses of the event window (0.5°) over ``box``."""
    from . import archive
    t0, t1 = archive.parse_t(ev["start"]), archive.parse_t(ev["end"])
    run = t0.replace(hour=(t0.hour // 6) * 6, minute=0, second=0, microsecond=0)
    bbox = (box[0] - 0.5, box[1] + 0.5, box[2] - 0.5, box[3] + 0.5)
    total = 0
    async with httpx.AsyncClient(timeout=900, headers={"User-Agent": USER_AGENT}, follow_redirects=True) as c:
        while run <= t1:
            if archive._path("gfs", run, 0, ("msl", 0)).exists():
                run += 6 * H
                continue
            ok = False
            for u in URLS:
                url = u.format(t=run)
                with tempfile.NamedTemporaryFile(suffix=".grb2", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                try:
                    async with c.stream("GET", url) as r:
                        if r.status_code != 200:
                            continue
                        with open(tmp_path, "wb") as fh:
                            async for chunk in r.aiter_bytes(1 << 20):
                                fh.write(chunk)
                    n = await asyncio.to_thread(_store, tmp_path, run, bbox)
                    total += n
                    ok = True
                    log(f"[gfs-arquivo] {run:%Y-%m-%d %H} UTC: {n} campos")
                    break
                except httpx.HTTPError as e:
                    log(f"[gfs-arquivo] {run:%Y%m%d%H}: {e}")
                finally:
                    tmp_path.unlink(missing_ok=True)
            if not ok:
                log(f"[gfs-arquivo] {run:%Y-%m-%d %H} UTC indisponível")
            run += 6 * H
    return total
