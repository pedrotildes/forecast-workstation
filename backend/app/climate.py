"""Daily Tmax/Tmin climatological normals (NOAA CPC Global Unified, 1991–2020, 0.5°).

The two global long-term-mean files (~275 MB each) are downloaded once, cropped to
the Euro-Atlantic sector and stored as a compact ``.npz`` (float16). CPC is a
station-based analysis, so normals exist over land only.
"""
from __future__ import annotations

import asyncio
import datetime as dt
from functools import lru_cache

import httpx
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

from .config import CACHE_DIR, USER_AGENT

CLIM_DIR = CACHE_DIR / "clim"
CLIM_DIR.mkdir(parents=True, exist_ok=True)
NPZ = CLIM_DIR / "cpc_tmaxmin_ltm_1991-2020.npz"
SRC = "https://downloads.psl.noaa.gov/Datasets/cpc_global_temp/{v}.day.ltm.1991-2020.nc"
BOX = (-75.0, 50.0, 15.0, 78.0)   # lon0, lon1, lat0, lat1

STATE = {"status": "missing" if not NPZ.exists() else "ready", "done": 0, "total": 0,
         "error": None}
_lock = asyncio.Lock()


class ClimateNotReady(Exception):
    pass


def ready() -> bool:
    return NPZ.exists()


async def ensure() -> None:
    """Download and build the climatology if needed (idempotent, concurrency-safe)."""
    if NPZ.exists():
        return
    async with _lock:
        if NPZ.exists():
            return
        STATE.update(status="downloading", done=0, total=0, error=None)
        try:
            paths = {}
            async with httpx.AsyncClient(timeout=None, headers={"User-Agent": USER_AGENT},
                                         follow_redirects=True) as c:
                sizes = {}
                for v in ("tmax", "tmin"):
                    r = await c.head(SRC.format(v=v))
                    sizes[v] = int(r.headers.get("content-length", 0))
                STATE["total"] = sum(sizes.values())
                for v in ("tmax", "tmin"):
                    p = CLIM_DIR / f"{v}.nc"
                    if not (p.exists() and p.stat().st_size == sizes[v]):
                        async with c.stream("GET", SRC.format(v=v)) as r:
                            r.raise_for_status()
                            with open(p.with_suffix(".part"), "wb") as fh:
                                async for chunk in r.aiter_bytes(1 << 20):
                                    fh.write(chunk)
                                    STATE["done"] += len(chunk)
                        p.with_suffix(".part").replace(p)
                    else:
                        STATE["done"] += sizes[v]
                    paths[v] = p
            STATE["status"] = "building"
            await asyncio.to_thread(_build, paths)
            for p in paths.values():
                p.unlink(missing_ok=True)
            STATE["status"] = "ready"
        except Exception as e:  # noqa: BLE001
            STATE.update(status="error", error=str(e))
            raise


def _build(paths: dict) -> None:
    import h5py
    out = {}
    for v, p in paths.items():
        with h5py.File(p, "r") as f:
            lat = f["lat"][:].astype(float)
            lon = f["lon"][:].astype(float)
            ds = f[v]
            lon180 = ((lon + 180) % 360) - 180
            ic = np.nonzero((lon180 >= BOX[0]) & (lon180 <= BOX[1]))[0]
            jr = np.nonzero((lat >= BOX[2]) & (lat <= BOX[3]))[0]
            data = ds[:, jr.min():jr.max() + 1, :][:, :, ic].astype(np.float32)
            fill = ds.attrs.get("missing_value", ds.attrs.get("_FillValue", None))
            if fill is not None:
                data[np.isclose(data, np.float32(np.ravel(fill)[0]))] = np.nan
            data[(data < -90) | (data > 70)] = np.nan
            la = lat[jr.min():jr.max() + 1]
            lo = lon180[ic]
            o = np.argsort(lo)
            lo, data = lo[o], data[:, :, o]
            if la[0] > la[-1]:
                la, data = la[::-1], data[:, ::-1, :]
            out[v] = data.astype(np.float16)
            out["lat"], out["lon"] = la, lo
    np.savez_compressed(NPZ, **out)


@lru_cache(maxsize=1)
def _load():
    if not NPZ.exists():
        raise ClimateNotReady("climatologia ainda não descarregada")
    z = np.load(NPZ)
    return {k: z[k] for k in z.files}


def doy_index(d: dt.date) -> int:
    """0..364 (29 Feb mapped to 28 Feb)."""
    if d.month == 2 and d.day == 29:
        d = d.replace(day=28)
    return dt.date(2001, d.month, d.day).timetuple().tm_yday - 1


@lru_cache(maxsize=256)
def _field_filled(var: str, idx: int) -> np.ndarray:
    """Normal for one day; NaNs (sea) filled from the nearest land cell within 2 cells so
    coastal model points still get a value."""
    c = _load()
    a = c[var][idx].astype(np.float32)
    nan = np.isnan(a)
    if nan.any() and (~nan).any():
        dist, (ii, jj) = distance_transform_edt(nan, return_indices=True)
        filled = a[ii, jj]
        a = np.where(nan & (dist <= 2), filled, a)
    return a


def normal_on(var: str, day: dt.date, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Normal Tmax/Tmin (°C) for ``day`` interpolated to a lat/lon grid (NaN over sea)."""
    c = _load()
    a = _field_filled(var, doy_index(day))
    it = RegularGridInterpolator((c["lat"], c["lon"]), a, bounds_error=False, fill_value=np.nan)
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    return it(np.stack([LA, LO], axis=-1)).astype(np.float32)


def normal_at(var: str, day: dt.date, lat: float, lon: float) -> float:
    return float(normal_on(var, day, np.array([lat]), np.array([lon]))[0, 0])
