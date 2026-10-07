"""Common model-source machinery.

Every NWP source exposes the same *canonical* vocabulary of raw fields so that
derived products, maps and soundings never need to know which model they are
talking to.  A field is addressed by a :data:`Key` ``(name, level)``; ``level``
is the pressure in hPa for upper-air fields and ``0`` for single-level fields.

Canonical names and units (after normalisation by each model):

=========  ===========================================  ==========
name       meaning                                      units
=========  ===========================================  ==========
gh t r q   geopotential height, temperature, RH, q      gpm, K, %, kg/kg
u v w      wind components, vertical velocity (omega)   m/s, m/s, Pa/s
msl sp     mean sea-level / surface pressure            Pa
orog       model orography                              m
t2m d2m    2 m temperature / dew point                  K
u10 v10    10 m wind                                    m/s
tp         total precipitation accumulated from t=0     mm
cape       surface-based CAPE                           J/kg
mlcape     mixed-layer CAPE (lowest 90 hPa)             J/kg
mucape     most-unstable CAPE                           J/kg
cin        surface-based CIN                            J/kg
pwat       precipitable water                           mm
tcc lcc    total / low cloud cover (mcc, hcc too)       %
gust       10 m wind gust                               m/s
refc       composite reflectivity                       dBZ
frzl       height of the 0 °C isotherm                  m
=========  ===========================================  ==========
"""
from __future__ import annotations

import asyncio
import datetime as dt
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import GRIB_CACHE
from ..grib import Field, decode_message

Key = tuple[str, int]


@dataclass(frozen=True)
class Domain:
    lon0: float
    lon1: float
    lat0: float
    lat1: float

    def rounded(self, step: float = 1.0) -> "Domain":
        f = lambda x: np.floor(x / step) * step  # noqa: E731
        c = lambda x: np.ceil(x / step) * step  # noqa: E731
        return Domain(float(f(self.lon0)), float(c(self.lon1)),
                      float(max(-90, f(self.lat0))), float(min(90, c(self.lat1))))

    @property
    def tag(self) -> str:
        return f"{self.lon0:g}_{self.lon1:g}_{self.lat0:g}_{self.lat1:g}"

    @classmethod
    def around(cls, lat: float, lon: float, half: float = 1.5) -> "Domain":
        return cls(lon - half, lon + half, lat - half, lat + half).rounded()

    @classmethod
    def tile(cls, lat: float, lon: float, size: float = 4.0) -> "Domain":
        """Fixed point-extraction tile (+1° margin) so nearby points share downloads."""
        la = np.floor(lat / size) * size
        lo = np.floor(lon / size) * size
        return cls(float(max(-180, lo - 1)), float(min(179.75, lo + size + 1)),
                   float(max(-90, la - 1)), float(min(90, la + size + 1)))


def key_str(k: Key) -> str:
    return f"{k[0]}_{k[1]}" if k[1] else k[0]


class FieldUnavailable(Exception):
    pass


class _LRU:
    """LRU of decoded fields bounded by total array memory."""

    def __init__(self, max_bytes: int):
        self.max_bytes, self.bytes, self.d = max_bytes, 0, OrderedDict()

    def get(self, k):
        if k in self.d:
            self.d.move_to_end(k)
            return self.d[k]
        return None

    def put(self, k, v):
        if k in self.d:
            self.bytes -= self.d.pop(k).values.nbytes
        self.d[k] = v
        self.bytes += v.values.nbytes
        while self.bytes > self.max_bytes and len(self.d) > 1:
            _, old = self.d.popitem(last=False)
            self.bytes -= old.values.nbytes


_DECODED = _LRU(500 * 2**20)


class Model:
    """Base class. Subclasses implement run/step discovery and downloads."""

    id: str = ""
    name: str = ""
    resolution: str = "0.25°"
    description: str = ""
    levels: list[int] = []            # pressure levels available (hPa)
    native: set[str] = set()          # canonical names this model provides
    global_download = False           # True -> files are global, cropped locally

    # ---- to implement -------------------------------------------------
    async def list_runs(self) -> list[dt.datetime]:
        raise NotImplementedError

    async def list_steps(self, run: dt.datetime) -> list[int]:
        raise NotImplementedError

    async def _download(self, run: dt.datetime, step: int, keys: list[Key],
                        domain: Domain) -> dict[Key, bytes]:
        raise NotImplementedError

    def _postprocess(self, key: Key, f: Field) -> Field:
        return f

    # ---- shared -------------------------------------------------------
    def has(self, name: str, level: int = 0) -> bool:
        if name not in self.native:
            return False
        return not level or level in self.levels

    def info(self) -> dict:
        return {"id": self.id, "name": self.name, "resolution": self.resolution,
                "description": self.description, "levels": self.levels,
                "fields": sorted(self.native)}

    def _path(self, run: dt.datetime, step: int, domain: Domain, key: Key) -> Path:
        dtag = "global" if self.global_download else domain.rounded().tag
        return (GRIB_CACHE / self.id / run.strftime("%Y%m%d%H") / f"{step:03d}"
                / dtag / f"{key_str(key)}.grib2")

    _locks: dict = {}

    async def fetch(self, run: dt.datetime, step: int, keys: list[Key],
                    domain: Domain) -> dict[Key, Field]:
        keys = list(dict.fromkeys(keys))
        want_zero_tp = step == 0 and ("tp", 0) in keys
        if want_zero_tp:
            keys = [k for k in keys if k != ("tp", 0)]
            if not keys:
                keys = [("msl", 0)]
        dl_domain = domain.rounded() if not self.global_download else domain

        lock_key = (self.id, run, step, dl_domain.tag if not self.global_download else "g")
        lock = self._locks.setdefault(lock_key, asyncio.Lock())
        async with lock:
            missing = [k for k in keys if not self._path(run, step, dl_domain, k).exists()]
            if missing:
                got = await self._download(run, step, missing, dl_domain)
                for k, buf in got.items():
                    p = self._path(run, step, dl_domain, k)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    tmp = p.with_suffix(".part")
                    tmp.write_bytes(buf)
                    tmp.replace(p)

        out: dict[Key, Field] = {}
        todo = []
        for k in keys:
            p = self._path(run, step, dl_domain, k)
            if not p.exists():
                continue
            ck = (str(p), domain.tag)
            f = _DECODED.get(ck)
            if f is None:
                todo.append((k, p, ck))
            else:
                out[k] = f
        if todo:
            pad = 0.5
            bbox = (domain.lon0 - pad, domain.lon1 + pad, domain.lat0 - pad, domain.lat1 + pad)

            def _decode_all():
                res = []
                for k, p, ck in todo:
                    _, f = decode_message(p.read_bytes(), bbox)
                    res.append((k, ck, self._postprocess(k, f)))
                return res

            for k, ck, f in await asyncio.to_thread(_decode_all):
                _DECODED.put(ck, f)
                out[k] = f
        if want_zero_tp and out:
            ref = next(iter(out.values()))
            out[("tp", 0)] = Field(ref.lats, ref.lons, np.zeros_like(ref.values), "mm")
        return out


class TTLCache:
    def __init__(self, ttl: float):
        self.ttl, self.d = ttl, {}

    def get(self, k):
        v = self.d.get(k)
        if v and time.time() - v[0] < self.ttl:
            return v[1]
        return None

    def put(self, k, v):
        self.d[k] = (time.time(), v)


def cycle_floor(now: dt.datetime, every: int = 6) -> dt.datetime:
    now = now.replace(minute=0, second=0, microsecond=0, tzinfo=None)
    return now.replace(hour=(now.hour // every) * every)


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def cleanup_cache(keep_runs: int = 3) -> None:
    """Delete GRIB caches of all but the newest ``keep_runs`` runs of each model."""
    import shutil
    for mdir in GRIB_CACHE.iterdir() if GRIB_CACHE.exists() else []:
        if not mdir.is_dir():
            continue
        runs = sorted((d for d in mdir.iterdir() if d.is_dir()), key=lambda d: d.name)
        for d in runs[:-keep_runs]:
            shutil.rmtree(d, ignore_errors=True)
