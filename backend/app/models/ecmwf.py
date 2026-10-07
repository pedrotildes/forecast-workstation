"""ECMWF Open Data — IFS (HRES 0.25°) and AIFS Single.

Files are global GRIB2 with a JSON-lines ``.index`` giving the byte offset and
length of every message, so we fetch only the messages we need via HTTP range
requests (Herbie-style) and cache each one individually.  The AWS mirror is
used as a fallback when data.ecmwf.int is slow or unavailable.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json

import httpx

from ..config import GRIB_CACHE, HTTP_TIMEOUT, USER_AGENT
from ..grib import Field
from .base import Domain, FieldUnavailable, Key, Model, TTLCache, cycle_floor, utcnow

MIRRORS = ["https://data.ecmwf.int/forecasts",
           "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"]

PL = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

# canonical -> (ECMWF param, is_upper)
_MAP = {
    "gh": ("gh", True), "t": ("t", True), "r": ("r", True), "q": ("q", True),
    "u": ("u", True), "v": ("v", True), "w": ("w", True),
    "msl": ("msl", False), "sp": ("sp", False), "t2m": ("2t", False),
    "d2m": ("2d", False), "u10": ("10u", False), "v10": ("10v", False),
    "tp": ("tp", False), "mucape": ("mucape", False), "pwat": ("tcwv", False),
    "tcc": ("tcc", False), "lcc": ("lcc", False), "mcc": ("mcc", False),
    "hcc": ("hcc", False), "gust": (("10fg", "10fg3", "10fg6"), False),
    "tmax": (("mx2t3", "mx2t6"), False), "tmin": (("mn2t3", "mn2t6"), False),
}


class _ECMWFBase(Model):
    global_download = True
    path_model = "ifs"
    resolution = "0.25°"
    levels = PL
    run_every = 6

    def __init__(self):
        self._idx_cache: dict = {}
        self._steps_cache = TTLCache(300)
        self._runs_cache = TTLCache(300)
        self._sem = asyncio.Semaphore(8)

    # --- layout ------------------------------------------------------
    def stream(self, run: dt.datetime) -> str:
        return "oper"

    def nominal_steps(self, run: dt.datetime) -> list[int]:
        raise NotImplementedError

    def _url(self, mirror: str, run: dt.datetime, step: int, ext: str) -> str:
        s = self.stream(run)
        return (f"{mirror}/{run:%Y%m%d}/{run:%H}z/{self.path_model}/0p25/{s}/"
                f"{run:%Y%m%d%H}0000-{step}h-{s}-fc.{ext}")

    # --- discovery ---------------------------------------------------
    async def _exists(self, c: httpx.AsyncClient, run, step) -> bool:
        for m in MIRRORS:
            try:
                r = await c.head(self._url(m, run, step, "index"))
                if r.status_code == 200:
                    return True
            except httpx.HTTPError:
                continue
        return False

    async def list_runs(self) -> list[dt.datetime]:
        cached = self._runs_cache.get("runs")
        if cached is not None:
            return cached
        base = cycle_floor(utcnow(), self.run_every)
        cands = [base - dt.timedelta(hours=self.run_every * i)
                 for i in range(int(48 / self.run_every))]
        async with httpx.AsyncClient(timeout=20, headers={"User-Agent": USER_AGENT}) as c:
            ok = await asyncio.gather(*(self._exists(c, r, 0) for r in cands))
        runs = [r for r, good in zip(cands, ok) if good][:5]
        self._runs_cache.put("runs", runs)
        return runs

    async def list_steps(self, run: dt.datetime) -> list[int]:
        s = self._steps_cache.get(run)
        if s is not None:
            return s
        nominal = self.nominal_steps(run)
        async with httpx.AsyncClient(timeout=20, headers={"User-Agent": USER_AGENT}) as c:
            if await self._exists(c, run, nominal[-1]):
                n = len(nominal)
            else:  # binary search for the last produced step
                lo, hi = 0, len(nominal) - 1
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if await self._exists(c, run, nominal[mid]):
                        lo = mid
                    else:
                        hi = mid - 1
                n = lo + 1
        s = nominal[:n]
        self._steps_cache.put(run, s)
        return s

    # --- download ----------------------------------------------------
    async def _index(self, c: httpx.AsyncClient, run, step) -> tuple[str, list[dict]]:
        ck = (run, step)
        if ck in self._idx_cache:
            return self._idx_cache[ck]
        p = GRIB_CACHE / self.id / run.strftime("%Y%m%d%H") / f"{step:03d}" / "index.jsonl"
        last_err = None
        for m in MIRRORS:
            try:
                if p.exists():
                    text = p.read_text()
                else:
                    r = await c.get(self._url(m, run, step, "index"))
                    r.raise_for_status()
                    text = r.text
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(text)
                entries = [json.loads(line) for line in text.splitlines() if line.strip()]
                res = (self._url(m, run, step, "grib2"), entries)
                self._idx_cache[ck] = res
                return res
            except (httpx.HTTPError, json.JSONDecodeError) as e:
                last_err = e
        raise FieldUnavailable(f"{self.name}: index unavailable for {run:%Y%m%d%H} +{step}h ({last_err})")

    def _find(self, entries: list[dict], key: Key) -> dict | None:
        name, level = key
        if name not in _MAP:
            return None
        param, upper = _MAP[name]
        params = param if isinstance(param, tuple) else (param,)
        for e in entries:
            if e.get("param") not in params:
                continue
            if upper:
                if e.get("levtype") == "pl" and int(e.get("levelist", -1)) == level:
                    return e
            elif e.get("levtype") == "sfc":
                return e
        return None

    async def _range(self, c: httpx.AsyncClient, url: str, e: dict) -> bytes:
        a, n = int(e["_offset"]), int(e["_length"])
        async with self._sem:
            for attempt in range(3):
                try:
                    r = await c.get(url, headers={"Range": f"bytes={a}-{a + n - 1}"})
                    if r.status_code in (200, 206) and len(r.content) == n:
                        return r.content
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(1.0 * (attempt + 1))
        raise FieldUnavailable(f"download failed: {url} [{a}+{n}]")

    async def _download(self, run, step, keys, domain: Domain) -> dict[Key, bytes]:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, http2=False,
                                     headers={"User-Agent": USER_AGENT}) as c:
            url, entries = await self._index(c, run, step)
            todo = [(k, e) for k in keys if (e := self._find(entries, k))]
            blobs = await asyncio.gather(*(self._range(c, url, e) for _, e in todo))
        return {k: b for (k, _), b in zip(todo, blobs)}

    def _postprocess(self, key: Key, f: Field) -> Field:
        name = key[0]
        if name == "tp":
            if f.units.strip() == "m":       # IFS: metres; AIFS: kg m-2 (= mm)
                f.values = f.values * 1000.0
            f.units = "mm"
        elif name in ("tcc", "lcc", "mcc", "hcc"):
            if float(f.values.max()) <= 1.01:
                f.values = f.values * 100.0
            f.units = "%"
        return f


class ECMWF_IFS(_ECMWFBase):
    id = "ecmwf"
    name = "ECMWF IFS"
    path_model = "ifs"
    description = "ECMWF IFS HRES (Open Data) — 0.25°, 00/12Z até +360 h, 06/18Z até +90 h"
    native = {"gh", "t", "r", "q", "u", "v", "w", "msl", "sp", "t2m", "d2m",
              "u10", "v10", "tp", "mucape", "pwat", "tcc", "gust", "tmax", "tmin"}

    def stream(self, run):
        return "oper" if run.hour in (0, 12) else "scda"

    def nominal_steps(self, run):
        if run.hour in (0, 12):
            return list(range(0, 144, 3)) + list(range(144, 361, 6))
        return list(range(0, 91, 3))


class ECMWF_AIFS(_ECMWFBase):
    id = "aifs"
    name = "ECMWF AIFS"
    path_model = "aifs-single"
    description = "ECMWF AIFS Single (modelo de IA) — 0.25°, 4 runs/dia, até +360 h"
    native = {"gh", "t", "q", "u", "v", "w", "msl", "sp", "t2m", "d2m",
              "u10", "v10", "tp", "pwat", "tcc", "lcc", "mcc", "hcc"}

    def nominal_steps(self, run):
        return list(range(0, 361, 6))


# ---------------------------------------------------------------------------
# ECMWF ENS (51 members) probability products ("ep"): probability that the 850 hPa
# temperature standardized anomaly (w.r.t. the model climate) exceeds ±1/1.5/2 σ.
ENS_PROB = {
    "p_gt1": "ptsa_gt_1stdev", "p_gt1p5": "ptsa_gt_1p5stdev", "p_gt2": "ptsa_gt_2stdev",
    "p_lt1": "ptsa_lt_1stdev", "p_lt1p5": "ptsa_lt_1p5stdev", "p_lt2": "ptsa_lt_2stdev",
}


class ECMWF_ENS_EP(_ECMWFBase):
    id = "ens"
    name = "ECMWF ENS"
    path_model = "ifs"
    description = "ECMWF ENS 51 membros — probabilidades de anomalia de T850 (12 em 12 h, até +360 h)"
    native = set(ENS_PROB)
    levels = []
    run_every = 12

    def stream(self, run):
        return "enfo"

    @staticmethod
    def _file_step(step: int) -> int:
        return 240 if step <= 240 else 360

    def _url(self, mirror, run, step, ext):
        fs = self._file_step(step)
        return (f"{mirror}/{run:%Y%m%d}/{run:%H}z/ifs/0p25/enfo/"
                f"{run:%Y%m%d%H}0000-{fs}h-enfo-ep.{ext}")

    def nominal_steps(self, run):
        return list(range(12, 361, 12))

    async def list_steps(self, run):
        s = self._steps_cache.get(run)
        if s is not None:
            return s
        async with httpx.AsyncClient(timeout=20, headers={"User-Agent": USER_AGENT}) as c:
            full = await self._exists(c, run, 360)
        s = list(range(12, 361 if full else 241, 12))
        self._steps_cache.put(run, s)
        return s

    async def _download(self, run, step, keys, domain):
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT}) as c:
            url, entries = await self._index(c, run, self._file_step(step))
            todo = []
            for k in keys:
                param = ENS_PROB.get(k[0])
                e = next((e for e in entries if e.get("param") == param
                          and str(e.get("step")) == str(step)), None)
                if e:
                    todo.append((k, e))
            blobs = await asyncio.gather(*(self._range(c, url, e) for _, e in todo))
        return {k: b for (k, _), b in zip(todo, blobs)}

    def _postprocess(self, key, f):
        f.units = "%"          # ECMWF "ep" probabilities are already in percent
        return f


ENS = ECMWF_ENS_EP()
