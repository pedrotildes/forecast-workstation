"""NHC ATCF data for tropical (and post-tropical) cyclones: the best track (b-deck) and the
official and model forecast tracks (a-deck), archived with the event.

Landfall = first point where a track enters land (Natural Earth 50 m), interpolated
linearly between the track points.
"""
from __future__ import annotations

import datetime as dt
import gzip
import re
from functools import lru_cache

import httpx

from .archive import ARCHIVE_DIR
from .config import USER_AGENT

ATCF_DIR = ARCHIVE_DIR / "atcf"
H = dt.timedelta(hours=1)

# technique -> (label, colour, line style)
TECHS = {
    "OFCL": ("NHC (oficial)", "#111111", "-"),
    "AVNO": ("GFS", "#2563eb", "-"),
    "AEMN": ("GEFS (média)", "#60a5fa", "--"),
    "UKX": ("Met Office", "#16a34a", "-"),
    "CMC": ("CMC (Canadá)", "#a16207", "-"),
    "NVGM": ("NAVGEM", "#7c3aed", "-"),
    "HWRF": ("HWRF", "#dc2626", "-"),
    "HMON": ("HMON", "#f97316", "-"),
    "TVCN": ("Consenso TVCN", "#6b7280", "--"),
    "HCCA": ("Consenso HCCA", "#374151", ":"),
}


def valid_id(storm: str) -> str:
    s = storm.strip().upper()
    if not re.fullmatch(r"(AL|EP|CP)\d{2}\d{4}", s):
        raise ValueError("identificador NHC inválido (ex.: AL132018)")
    return s


def _urls(storm: str) -> dict:
    b, n, y = storm[:2].lower(), storm[2:4], storm[4:]
    arch = f"https://ftp.nhc.noaa.gov/atcf/archive/{y}"
    return {"a": [f"{arch}/a{b}{n}{y}.dat.gz", f"https://ftp.nhc.noaa.gov/atcf/aid_public/a{b}{n}{y}.dat.gz"],
            "b": [f"{arch}/b{b}{n}{y}.dat.gz", f"https://ftp.nhc.noaa.gov/atcf/btk/b{b}{n}{y}.dat"]}


async def fetch(storm: str, log=print) -> bool:
    """Download (once) the a- and b-decks into the archive."""
    storm = valid_id(storm)
    d = ATCF_DIR / storm
    d.mkdir(parents=True, exist_ok=True)
    ok = True
    async with httpx.AsyncClient(timeout=120, headers={"User-Agent": USER_AGENT}, follow_redirects=True) as c:
        for deck, urls in _urls(storm).items():
            p = d / f"{deck}.dat"
            if p.exists():
                continue
            for u in urls:
                try:
                    r = await c.get(u)
                except httpx.HTTPError:
                    continue
                if r.status_code == 200 and r.content:
                    raw = gzip.decompress(r.content) if u.endswith(".gz") else r.content
                    p.write_bytes(raw)
                    log(f"[atcf] {storm} {deck}-deck: {len(raw) // 1024} kB")
                    break
            else:
                ok = False
                log(f"[atcf] {storm} {deck}-deck indisponível")
    _load.cache_clear()
    return ok


def _ll(s: str) -> float:
    v = int(s[:-1]) / 10
    return v if s[-1] in "NE" else -v


@lru_cache(maxsize=8)
def _load(storm: str):
    d = ATCF_DIR / storm
    best, fc = {}, {}
    if (d / "b.dat").exists():
        for line in (d / "b.dat").read_text(errors="ignore").splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) < 11 or not p[6]:
                continue
            t = dt.datetime.strptime(p[2], "%Y%m%d%H")
            best[t] = {"time": t, "lat": _ll(p[6]), "lon": _ll(p[7]), "vmax": int(p[8] or 0),
                       "mslp": int(p[9] or 0), "type": p[10]}
    if (d / "a.dat").exists():
        for line in (d / "a.dat").read_text(errors="ignore").splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) < 9 or not p[6] or not p[7] or p[4] not in TECHS:
                continue
            base = dt.datetime.strptime(p[2], "%Y%m%d%H")
            tau = int(p[5])
            fc.setdefault((base, p[4]), {})[tau] = {
                "time": base + tau * H, "tau": tau, "lat": _ll(p[6]), "lon": _ll(p[7]),
                "vmax": int(p[8] or 0), "mslp": int(p[9] or 0) if len(p) > 9 and p[9] else 0}
    return ([best[t] for t in sorted(best)],
            {k: [v[t] for t in sorted(v)] for k, v in fc.items()})


def best_track(storm: str) -> list[dict]:
    return _load(valid_id(storm))[0]


def forecasts(storm: str) -> dict:
    return _load(valid_id(storm))[1]


def bases(storm: str) -> list[dt.datetime]:
    return sorted({b for b, t in forecasts(storm) if t == "OFCL"} or {b for b, _ in forecasts(storm)})


@lru_cache(maxsize=1)
def _land():
    import cartopy.io.shapereader as shpreader
    from shapely.ops import unary_union
    from shapely.prepared import prep
    # continents and large islands only (≥ 2 deg², e.g. Ireland, Great Britain): a forecast track
    # crossing Madeira or the Azores is not a landfall
    geoms = []
    for g in shpreader.Reader(shpreader.natural_earth("50m", "physical", "land")).geometries():
        parts = getattr(g, "geoms", [g])
        geoms += [q for q in parts if q.area >= 2.0]
    return prep(unary_union(geoms))


def landfall(track: list[dict], after: dt.datetime | None = None) -> dict | None:
    """First crossing from sea to land (lat, lon, time). Each segment between track points is
    walked in ~5 km steps (a track can cross a peninsula between two points at sea), then the
    crossing is refined by bisection."""
    import math

    from shapely.geometry import Point
    land = _land()
    pts = [p for p in track if after is None or p["time"] >= after]

    def on_land(a, b, f):
        return land.contains(Point(a["lon"] + f * (b["lon"] - a["lon"]), a["lat"] + f * (b["lat"] - a["lat"])))

    prev_land = None
    for a, b in zip(pts, pts[1:]):
        km = 111.2 * math.hypot(b["lat"] - a["lat"], (b["lon"] - a["lon"]) * math.cos(math.radians(a["lat"])))
        n = max(2, int(km / 5))
        fs = [i / n for i in range(n + 1)]
        states = [on_land(a, b, f) for f in fs]
        if prev_land is None:
            prev_land = states[0]
        for (f0, s0), (f1, s1) in zip(zip(fs, [prev_land] + states[1:]), zip(fs[1:], states[1:])):
            if not s0 and s1:
                lo, hi = f0, f1
                for _ in range(20):
                    mid = (lo + hi) / 2
                    if on_land(a, b, mid):
                        hi = mid
                    else:
                        lo = mid
                f = hi
                return {"lat": a["lat"] + f * (b["lat"] - a["lat"]), "lon": a["lon"] + f * (b["lon"] - a["lon"]),
                        "time": a["time"] + (b["time"] - a["time"]) * f}
        prev_land = states[-1]
    return None


def landfall_table(storm: str, techs=tuple(TECHS), start: dt.datetime | None = None) -> dict:
    """{"observed": landfall of the best track, "rows": [{base, tech, lead_h, lat, lon, time}]}"""
    best = best_track(storm)
    obs = landfall(best, after=start)
    rows = []
    if obs:
        for (base, tech), tr in sorted(forecasts(storm).items()):
            if tech not in techs or base >= obs["time"] or base.hour % 6:
                continue
            lf = landfall(tr)
            if lf and abs((lf["time"] - obs["time"]).total_seconds()) < 4 * 86400:
                rows.append({"base": base, "tech": tech, "lead_h": (obs["time"] - base).total_seconds() / 3600,
                             "lat": lf["lat"], "lon": lf["lon"], "time": lf["time"]})
    return {"observed": obs, "rows": rows}
