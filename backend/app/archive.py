"""Event archive ("congelar eventos").

Model servers keep data only for a short time (DWD ICON-EU ≈ 24 h, ECMWF Open Data a few
days on data.ecmwf.int, NOMADS ≈ 10 days, RainViewer radar ≈ 2 h). An *event* (name, time
window, region, models) makes the app copy the relevant fields — cropped to the event
region and quantised to 16 bit — into a permanent archive that the cache clean-up never
touches. ``Model.fetch`` transparently falls back to this archive, and archived runs/steps
show up in the run lists, so every product (maps, soundings, meteograms, comparisons) keeps
working for the event after the data has left the servers.

Layout (under ``ARCHIVE_DIR``)::

    events.json
    fields/<model>/<YYYYMMDDHH>/<step>/<key>.npz     (cropped, post-processed field)
    radar/<unix time>/<z>_<x>_<y>.png                 (RainViewer tiles captured live,
                                                       shared by all events)
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import math
import re
import shutil
import time
from pathlib import Path

import httpx
import numpy as np

from .config import ARCHIVE_DIR, USER_AGENT
from .grib import Field, decode_message

FIELDS_DIR = ARCHIVE_DIR / "fields"
RADAR_DIR = ARCHIVE_DIR / "radar"
EVENTS_FILE = ARCHIVE_DIR / "events.json"
for _d in (FIELDS_DIR, RADAR_DIR):
    _d.mkdir(parents=True, exist_ok=True)

H = dt.timedelta(hours=1)


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
_TOL = 0.6   # degrees: a stored field may serve a request slightly larger than it


# ===================================================================== field store
def _kstr(key) -> str:
    return f"{key[0]}_{key[1]}" if key[1] else key[0]


def _path(model_id: str, run: dt.datetime, step: int, key) -> Path:
    return FIELDS_DIR / model_id / run.strftime("%Y%m%d%H") / f"{step:03d}" / f"{_kstr(key)}.npz"


def save_field(model_id: str, run, step: int, key, f: Field) -> int:
    v = np.asarray(f.values, dtype=np.float64)
    finite = np.isfinite(v)
    lo = float(np.nanmin(v)) if finite.any() else 0.0
    hi = float(np.nanmax(v)) if finite.any() else 0.0
    scale = (hi - lo) / 65000.0 or 1.0
    q = np.where(finite, np.round((v - lo) / scale), 65535).astype(np.uint16)
    p = _path(model_id, run, step, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez_compressed(tmp, q=q, lo=lo, scale=scale, lats=f.lats.astype(np.float32),
                        lons=f.lons.astype(np.float32), units=f.units)
    tmp.replace(p)
    return p.stat().st_size


def load_field(model_id: str, run, step: int, key, domain) -> Field | None:
    p = _path(model_id, run, step, key)
    if not p.exists():
        return None
    z = np.load(p)
    lats, lons = z["lats"].astype(float), z["lons"].astype(float)
    if (domain.lon0 < lons[0] - _TOL or domain.lon1 > lons[-1] + _TOL or
            domain.lat0 < lats[0] - _TOL or domain.lat1 > lats[-1] + _TOL):
        return None           # archived region does not cover the request
    q = z["q"]
    vals = (q.astype(np.float32) * np.float32(z["scale"]) + np.float32(z["lo"]))
    vals[q == 65535] = np.nan
    f = Field(lats, lons, vals, str(z["units"]))
    pad = 0.5
    return f.crop(domain.lon0 - pad, domain.lon1 + pad, domain.lat0 - pad, domain.lat1 + pad)


def archived_runs(model_id: str) -> list[dt.datetime]:
    d = FIELDS_DIR / model_id
    if not d.exists():
        return []
    return sorted((dt.datetime.strptime(x.name, "%Y%m%d%H") for x in d.iterdir()
                   if x.is_dir() and re.fullmatch(r"\d{10}", x.name)), reverse=True)


def archived_steps(model_id: str, run: dt.datetime) -> list[int]:
    d = FIELDS_DIR / model_id / run.strftime("%Y%m%d%H")
    if not d.exists():
        return []
    return sorted(int(x.name) for x in d.iterdir() if x.is_dir() and any(x.iterdir()))


# ===================================================================== events
def load_events() -> list[dict]:
    if EVENTS_FILE.exists():
        try:
            return json.loads(EVENTS_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return []


def save_events(evs: list[dict]) -> None:
    tmp = EVENTS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(evs, indent=1, ensure_ascii=False, default=str))
    tmp.replace(EVENTS_FILE)


def get_event(eid: str) -> dict | None:
    return next((e for e in load_events() if e["id"] == eid), None)


def update_event(eid: str, **kw) -> None:
    evs = load_events()
    for e in evs:
        if e["id"] == eid:
            e.update(kw)
    save_events(evs)


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower().encode("ascii", "ignore").decode()).strip("-")
    return s or f"evento-{int(time.time())}"


def parse_t(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "")).replace(tzinfo=None)


# ===================================================================== what to archive
SFC = ["msl", "sp", "tp", "t2m", "d2m", "u10", "v10", "gust", "tcc", "lcc", "mcc", "hcc",
       "cape", "mlcape", "mucape", "cin", "pwat", "refc", "frzl", "tmax", "tmin", "orog"]
KEY_LEVELS = [850, 700, 500, 250]
KEY_VARS = ["gh", "t", "u", "v", "r"]
FULL_VARS = ["gh", "t", "u", "v", "r", "w"]


LIGHT = ["msl", "tp", "gust", "u10", "v10", "t2m"]
_RANK = {"light": 0, "key": 1, "full": 2}


def _keys(model, full) -> list:
    """Fields for a job kind: "full" (all levels), "key"/False (main levels) or "light"
    (surface pressure, wind, gust, precipitation, T2m — the hourly short-range chain)."""
    if full == "light":
        return [(n, 0) for n in LIGHT if model.has(n)]
    full = full in (True, "full")
    ks = [(n, 0) for n in SFC if model.has(n)]
    levels = model.levels if full else [p for p in KEY_LEVELS if p in model.levels]
    for n in (FULL_VARS if full else KEY_VARS):
        if n == "r" and not model.has("r"):
            n = "q"
        if model.has(n):
            ks += [(n, p) for p in levels]
    return ks


def plan_event(model, ev: dict) -> list[tuple[dt.datetime, int, bool]]:
    """(run, step, full) jobs.

    * analysis sequence — every run (00/06/12/18Z) started during the event: step 0 with
      the full vertical resolution (soundings) and step 3 with the key fields;
    * forecasts — the 00Z and 12Z runs from ``lead_days`` before the event, every 6 h
      inside the window (24 h earlier too, for precipitation accumulations)."""
    t0, t1 = parse_t(ev["start"]), parse_t(ev["end"])
    lead = float(ev.get("lead_days", 2))
    now = _utcnow()
    jobs: dict[tuple, str] = {}

    def add(run, step, kind):
        cur = jobs.get((run, step))
        if cur is None or _RANK[kind] > _RANK[cur]:
            jobs[(run, step)] = kind

    # short-range chain step: hourly where the model provides it (GFS, ICON-EU)
    chain = {"gfs": 1, "icon_eu": 1, "ecmwf": 3, "aifs": 6}.get(model.id, 3)
    run = (t0 - dt.timedelta(days=lead)).replace(minute=0, second=0, microsecond=0)
    run = run.replace(hour=(run.hour // 6) * 6)
    while run <= min(t1, now):
        if run >= t0 - 6 * H:
            add(run, 0, "full")
            if model.id != "aifs":
                add(run, 3, "key")
            for s in range(chain, 7, chain):           # continuous 0–6 h chain
                add(run, s, "light")
        if run.hour in (0, 12):
            nominal = model.last_nominal_step(run) or 120
            for s in range(0, nominal + 1, 6):
                v = run + s * H
                if t0 - 24 * H <= v <= t1:
                    add(run, s, "key")
        run += 6 * H
    return [(r, s, k) for (r, s), k in sorted(jobs.items())]


def _absent_file(model_id, run, step) -> Path:
    return _path(model_id, run, step, ("x", 0)).parent / "absent.json"


def _absent(model_id, run, step) -> set:
    p = _absent_file(model_id, run, step)
    try:
        return {tuple(k) for k in json.loads(p.read_text())}
    except (OSError, json.JSONDecodeError):
        return set()


def _mark_absent(model_id, run, step, keys) -> None:
    """Remember fields the server does not offer for this run/step (never ask again)."""
    if not keys:
        return
    p = _absent_file(model_id, run, step)
    p.parent.mkdir(parents=True, exist_ok=True)
    cur = _absent(model_id, run, step) | {tuple(k) for k in keys}
    p.write_text(json.dumps(sorted(cur)))


def _missing(model, run, step, keys) -> list:
    gone = _absent(model.id, run, step)
    return [k for k in keys if tuple(k) not in gone and not _path(model.id, run, step, k).exists()]


async def archive_event(eid: str, log=print) -> dict:
    """Download (incrementally) everything an event needs. Safe to call repeatedly."""
    from .models import MODELS
    from .models.base import Domain
    from .regions import REGIONS
    ev = get_event(eid)
    if ev is None:
        raise KeyError(eid)
    box = REGIONS[ev.get("region", "nea")].download_domain
    if is_historical(ev):
        return await archive_historical(ev, (box.lon0, box.lon1, box.lat0, box.lat1), log)
    dom = Domain(box.lon0, box.lon1, box.lat0, box.lat1)
    stats = {}

    async def do_model(mid):
        m = MODELS[mid]
        if not m.covers_box(dom.lon0, dom.lon1, dom.lat0, dom.lat1):
            return
        st = {"done": 0, "total": 0, "fields": 0, "bytes": 0, "unavailable": 0}
        stats[mid] = st
        jobs = plan_event(m, ev)
        st["total"] = len(jobs)
        sem = asyncio.Semaphore(2 if m.global_download else 3)

        async def job(run, step, full):
            keys = _keys(m, full)
            miss = _missing(m, run, step, keys)
            if not miss:
                st["done"] += 1
                return
            async with sem:
                try:
                    got = await m._download(run, step, miss, dom)
                except Exception as e:  # noqa: BLE001 (run absent on the server, network…)
                    got = {}
                    log(f"[{mid}] {run:%Y%m%d%H}+{step}: {e}")
                step_dir = _path(m.id, run, step, ("x", 0)).parent
                if got or (step_dir.exists() and any(step_dir.glob("*.npz"))):
                    # the server answered for this step before/now: what it does not send
                    # (e.g. a field this model does not produce) does not exist
                    _mark_absent(m.id, run, step, [k for k in miss if k not in got])

                def store():
                    n = b = 0
                    bbox = (dom.lon0 - 0.5, dom.lon1 + 0.5, dom.lat0 - 0.5, dom.lat1 + 0.5)
                    for k, buf in got.items():
                        try:
                            _, f = decode_message(buf, bbox)
                            f = m._postprocess(k, f)
                            if f.values.size:
                                b += save_field(m.id, run, step, k, f)
                                n += 1
                        except Exception as e:  # noqa: BLE001
                            log(f"[{mid}] decode {k}: {e}")
                    return n, b

                await asyncio.to_thread(store)
                st["done"] += 1
                update_event(eid, progress=stats, updated=_utcnow().isoformat())

        await asyncio.gather(*(job(*j) for j in jobs))

        # totals from disk (not just this pass)
        def totals():
            n = b = complete = empty = 0
            for run, step, full in jobs:
                d = _path(m.id, run, step, ("x", 0)).parent
                files = list(d.glob("*.npz")) if d.exists() else []
                n += len(files)
                b += sum(f.stat().st_size for f in files)
                if not files:
                    empty += 1
                elif not _missing(m, run, step, _keys(m, full)):
                    complete += 1
            return n, b, complete, empty

        st["fields"], st["bytes"], st["done"], st["unavailable"] = await asyncio.to_thread(totals)

    await asyncio.gather(*(do_model(mid) for mid in ev.get("models", [])))
    update_event(eid, progress=stats, updated=_utcnow().isoformat(),
                 size=event_size(eid))
    return stats


def event_size(eid: str) -> int:
    """Approximate disk use of an event (fields inside its window + its radar)."""
    ev = get_event(eid)
    if not ev:
        return 0
    total = 0
    for t in radar_frames(eid):
        total += sum(p.stat().st_size for p in (RADAR_DIR / str(t)).glob("*.png"))
    t0 = parse_t(ev["start"]) - dt.timedelta(days=float(ev.get("lead_days", 2)) + 0.3)
    t1 = parse_t(ev["end"])
    for mdir in FIELDS_DIR.iterdir():
        for rdir in mdir.iterdir() if mdir.is_dir() else []:
            try:
                r = dt.datetime.strptime(rdir.name, "%Y%m%d%H")
            except ValueError:
                continue
            if t0 <= r <= t1:
                total += sum(p.stat().st_size for p in rdir.rglob("*.npz"))
    return total


# ===================================================================== radar (RainViewer)
RV_INDEX = "https://api.rainviewer.com/public/weather-maps.json"
RADAR_ZOOMS = {3: None, 4: None, 5: None, 6: (-20.0, 4.0, 30.0, 45.0)}   # z6 only on the core


def _tiles(z: int, lon0, lon1, lat0, lat1):
    n = 2 ** z

    def tx(lon):
        return int((lon + 180) / 360 * n)

    def ty(lat):
        la = math.radians(max(min(lat, 85), -85))
        return int((1 - math.log(math.tan(la) + 1 / math.cos(la)) / math.pi) / 2 * n)

    for x in range(tx(lon0), tx(lon1) + 1):
        for y in range(ty(lat1), ty(lat0) + 1):
            yield x, y


async def capture_radar(eid: str, log=print) -> int:
    """Save every RainViewer frame currently offered that falls in the event window."""
    from .regions import REGIONS
    ev = get_event(eid)
    if ev is None:
        return 0
    t0, t1 = parse_t(ev["start"]), parse_t(ev["end"])
    box = REGIONS[ev.get("region", "nea")].extent
    saved = 0
    async with httpx.AsyncClient(timeout=30, headers={"User-Agent": USER_AGENT}) as c:
        idx = (await c.get(RV_INDEX)).json()
        host = idx["host"]
        for fr in idx["radar"]["past"]:
            ft = dt.datetime.fromtimestamp(fr["time"], dt.timezone.utc).replace(tzinfo=None)
            if not (t0 <= ft <= t1):
                continue
            d = RADAR_DIR / str(fr["time"])
            if (d / "done").exists():
                continue
            d.mkdir(parents=True, exist_ok=True)
            ok = True
            for z, sub in RADAR_ZOOMS.items():
                lon0, lon1, lat0, lat1 = sub or box
                for x, y in _tiles(z, lon0, lon1, lat0, lat1):
                    p = d / f"{z}_{x}_{y}.png"
                    if p.exists():
                        continue
                    url = f"{host}{fr['path']}/512/{z}/{x}/{y}/4/1_1.png"
                    try:
                        r = await c.get(url)
                        if r.status_code == 200 and r.content[:4] == b"\x89PNG":
                            p.write_bytes(r.content)
                        else:
                            ok = False
                    except httpx.HTTPError:
                        ok = False
                    await asyncio.sleep(0.15)       # stay well below RainViewer's limit
            if ok:
                (d / "done").write_text("1")
                saved += 1
    return saved


def radar_frames(eid: str) -> list[int]:
    """Archived radar frames (unix times) inside the event window."""
    ev = get_event(eid)
    if ev is None:
        return []
    a = parse_t(ev["start"]).replace(tzinfo=dt.timezone.utc).timestamp()
    b = parse_t(ev["end"]).replace(tzinfo=dt.timezone.utc).timestamp()
    return sorted(int(x.name) for x in RADAR_DIR.iterdir()
                  if x.name.isdigit() and a <= int(x.name) <= b and (x / "done").exists())


def delete_event(eid: str, delete_fields: bool = True) -> None:
    """Remove an event; its fields are deleted only if no other event still needs them."""
    evs = load_events()
    ev = next((e for e in evs if e["id"] == eid), None)
    if ev is None:
        return
    rest = [e for e in evs if e["id"] != eid]
    save_events(rest)
    if not delete_fields:
        return
    # radar frames not covered by any remaining event
    for t in [int(x.name) for x in RADAR_DIR.iterdir() if x.name.isdigit()]:
        ts = dt.datetime.fromtimestamp(t, dt.timezone.utc).replace(tzinfo=None)
        inside = parse_t(ev["start"]) <= ts <= parse_t(ev["end"])
        if inside and not any(parse_t(e["start"]) <= ts <= parse_t(e["end"]) for e in rest):
            shutil.rmtree(RADAR_DIR / str(t), ignore_errors=True)

    def needed(r: dt.datetime) -> bool:
        for e in rest:
            a = parse_t(e["start"]) - dt.timedelta(days=float(e.get("lead_days", 2)) + 0.3)
            if a <= r <= parse_t(e["end"]):
                return True
        return False

    a = parse_t(ev["start"]) - dt.timedelta(days=float(ev.get("lead_days", 2)) + 0.3)
    b = parse_t(ev["end"])
    for mdir in FIELDS_DIR.iterdir():
        for rdir in list(mdir.iterdir()) if mdir.is_dir() else []:
            try:
                r = dt.datetime.strptime(rdir.name, "%Y%m%d%H")
            except ValueError:
                continue
            if a <= r <= b and not needed(r):
                shutil.rmtree(rdir, ignore_errors=True)


HISTORICAL_DAYS = 10      # events that ended longer ago than this come from the archives


def is_historical(ev: dict) -> bool:
    """Too old for the forecast servers: filled from reanalysis/analysis archives instead."""
    return bool(ev.get("historical")) or parse_t(ev["end"]) < _utcnow() - dt.timedelta(days=HISTORICAL_DAYS)


def create_event(name: str, start: str, end: str, models: list[str], region: str = "nea",
                 lead_days: float = 2, notes: str = "", atcf: str | None = None) -> dict:
    evs = load_events()
    eid = slug(name)
    base, k = eid, 2
    while any(e["id"] == eid for e in evs):
        eid, k = f"{base}-{k}", k + 1
    ev = {"id": eid, "name": name, "start": parse_t(start).isoformat(),
          "end": parse_t(end).isoformat(), "models": models, "region": region,
          "lead_days": lead_days, "notes": notes,
          "created": _utcnow().isoformat(), "progress": {}}
    if atcf:
        ev["atcf"] = atcf
    if parse_t(end) < _utcnow() - dt.timedelta(days=HISTORICAL_DAYS):
        ev["historical"] = True
    evs.append(ev)
    save_events(evs)
    return ev


_HIST_RUNNING: set[str] = set()


async def archive_historical(ev: dict, box, log=print) -> dict:
    """Historical event: ERA5 reanalysis, GFS analyses (NOAA archive), NHC ATCF tracks and
    radiosondes. Each part is skipped when already archived; safe to call again."""
    eid = ev["id"]
    if eid in _HIST_RUNNING:
        return ev.get("progress", {})
    _HIST_RUNNING.add(eid)
    prog = dict(ev.get("progress") or {})

    def mark(part, **kw):
        prog[part] = {**prog.get(part, {}), **kw}
        update_event(eid, progress=prog, updated=_utcnow().isoformat())
    try:
        models = ev.get("models", [])
        if "era5" in models:
            mark("era5", estado="a importar")
            try:
                from .era5 import import_event as era5_import
                n = await era5_import(ev, box, log)
                mark("era5", estado="concluído", campos=sum(n.values()))
            except Exception as e:  # noqa: BLE001
                mark("era5", estado=f"erro: {e}")
                log(f"[era5] {eid}: {e}")
        if "gfs" in models:
            mark("gfs", estado="a importar")
            try:
                from .gfs_hist import import_event as gfs_import
                n = await gfs_import(ev, box, log)
                mark("gfs", estado="concluído", campos=n)
            except Exception as e:  # noqa: BLE001
                mark("gfs", estado=f"erro: {e}")
                log(f"[gfs-arquivo] {eid}: {e}")
        if ev.get("atcf"):
            from . import atcf
            ok = await atcf.fetch(ev["atcf"], log)
            mark("atcf", estado="concluído" if ok else "incompleto", ciclone=ev["atcf"])
        try:
            from .obs import capture_event_obs
            n = await capture_event_obs(ev, log)
            mark("radiossondas", estado="concluído", sondagens=n)
        except Exception as e:  # noqa: BLE001
            log(f"[obs] {eid}: {e}")
        update_event(eid, progress=prog, updated=_utcnow().isoformat(), size=event_size(eid))
        return prog
    finally:
        _HIST_RUNNING.discard(eid)


async def archiver_loop(log=print) -> None:
    """Background task: keep open events archived and capture radar while they last
    (skipped while a stand-alone ``python -m app.archive`` worker is doing it)."""
    while True:
        if other_worker_running():
            await asyncio.sleep(600)
            continue
        now = _utcnow()
        for ev in load_events():
            end = parse_t(ev["end"])
            if now > end + dt.timedelta(days=6):
                continue            # long finished: nothing new will appear on servers
            try:
                if parse_t(ev["start"]) - dt.timedelta(hours=3) <= now <= end + 2 * H:
                    await capture_radar(ev["id"], log)
                from .obs import capture_event_obs
                await capture_event_obs(ev, log)
                await archive_event(ev["id"], log)
            except Exception as e:  # noqa: BLE001
                log(f"[archive] {ev['id']}: {e}")
        await asyncio.sleep(600)


# ===================================================================== stand-alone worker
PID_FILE = ARCHIVE_DIR / "worker.pid"


def other_worker_running() -> bool:
    """True if a stand-alone archiver (``python -m app.archive``) is alive."""
    import os
    try:
        pid = int(PID_FILE.read_text())
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    if pid == os.getpid():
        return False
    if os.name == "posix":
        # after a reboot the PID in the file may belong to an unrelated process
        import subprocess
        try:
            cmd = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True,
                                 text=True, timeout=5).stdout
        except (OSError, subprocess.SubprocessError):
            return True
        return "app.archive" in cmd
    return True


async def _worker_main():
    """`python -m app.archive` — archive all events now and keep capturing radar until
    every event has ended (used when the app itself is not running)."""
    import os
    if other_worker_running():          # e.g. started again by the login agent
        print("arquivador já em execução — nada a fazer", flush=True)
        return
    PID_FILE.write_text(str(os.getpid()))
    def log(msg):
        line = f"{_utcnow():%Y-%m-%d %H:%M:%S} {msg}"
        print(line, flush=True)
        with open(ARCHIVE_DIR / "worker.log", "a") as fh:
            fh.write(line + "\n")

    while True:
        evs = load_events()
        now = _utcnow()
        for ev in evs:
            if parse_t(ev["start"]) - 3 * H <= now <= parse_t(ev["end"]) + 2 * H:
                try:
                    n = await capture_radar(ev["id"], log)
                    log(f"radar {ev['id']}: +{n} imagens (total {len(radar_frames(ev['id']))})")
                except Exception as e:  # noqa: BLE001
                    log(f"radar {ev['id']}: erro {e}")
        for ev in sorted(evs, key=lambda e: e["end"]):
            if now > parse_t(ev["end"]) + dt.timedelta(days=6):
                continue
            t = time.time()
            try:
                from .obs import capture_event_obs
                n = await capture_event_obs(ev, log)
                log(f"radiossondas {ev['id']}: {n} arquivadas")
            except Exception as e:  # noqa: BLE001
                log(f"radiossondas {ev['id']}: erro {e}")
            st = await archive_event(ev["id"], log)
            log(f"campos {ev['id']}: " + ", ".join(
                f"{k} {v['done']}/{v['total']} ({v['fields']} campos, {v['bytes'] / 1e6:.0f} MB, "
                f"{v['unavailable']} indisponíveis)" for k, v in st.items()) + f" [{time.time() - t:.0f}s]")
        # keep going ~12 h after the end: runs started near the end are published later
        if all(now > parse_t(e["end"]) + 12 * H for e in evs):
            log("todos os eventos terminaram — arquivador encerrado")
            return
        await asyncio.sleep(600)


if __name__ == "__main__":
    asyncio.run(_worker_main())
