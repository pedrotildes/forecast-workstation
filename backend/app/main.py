"""Forecast Workstation — HTTP API and static front-end."""
from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import FRONTEND_DIR, IMG_CACHE, VERSION
from .maps.compose import PRESETS, build_job, fmt_time, preset_layers
from .models import MODELS, FieldUnavailable, get_model
from .products import VARS
from .regions import REGIONS
from .models.base import cleanup_cache
from .series import PROGRESS, has_profile, point_series, resolve_run
from .workers import (compare_sounding_worker, render_map_worker, render_meteogram_worker,
                      render_multi_worker, render_sounding_worker, render_thermogram_worker)

app = FastAPI(title="Forecast Workstation", version=VERSION)
POOL: ProcessPoolExecutor | None = None
RENDER_VERSION = "7"


@app.on_event("startup")
async def _startup():
    global POOL
    import multiprocessing as mp
    POOL = ProcessPoolExecutor(max_workers=3, mp_context=mp.get_context("spawn"))
    asyncio.create_task(_housekeeping())


async def _housekeeping():
    """Keep only recent runs in the GRIB cache and drop old rendered images."""
    import time as _t
    while True:
        try:
            await asyncio.to_thread(cleanup_cache, 3)
            cutoff = _t.time() - 2 * 86400
            for f in IMG_CACHE.iterdir():
                if f.stat().st_mtime < cutoff:
                    f.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
        await asyncio.sleep(3600)


@app.on_event("shutdown")
async def _shutdown():
    if POOL:
        POOL.shutdown(cancel_futures=True)


def _parse_run(run: str) -> dt.datetime:
    try:
        return dt.datetime.strptime(run, "%Y%m%d%H")
    except ValueError:
        raise HTTPException(400, "run deve ter o formato AAAAMMDDHH") from None


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_clean(v) for v in o.tolist()]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else round(f, 3)
    if isinstance(o, np.integer):
        return int(o)
    return o


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": VERSION}


# ---------------------------------------------------------------- catalogue
@app.get("/api/models")
async def models():
    return [m.info() for m in MODELS.values()]


@app.get("/api/models/{model_id}/runs")
async def runs(model_id: str):
    m = _model(model_id)
    rs = await m.list_runs()
    out = []
    for i, r in enumerate(rs):
        steps = await m.list_steps(r) if i < 2 else None
        out.append({"run": r.strftime("%Y%m%d%H"), "label": fmt_time(r), "steps": steps})
    return out


@app.get("/api/models/{model_id}/runs/{run}/steps")
async def steps(model_id: str, run: str):
    return await _model(model_id).list_steps(_parse_run(run))


@app.get("/api/regions")
async def regions():
    return [r.info() for r in REGIONS.values()]


@app.get("/api/catalog")
async def catalog(model: str = "gfs"):
    m = _model(model)
    return {
        "variables": [v.info(m) for v in VARS if v.is_available(m)],
        "presets": [{"id": k, "name": v["name"], "layers": preset_layers(k, m)}
                    for k, v in PRESETS.items()],
        "wind_levels": [0] + m.levels,
    }


def _model(model_id):
    try:
        return get_model(model_id)
    except KeyError as e:
        raise HTTPException(404, str(e)) from None


# ---------------------------------------------------------------- maps
@app.get("/api/map")
async def map_png(model: str, run: str, step: int, region: str = "europe",
                  preset: str | None = None, layers: str | None = None):
    m = _model(model)
    r = _parse_run(run)
    if region not in REGIONS:
        raise HTTPException(404, f"região desconhecida: {region}")
    if layers:
        try:
            lys = json.loads(layers)
            assert isinstance(lys, list)
        except Exception:  # noqa: BLE001
            raise HTTPException(400, "layers deve ser uma lista JSON") from None
    else:
        lys = preset_layers(preset or "sfc_precip", m)
    key = hashlib.sha1(json.dumps([RENDER_VERSION, model, run, step, region, lys],
                                  sort_keys=True).encode()).hexdigest()
    png_p, meta_p = IMG_CACHE / f"{key}.png", IMG_CACHE / f"{key}.json"
    if png_p.exists() and meta_p.exists():
        meta = json.loads(meta_p.read_text())
    else:
        steps_ = await m.list_steps(r)
        if step not in steps_:
            raise HTTPException(404, f"passo +{step}h não disponível para {m.name} {run}")
        try:
            job = await build_job(m, r, step, steps_, region, lys)
        except FieldUnavailable as e:
            raise HTTPException(404, str(e)) from None
        png, rect = await asyncio.get_running_loop().run_in_executor(POOL, render_map_worker, job)
        meta = {"axes": rect}
        png_p.write_bytes(png)
        meta_p.write_text(json.dumps(meta))
    return FileResponse(png_p, media_type="image/png", headers={
        "X-Map-Axes": ",".join(f"{x:.6f}" for x in meta["axes"]),
        "Access-Control-Expose-Headers": "X-Map-Axes",
        "Cache-Control": "public, max-age=86400"})


@app.get("/api/locate")
async def locate(region: str, fx: float, fy: float):
    if region not in REGIONS:
        raise HTTPException(404, "região desconhecida")
    lon, lat = REGIONS[region].to_lonlat(fx, fy)
    return {"lat": round(lat, 3), "lon": round(lon, 3)}


# ---------------------------------------------------------------- soundings
async def _sounding(model, run, step, lat, lon, parcel, name):
    from .sounding.profile import get_profile
    m = _model(model)
    r = _parse_run(run)
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise HTTPException(400, "coordenadas inválidas")
    try:
        snd = await get_profile(m, r, step, lat, lon)
    except FieldUnavailable as e:
        raise HTTPException(404, str(e)) from None
    valid = r + dt.timedelta(hours=step)
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    place = f"{name} · " if name else ""
    title = (f"{m.name} {m.resolution} — Sondagem prevista   {place}"
             f"{abs(lat):.2f}°{ns} {abs(lon):.2f}°{ew}")
    subtitle = (f"Run {fmt_time(r)}   ·   Válido {fmt_time(valid)} (+{step} h)   ·   "
                f"Superfície {snd['surface']['p']:.0f} hPa, {snd['surface']['z']:.0f} m")
    return await asyncio.get_running_loop().run_in_executor(
        POOL, render_sounding_worker, snd, title, subtitle, parcel)


@app.get("/api/sounding.png")
async def sounding_png(model: str, run: str, step: int, lat: float, lon: float,
                       parcel: str = "ML", name: str = ""):
    png, _ = await _sounding(model, run, step, lat, lon, parcel, name)
    return Response(png, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/sounding.json")
async def sounding_json(model: str, run: str, step: int, lat: float, lon: float,
                        parcel: str = "ML"):
    _, data = await _sounding(model, run, step, lat, lon, parcel, "")
    return JSONResponse(_clean(data))


# ---------------------------------------------------------------- meteograms
async def _cached_png(key_parts, producer):
    key = hashlib.sha1(json.dumps([RENDER_VERSION] + key_parts, sort_keys=True,
                                  default=str).encode()).hexdigest()
    p = IMG_CACHE / f"mg_{key}.png"
    if not p.exists():
        png = await producer()
        p.write_bytes(png)
    return FileResponse(p, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})


def _check_point(lat, lon):
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise HTTPException(400, "coordenadas inválidas")


@app.get("/api/progress")
async def progress(job: str):
    return PROGRESS.get(job, {})


@app.get("/api/meteogram.png")
async def meteogram_png(model: str, lat: float, lon: float, run: str | None = None,
                        hours: int = 240, profile: bool = True, name: str = "",
                        job: str | None = None):
    _check_point(lat, lon)
    m = _model(model)
    r = _parse_run(run) if run else (await m.list_runs())[0]
    profile = profile and has_profile(m)

    async def produce():
        try:
            data = await point_series(m, r, lat, lon, hours, profile, job)
        except FieldUnavailable as e:
            raise HTTPException(404, str(e)) from None
        return await asyncio.get_running_loop().run_in_executor(
            POOL, render_meteogram_worker, data, name)

    return await _cached_png(["mg", model, r, round(lat, 3), round(lon, 3), hours, profile, name],
                             produce)


def _model_list(models: str):
    ms = [_model(x) for x in dict.fromkeys(models.split(",")) if x]
    if not ms:
        raise HTTPException(400, "indique pelo menos um modelo")
    return ms


@app.get("/api/compare/meteogram.png")
async def compare_meteogram(models: str, lat: float, lon: float, hours: int = 240,
                            name: str = "", job: str | None = None):
    _check_point(lat, lon)
    ms = _model_list(models)
    runs_ = [(await m.list_runs())[0] for m in ms]

    async def produce():
        datas = []
        for i, (m, r) in enumerate(zip(ms, runs_)):
            if job:
                PROGRESS.setdefault(job, {}).update(model_i=i, models=len(ms))
            try:
                datas.append(await point_series(m, r, lat, lon, hours, False, job))
            except FieldUnavailable:
                continue
        if not datas:
            raise HTTPException(404, "sem dados para nenhum modelo")
        return await asyncio.get_running_loop().run_in_executor(
            POOL, render_multi_worker, datas, name)

    return await _cached_png(["cmg", models, runs_, round(lat, 3), round(lon, 3), hours, name],
                             produce)


# ---------------------------------------------------------------- comparison
@app.get("/api/compare/times")
async def compare_times(models: str):
    """Valid times available in every selected model (newest run containing each)."""
    ms = _model_list(models)
    per: dict[str, dict] = {}
    for m in ms:
        table = {}
        for r in (await m.list_runs())[:2]:
            for st in await m.list_steps(r):
                v = r + dt.timedelta(hours=st)
                table.setdefault(v, {"run": r.strftime("%Y%m%d%H"), "step": st})
        per[m.id] = table
    common = sorted(set.intersection(*(set(t) for t in per.values())))
    now = dt.datetime.utcnow() - dt.timedelta(hours=3)
    common = [v for v in common if v >= now] or common
    return [{"valid": v.strftime("%Y%m%d%H"), "label": fmt_time(v),
             "models": {mid: per[mid][v] for mid in per}} for v in common]


async def _resolve(m, valid: dt.datetime):
    rs = await resolve_run(m, valid)
    if rs is None:
        raise HTTPException(404, f"{m.name}: sem previsão válida para {fmt_time(valid)}")
    return rs


@app.get("/api/compare/sounding.png")
async def compare_sounding(models: str, valid: str, lat: float, lon: float, name: str = ""):
    from .sounding.profile import get_profile
    _check_point(lat, lon)
    v = _parse_run(valid)
    items = []
    for m in _model_list(models):
        r, st = await _resolve(m, v)
        try:
            snd = await get_profile(m, r, st, lat, lon)
        except FieldUnavailable:
            continue
        items.append({"model": m.id, "name": m.name, "short": m.name.replace("ECMWF ", ""),
                      "run": r, "step": st, "snd": snd})
    if not items:
        raise HTTPException(404, "sem dados")
    ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
    title = (f"Sondagens comparadas   {name + ' · ' if name else ''}"
             f"{abs(lat):.2f}°{ns} {abs(lon):.2f}°{ew}")
    subtitle = f"Válido {fmt_time(v)}   ·   cada modelo na run mais recente que cobre esta hora"
    png = await asyncio.get_running_loop().run_in_executor(
        POOL, compare_sounding_worker, items, title, subtitle)
    return Response(png, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/compare/diff.png")
async def compare_diff(a: str, b: str, var: str, valid: str, region: str = "europe",
                       level: int = 0, accum: int = 6):
    from .maps.diff import build_diff_job
    if region not in REGIONS:
        raise HTTPException(404, "região desconhecida")
    ma, mb = _model(a), _model(b)
    v = _parse_run(valid)
    ra, sa = await _resolve(ma, v)
    rb, sb = await _resolve(mb, v)
    key = hashlib.sha1(json.dumps([RENDER_VERSION, "diff", a, b, var, level, accum, region,
                                   ra, sa, rb, sb], default=str).encode()).hexdigest()
    png_p, meta_p = IMG_CACHE / f"{key}.png", IMG_CACHE / f"{key}.json"
    if not (png_p.exists() and meta_p.exists()):
        try:
            job = await build_diff_job((ma, ra, sa), (mb, rb, sb), var, level, region, accum)
        except FieldUnavailable as e:
            raise HTTPException(404, str(e)) from None
        png, rect = await asyncio.get_running_loop().run_in_executor(POOL, render_map_worker, job)
        png_p.write_bytes(png)
        meta_p.write_text(json.dumps({"axes": rect}))
    meta = json.loads(meta_p.read_text())
    return FileResponse(png_p, media_type="image/png", headers={
        "X-Map-Axes": ",".join(f"{x:.6f}" for x in meta["axes"]),
        "Access-Control-Expose-Headers": "X-Map-Axes"})


# ---------------------------------------------------------------- heat / cold waves
@app.get("/api/climate/status")
async def climate_status():
    from . import climate
    return {**climate.STATE, "ready": climate.ready()}


@app.post("/api/climate/download")
async def climate_download():
    from . import climate
    if not climate.ready():
        asyncio.create_task(climate.ensure())
    return {**climate.STATE, "ready": climate.ready()}


async def _climate_or_503():
    from . import climate
    if not climate.ready():
        asyncio.create_task(climate.ensure())
        st = climate.STATE
        pct = f" ({100 * st['done'] / st['total']:.0f}%)" if st.get("total") else ""
        raise HTTPException(503, "A descarregar a climatologia CPC 1991–2020 — só acontece uma vez "
                                 f"(~550 MB, ~1 min){pct}. Tente novamente dentro de instantes.")


async def _map_png_cached(key_parts, build):
    key = hashlib.sha1(json.dumps([RENDER_VERSION] + key_parts, default=str).encode()).hexdigest()
    png_p, meta_p = IMG_CACHE / f"{key}.png", IMG_CACHE / f"{key}.json"
    if not (png_p.exists() and meta_p.exists()):
        try:
            job = await build()
        except FieldUnavailable as e:
            raise HTTPException(404, str(e)) from None
        png, rect = await asyncio.get_running_loop().run_in_executor(POOL, render_map_worker, job)
        png_p.write_bytes(png)
        meta_p.write_text(json.dumps({"axes": rect}))
    meta = json.loads(meta_p.read_text())
    return FileResponse(png_p, media_type="image/png", headers={
        "X-Map-Axes": ",".join(f"{x:.6f}" for x in meta["axes"]),
        "Access-Control-Expose-Headers": "X-Map-Axes"})


@app.get("/api/extremes/map.png")
async def extremes_map(kind: str, model: str, region: str = "iberia", d0: int = 1, d1: int = 16,
                       job: str | None = None):
    from .heatwave import build_map_job
    if region not in REGIONS:
        raise HTTPException(404, "região desconhecida")
    await _climate_or_503()
    m = _model(model)
    r = (await m.list_runs())[0]
    prog = PROGRESS.setdefault(job, {}) if job else None
    return await _map_png_cached(["xmap", kind, model, r, region, d0, d1],
                                 lambda: build_map_job(kind, m, r, region, d0, d1, prog))


@app.get("/api/extremes/ens/steps")
async def extremes_ens_steps():
    from .models.ecmwf import ENS
    r = (await ENS.list_runs())[0]
    return {"run": r.strftime("%Y%m%d%H"), "label": fmt_time(r), "steps": await ENS.list_steps(r)}


@app.get("/api/extremes/ens.png")
async def extremes_ens(param: str = "p_gt1p5", region: str = "europe", step: int | None = None,
                       d0: int = 1, d1: int = 15):
    from .heatwave import ENS_LABEL, build_ens_job
    from .models.ecmwf import ENS
    if param not in ENS_LABEL:
        raise HTTPException(400, "parâmetro inválido")
    if region not in REGIONS:
        raise HTTPException(404, "região desconhecida")
    r = (await ENS.list_runs())[0]
    return await _map_png_cached(["xens", param, region, r, step, d0, d1],
                                 lambda: build_ens_job(param, region, r, step, d0, d1))


@app.get("/api/extremes/point.png")
async def extremes_point(lat: float, lon: float, models: str = "gfs,ecmwf,aifs", name: str = "",
                         job: str | None = None):
    from .heatwave import point_data
    from .models.ecmwf import ENS
    _check_point(lat, lon)
    await _climate_or_503()
    ms = _model_list(models)
    runs_ = [(await m.list_runs())[0] for m in ms] + [(await ENS.list_runs())[0]]

    async def produce():
        prog = PROGRESS.setdefault(job, {}) if job else None
        data = await point_data(ms, lat, lon, prog)
        if not data["models"]:
            raise HTTPException(404, "sem dados")
        return await asyncio.get_running_loop().run_in_executor(
            POOL, render_thermogram_worker, data, name)

    return await _cached_png(["thermo", models, runs_, round(lat, 3), round(lon, 3), name], produce)


# ---------------------------------------------------------------- satellite / radar
@app.get("/api/live/layers")
async def live_layers():
    from .live import layers
    try:
        return await layers()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"EUMETView indisponível: {e}") from None


# ---------------------------------------------------------------- front-end
@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
