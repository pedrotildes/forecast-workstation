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
                      render_case_worker, render_multi_worker, render_sounding_worker,
                      render_thermogram_worker, render_xsection_worker,
                      render_xsection_multi_worker)

app = FastAPI(title="Forecast Workstation", version=VERSION)
POOL: ProcessPoolExecutor | None = None
RENDER_VERSION = "15"


@app.middleware("http")
async def _no_stale_frontend(request, call_next):
    # front-end files must be revalidated (ETag → 304) so an updated app never runs old JS
    resp = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.on_event("startup")
async def _startup():
    global POOL
    import multiprocessing as mp
    POOL = ProcessPoolExecutor(max_workers=3, mp_context=mp.get_context("spawn"))
    asyncio.create_task(_housekeeping())
    from .archive import archiver_loop
    asyncio.create_task(archiver_loop())


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
    from .archive import archived_runs
    arch = set(archived_runs(m.id))
    out = []
    for i, r in enumerate(rs):
        steps = await m.list_steps(r) if i < 2 else None
        label = fmt_time(r) + (" · arquivo" if r in arch else "")
        out.append({"run": r.strftime("%Y%m%d%H"), "label": label, "steps": steps})
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


# ---------------------------------------------------------------- vertical cross-sections
def _xs_meta(kind, title, subtitle, source, a, b):
    return {"kind": kind, "title": title, "subtitle": subtitle, "source": source,
            "a_name": a, "b_name": b}


def _xs_check(kind, lat0, lon0, lat1, lon1):
    from .xsection import KINDS, check_line
    if kind not in KINDS:
        raise HTTPException(400, "tipo de corte desconhecido")
    try:
        return check_line(lat0, lon0, lat1, lon1)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None


async def _xs_render(sec, meta):
    if meta["kind"] == "omega" and sec.get("w") is None:
        raise HTTPException(404, "este modelo não fornece velocidade vertical (ω)")
    try:
        return await asyncio.get_running_loop().run_in_executor(POOL, render_xsection_worker, sec, meta)
    except ValueError as e:
        raise HTTPException(404, str(e)) from None


@app.get("/api/xsection/kinds")
async def xsection_kinds():
    from .xsection import KINDS
    return [{"id": k, "name": v} for k, v in KINDS.items()]


@app.get("/api/xsection.png")
async def xsection_png(model: str, run: str, step: int, lat0: float, lon0: float, lat1: float,
                       lon1: float, kind: str = "dyn", a: str = "", b: str = ""):
    """Vertical cross-section A→B of a forecast (model, run, step)."""
    from .xsection import KINDS, section
    m = _model(model)
    r = _parse_run(run)
    _xs_check(kind, lat0, lon0, lat1, lon1)
    if not (m.covers(lat0, lon0) and m.covers(lat1, lon1)):
        raise HTTPException(404, m.outside_msg())

    async def produce():
        if step not in await m.list_steps(r):
            raise HTTPException(404, f"passo +{step}h não disponível para {m.name} {run}")
        try:
            sec = await section(m, r, step, lat0, lon0, lat1, lon1)
        except FieldUnavailable as e:
            raise HTTPException(404, str(e)) from None
        valid = r + dt.timedelta(hours=step)
        title = f"Corte vertical {a or 'A'} → {b or 'B'} — {m.name} {m.resolution}"
        sub = (f"{KINDS[kind]}   ·   Run {fmt_time(r)}   ·   Válido {fmt_time(valid)} (+{step} h)")
        return await _xs_render(sec, _xs_meta(kind, title, sub, m.source, a, b))
    return await _cached_png(["xs", model, run, step, round(lat0, 3), round(lon0, 3), round(lat1, 3),
                              round(lon1, 3), kind, a, b], produce)


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
    runs_ = [await m.best_run() for m in ms]

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


@app.get("/api/compare/xsection.png")
async def compare_xsection(models: str, valid: str, lat0: float, lon0: float, lat1: float, lon1: float,
                           kind: str = "dyn", a: str = "", b: str = "", diff: bool = False):
    """Vertical cross-section A→B of several models at the same valid time (stacked panels,
    common colour scale; optional difference panel first − second model)."""
    from .xsection import KINDS, section
    _xs_check(kind, lat0, lon0, lat1, lon1)
    v = _parse_run(valid)
    ms = _model_list(models)
    plan, skipped = [], []
    for m in ms:
        if not (m.covers(lat0, lon0) and m.covers(lat1, lon1)):
            skipped.append(f"{m.name} (linha fora do domínio)")
            continue
        rs = await resolve_run(m, v)
        if rs is None:
            skipped.append(f"{m.name} (sem previsão para esta hora)")
            continue
        plan.append((m, *rs))
    if not plan:
        raise HTTPException(404, "nenhum modelo cobre esta linha e esta hora")

    async def produce():
        async def one(m, r, st):
            try:
                return m, r, st, await section(m, r, st, lat0, lon0, lat1, lon1)
            except FieldUnavailable:
                return m, r, st, None
        items, miss = [], list(skipped)
        for m, r, st, sec in await asyncio.gather(*(one(*p) for p in plan)):
            if sec is None:
                miss.append(f"{m.name} (sem dados)")
                continue
            if kind == "omega" and sec.get("w") is None:
                miss.append(f"{m.name} (sem ω)")
                continue
            items.append({"model": m.id, "name": m.name.replace("ECMWF ", ""),
                          "label": f"{m.name} — run {r:%d/%m %H}Z, +{st} h", "sec": sec})
        if not items:
            raise HTTPException(404, "sem dados para o corte: " + ", ".join(miss))
        title = f"Corte vertical {a or 'A'} → {b or 'B'} — comparação de modelos"
        sub = (f"{KINDS[kind]}   ·   Válido {fmt_time(v)}   ·   cada modelo na run mais recente que cobre esta hora"
               + (f"   ·   sem: {', '.join(miss)}" if miss else ""))
        meta = _xs_meta(kind, title, sub, " · ".join(dict.fromkeys(MODELS[it["model"]].source for it in items)), a, b)
        meta["diff"] = diff
        try:
            return await asyncio.get_running_loop().run_in_executor(POOL, render_xsection_multi_worker, items, meta)
        except ValueError as e:
            raise HTTPException(404, str(e)) from None
    key = ["cxs", valid, [(m.id, r.isoformat(), st) for m, r, st in plan], round(lat0, 3), round(lon0, 3),
           round(lat1, 3), round(lon1, 3), kind, a, b, diff]
    return await _cached_png(key, produce)


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
    r = await m.best_run()
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
async def extremes_point(lat: float, lon: float, models: str = "gfs,ecmwf,aifs,icon_eu", name: str = "",
                         job: str | None = None):
    from .heatwave import point_data
    from .models.ecmwf import ENS
    _check_point(lat, lon)
    await _climate_or_503()
    ms = _model_list(models)
    runs_ = [await m.best_run() for m in ms] + [(await ENS.list_runs())[0]]

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


# ---------------------------------------------------------------- events (archive)
def _event_view(ev: dict) -> dict:
    from . import archive
    return {**ev, "radar_frames": len(archive.radar_frames(ev["id"])),
            "start_label": fmt_time(archive.parse_t(ev["start"])),
            "end_label": fmt_time(archive.parse_t(ev["end"]))}


@app.get("/api/events")
async def events_list():
    from . import archive
    return [_event_view(e) for e in sorted(archive.load_events(), key=lambda e: e["start"])]


@app.post("/api/events")
async def events_create(body: dict):
    from . import archive
    try:
        t0, t1 = archive.parse_t(body["start"]), archive.parse_t(body["end"])
    except (KeyError, ValueError):
        raise HTTPException(400, "datas inválidas (use AAAA-MM-DDTHH:MM, UTC)") from None
    if t1 <= t0:
        raise HTTPException(400, "o fim tem de ser depois do início")
    if (t1 - t0).days > 14:
        raise HTTPException(400, "um evento pode ter no máximo 14 dias")
    models = [m for m in body.get("models", []) if m in MODELS] or ["gfs", "ecmwf"]
    region = body.get("region", "nea")
    if region not in REGIONS:
        raise HTTPException(400, "região desconhecida")
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "indique um nome")
    ev = archive.create_event(name, t0.isoformat(), t1.isoformat(), models, region,
                              float(body.get("lead_days", 2)), body.get("notes", ""))
    asyncio.create_task(archive.archive_event(ev["id"]))
    return _event_view(ev)


@app.post("/api/events/{eid}/archive")
async def events_archive_now(eid: str):
    from . import archive
    if not archive.get_event(eid):
        raise HTTPException(404, "evento desconhecido")
    asyncio.create_task(archive.archive_event(eid))
    if archive.parse_t(archive.get_event(eid)["end"]) >= dt.datetime.utcnow():
        asyncio.create_task(archive.capture_radar(eid))
    return {"ok": True}


@app.delete("/api/events/{eid}")
async def events_delete(eid: str):
    from . import archive
    if not archive.get_event(eid):
        raise HTTPException(404, "evento desconhecido")
    await asyncio.to_thread(archive.delete_event, eid, True)
    return {"ok": True}


@app.get("/api/events/{eid}/radar")
async def events_radar(eid: str):
    from . import archive
    return archive.radar_frames(eid)


@app.get("/api/archive/radar/{t}/{z}/{x}/{y}.png")
async def archive_radar_tile(t: int, z: int, x: int, y: int):
    from .archive import RADAR_DIR
    p = RADAR_DIR / str(t) / f"{z}_{x}_{y}.png"
    if not p.exists():
        raise HTTPException(404, "sem mosaico")
    return FileResponse(p, media_type="image/png", headers={"Cache-Control": "max-age=86400"})


# ---------------------------------------------------------------- case studies
def _case_event(eid: str) -> dict:
    from . import archive
    ev = archive.get_event(eid)
    if ev is None:
        raise HTTPException(404, "evento desconhecido")
    return ev


def _case_key(ev, *parts):
    # images are cached until the archive changes (event "updated" stamp)
    return ["case", ev["id"], ev.get("updated"), ev.get("systems"), *parts]


@app.get("/api/cases/{eid}/systems")
async def case_systems(eid: str):
    from .cases import systems
    return systems(_case_event(eid))


@app.put("/api/cases/{eid}/systems")
async def case_systems_set(eid: str, body: list[dict]):
    from . import archive
    _case_event(eid)
    clean = []
    for s in body:
        try:
            clean.append({"name": str(s["name"]).strip()[:40], "lat": float(s["lat"]),
                          "lon": float(s["lon"]), "time": archive.parse_t(s["time"]).isoformat()})
        except (KeyError, ValueError, TypeError):
            raise HTTPException(400, "sistema inválido (nome, lat, lon, hora UTC)") from None
    archive.update_event(eid, systems=clean)
    return clean


@app.get("/api/cases/{eid}/tracks.json")
async def case_tracks(eid: str, models: str = "gfs,ecmwf,aifs,icon_eu"):
    from .cases import all_tracks
    ev = _case_event(eid)
    return _clean(await all_tracks(ev, [m for m in models.split(",") if m in MODELS]))


@app.get("/api/cases/{eid}/track.png")
async def case_track_png(eid: str, models: str = "gfs,ecmwf,aifs,icon_eu", fc: str = "",
                         region: str = "nea"):
    from .cases import build_track_job
    ev = _case_event(eid)
    ms = [m for m in models.split(",") if m in MODELS] or ["ecmwf"]
    if region not in REGIONS:
        raise HTTPException(404, "região desconhecida")

    async def build():
        job, _ = await build_track_job(ev, ms, fc or None, region)
        return job
    return await _map_png_cached(_case_key(ev, "track", ms, fc, region), build)


@app.get("/api/cases/{eid}/totals.png")
async def case_totals_png(eid: str, model: str = "ecmwf", kind: str = "precip", region: str = "nea"):
    from .cases import build_totals_job
    ev = _case_event(eid)
    if model not in MODELS or kind not in ("precip", "gust") or region not in REGIONS:
        raise HTTPException(400, "parâmetros inválidos")
    return await _map_png_cached(_case_key(ev, "totals", model, kind, region),
                                 lambda: build_totals_job(ev, model, kind, region))


@app.get("/api/cases/{eid}/pressure.png")
async def case_pressure_png(eid: str, models: str = "gfs,ecmwf,aifs,icon_eu", fc: str = "ecmwf"):
    from .cases import all_tracks, forecast_runs, select_forecasts
    ev = _case_event(eid)
    ms = [m for m in models.split(",") if m in MODELS] or ["ecmwf"]

    async def produce():
        tracks = await all_tracks(ev, ms)
        fcs = {}
        if fc in MODELS:
            for name, per in tracks.items():
                ref = per.get(fc) or next((v for v in per.values() if v), [])
                sel = select_forecasts(await forecast_runs(MODELS[fc], ev, ref), ref)
                fcs[name] = {r: tr for r, (tr, lead) in sel.items()}
        data = {"event": ev["name"], "tracks": tracks, "forecasts": fcs, "fc_model": fc}
        try:
            return await asyncio.get_running_loop().run_in_executor(
                POOL, render_case_worker, "pressure", data)
        except ValueError as e:
            raise HTTPException(404, str(e)) from None
    return await _cached_png(_case_key(ev, "pressure", ms, fc), produce)


@app.get("/api/cases/{eid}/sequence.png")
async def case_sequence_png(eid: str, model: str = "ecmwf", every: int = 12, region: str = "nea"):
    from .cases import sequence_data
    ev = _case_event(eid)
    if model not in MODELS or region not in REGIONS or every not in (6, 12, 24):
        raise HTTPException(400, "parâmetros inválidos")

    async def produce():
        data = await sequence_data(ev, model, every, region)
        if not data["panels"]:
            raise HTTPException(404, "sem análises arquivadas para a sequência")
        return await asyncio.get_running_loop().run_in_executor(
            POOL, render_case_worker, "sequence", data)
    return await _cached_png(_case_key(ev, "sequence", model, every, region), produce)


@app.get("/api/cases/{eid}/point.png")
async def case_point_png(eid: str, lat: float, lon: float, name: str = "",
                         models: str = "gfs,ecmwf,aifs,icon_eu"):
    from .cases import point_chain
    ev = _case_event(eid)
    _check_point(lat, lon)
    ms = [m for m in models.split(",") if m in MODELS]

    async def produce():
        series = {}
        for mid in ms:
            try:
                series[mid] = await point_chain(MODELS[mid], ev, lat, lon)
            except FieldUnavailable:
                continue
        series = {k: v for k, v in series.items() if v["time"]}
        if not series:
            raise HTTPException(404, "sem dados arquivados neste ponto")
        data = {"event": ev["name"], "place": name, "lat": lat, "lon": lon, "series": series}
        return await asyncio.get_running_loop().run_in_executor(POOL, render_case_worker, "point", data)
    return await _cached_png(_case_key(ev, "point", round(lat, 3), round(lon, 3), name, ms), produce)


@app.get("/api/cases/{eid}/obs")
async def case_obs(eid: str):
    """Stations, the event's 00/12Z times and the observed soundings already archived."""
    from . import obs
    ev = _case_event(eid)
    return {"stations": [obs.station_info(s[0]) for s in obs.STATIONS],
            "times": [t.isoformat() for t in obs.event_obs_times(ev)],
            "archived": obs.archived_obs(ev)}


@app.get("/api/cases/{eid}/sounding.png")
async def case_sounding_png(eid: str, time: str, station: str = "", lat: float | None = None,
                            lon: float | None = None, name: str = "",
                            models: str = "gfs,ecmwf,aifs,icon_eu"):
    from . import obs
    from .archive import parse_t
    from .cases import case_sounding_items
    from .maps.compose import fmt_time
    ev = _case_event(eid)
    t = parse_t(time)
    if station:
        if station not in obs.STATION:
            raise HTTPException(404, "estação desconhecida")
        si = obs.station_info(station)
        lat, lon, name = si["lat"], si["lon"], si["name"]
    if lat is None or lon is None:
        raise HTTPException(400, "indique uma estação ou um ponto")
    _check_point(lat, lon)
    ms = [m for m in models.split(",") if m in MODELS]

    async def produce():
        items = await case_sounding_items(ev, t, lat, lon, station or None, ms)
        if not items:
            raise HTTPException(404, "sem sondagem observada nem análises arquivadas para esta hora")
        has_obs = any(it.get("obs") for it in items)
        ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
        title = (f"Sondagem — {name + ' · ' if name else ''}{abs(lat):.2f}°{ns} {abs(lon):.2f}°{ew}"
                 f" · evento «{ev['name']}»")
        sub = (f"{fmt_time(t)} · " + ("radiossonda observada (preto) e " if has_obs else
               ("sem radiossonda arquivada para esta hora · " if station else "")) +
               "análises dos modelos (passo +0 h)")
        data = {"items": items, "title": title, "subtitle": sub}
        return await asyncio.get_running_loop().run_in_executor(POOL, render_case_worker, "sounding", data)
    return await _cached_png(_case_key(ev, "sounding", time, station, round(lat, 3), round(lon, 3), ms,
                                       obs._path(station, t).exists() if station else None), produce)


@app.get("/api/cases/{eid}/meteogram.png")
async def case_meteogram_png(eid: str, lat: float, lon: float, model: str = "ecmwf", name: str = ""):
    from .cases import case_meteogram_data
    ev = _case_event(eid)
    _check_point(lat, lon)
    if model not in MODELS:
        raise HTTPException(400, "modelo desconhecido")

    async def produce():
        try:
            data = await case_meteogram_data(ev, model, lat, lon, name)
        except FieldUnavailable as e:
            raise HTTPException(404, str(e)) from None
        try:
            return await asyncio.get_running_loop().run_in_executor(POOL, render_case_worker, "meteogram", data)
        except ValueError as e:
            raise HTTPException(404, str(e)) from None
    return await _cached_png(_case_key(ev, "meteogram", model, round(lat, 3), round(lon, 3), name), produce)


@app.get("/api/cases/{eid}/xtimes")
async def case_xtimes(eid: str, model: str = "ecmwf"):
    """6-hourly analysis times of the event; ``archived`` when the run's step 0 is archived."""
    from . import archive
    ev = _case_event(eid)
    if model not in MODELS:
        raise HTTPException(400, "modelo desconhecido")
    t0, t1 = archive.parse_t(ev["start"]), min(archive.parse_t(ev["end"]), archive._utcnow())
    t = t0.replace(hour=(t0.hour // 6) * 6, minute=0, second=0, microsecond=0)
    if t < t0:
        t += dt.timedelta(hours=6)
    out = []
    while t <= t1:
        out.append({"time": t.isoformat(), "archived": 0 in archive.archived_steps(model, t)})
        t += dt.timedelta(hours=6)
    return out


@app.get("/api/cases/{eid}/xsection.png")
async def case_xsection_png(eid: str, time: str, lat0: float, lon0: float, lat1: float, lon1: float,
                            model: str = "ecmwf", kind: str = "dyn", a: str = "", b: str = ""):
    """Vertical cross-section A→B of the archived analysis (step 0 of the run at `time`)."""
    from . import archive
    from .archive import parse_t
    from .models.base import Domain
    from .xsection import KINDS, section
    ev = _case_event(eid)
    if model not in MODELS:
        raise HTTPException(400, "modelo desconhecido")
    m = MODELS[model]
    t = parse_t(time)
    _xs_check(kind, lat0, lon0, lat1, lon1)
    if not (m.covers(lat0, lon0) and m.covers(lat1, lon1)):
        raise HTTPException(404, m.outside_msg())
    d = REGIONS[ev.get("region", "nea")].download_domain
    clip = Domain(d.lon0, d.lon1, d.lat0, d.lat1)
    if m.coverage:
        lo0, lo1, la0, la1 = m.coverage
        clip = Domain(max(clip.lon0, lo0), min(clip.lon1, lo1), max(clip.lat0, la0), min(clip.lat1, la1))

    async def produce():
        try:
            sec = await section(m, t, 0, lat0, lon0, lat1, lon1, clip=clip)
        except FieldUnavailable as e:
            if 0 not in archive.archived_steps(model, t):
                raise HTTPException(404, f"A análise {m.name} de {fmt_time(t)} ainda não está no arquivo "
                                         "(run ainda não publicada ou por arquivar).") from None
            raise HTTPException(404, f"{e} (análise {m.name} de {fmt_time(t)})") from None
        title = f"Corte vertical {a or 'A'} → {b or 'B'} — {m.name} · evento «{ev['name']}»"
        sub = f"{KINDS[kind]}   ·   Análise {fmt_time(t)} (passo +0 h, arquivo do evento)"
        return await _xs_render(sec, _xs_meta(kind, title, sub, m.source, a, b))
    return await _cached_png(_case_key(ev, "xsection", model, time, round(lat0, 3), round(lon0, 3),
                                       round(lat1, 3), round(lon1, 3), kind, a, b), produce)


# ---------------------------------------------------------------- front-end
@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
