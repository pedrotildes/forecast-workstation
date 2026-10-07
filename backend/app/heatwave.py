"""Heat-wave / cold-wave guidance.

Deterministic criterion (IPMA / WMO Heat Wave Duration Index): a heat wave occurs when,
for at least 6 consecutive days, Tmax ≥ normal Tmax + 5 °C; a cold wave when, for at
least 6 consecutive days, Tmin ≤ normal Tmin − 5 °C. Normals: CPC 1991–2020.

Probabilistic signal: ECMWF ENS probability of the 850 hPa temperature standardized
anomaly exceeding ±1/1.5/2 σ (relative to the ENS model climate).
"""
from __future__ import annotations

import asyncio
import datetime as dt

import numpy as np

from . import climate
from .daily import daily_fields
from .maps.compose import fmt_time
from .maps.styles import Fill, _interp, rng
from .models import Domain, FieldUnavailable, Model
from .models.ecmwf import ENS
from .regions import REGIONS

THRESH = 5.0
MIN_DAYS = 6
WEEK = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
MON = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]


def fmt_day(d: dt.date) -> str:
    return f"{WEEK[d.weekday()]} {d.day:02d} {MON[d.month - 1]}"


def longest_run(mask: np.ndarray):
    """mask (days, ...) -> (longest run length, start index of it, run open at the end)."""
    cur = np.zeros(mask.shape[1:], dtype=np.int16)
    best = np.zeros_like(cur)
    start = np.zeros_like(cur)
    for i, m in enumerate(mask):
        cur = np.where(m, cur + 1, 0).astype(np.int16)
        better = cur > best
        start = np.where(better, i - cur + 1, start)
        best = np.where(better, cur, best)
    return best, start, cur > 0


async def anomalies(model: Model, run, domain: Domain, progress=None):
    await climate.ensure()
    d = await daily_fields(model, run, domain, progress=progress)
    g = d["grid"]
    out = {"days": d["days"], "grid": g, "tmax": d["tmax"], "tmin": d["tmin"],
           "ntmax": {}, "ntmin": {}}
    for day in d["days"]:
        out["ntmax"][day] = climate.normal_on("tmax", day, g.lats, g.lons)
        out["ntmin"][day] = climate.normal_on("tmin", day, g.lats, g.lons)
    return out


# ------------------------------------------------------------------ map styles
PERSIST_LEVELS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16]
HEAT_FILL = Fill(PERSIST_LEVELS, _interp(["#fff3b0", "#ffd166", "#f8961e", "#f3722c", "#e63946",
                                          "#b5179e", "#7209b7", "#3a0ca3", "#240046", "#10002b"], 10),
                 extend="max", over="#000000", transparent_under=True, tick_every=1)
COLD_FILL = Fill(PERSIST_LEVELS, _interp(["#e0fbfc", "#a9def9", "#72b7f2", "#4895ef", "#4361ee",
                                          "#3f37c9", "#3a0ca3", "#480ca8", "#560bad", "#240046"], 10),
                 extend="max", over="#000000", transparent_under=True, tick_every=1)


def anom_fill(lim=12, step=1):
    lv = rng(-lim, lim, step)
    cols = _interp(["#08306b", "#2171b5", "#6baed6", "#c6dbef", "#f7f7f7", "#f7f7f7",
                    "#fcbba1", "#fb6a4a", "#cb181d", "#67000d"], len(lv) - 1)
    return Fill(lv, cols, under="#041a3d", over="#3d0007", tick_every=2)


PROB_FILL_WARM = Fill([10, 20, 30, 40, 50, 60, 70, 80, 90, 100.01],
                      _interp(["#fff3b0", "#ffd166", "#f8961e", "#e63946", "#9d0208", "#6a040f"], 9),
                      extend="neither", transparent_under=True)
PROB_FILL_COLD = Fill([10, 20, 30, 40, 50, 60, 70, 80, 90, 100.01],
                      _interp(["#e0fbfc", "#a9def9", "#4895ef", "#3f37c9", "#3a0ca3", "#10002b"], 9),
                      extend="neither", transparent_under=True)

ENS_LABEL = {"p_gt1": "T850 > +1σ", "p_gt1p5": "T850 > +1,5σ", "p_gt2": "T850 > +2σ",
             "p_lt1": "T850 < −1σ", "p_lt1p5": "T850 < −1,5σ", "p_lt2": "T850 < −2σ"}


def _job(region_id, grid, layers, left1, left2, layers_txt, right1, right2, footer):
    return {"region": region_id, "lats": grid.lats, "lons": grid.lons, "layers": layers,
            "header": {"left1": left1, "left2": left2, "layers": layers_txt,
                       "right1": right1, "right2": right2}, "footer": footer}


async def build_map_job(kind: str, model: Model, run, region_id: str, d0: int, d1: int,
                        progress=None) -> dict:
    region = REGIONS[region_id]
    if not model.covers_box(*region.extent):
        raise FieldUnavailable(model.outside_msg())
    a = await anomalies(model, run, region.download_domain, progress)
    days = a["days"]
    d0 = max(1, d0)
    d1 = min(len(days), d1)
    if d0 > d1:
        raise FieldUnavailable("intervalo de dias inválido")
    sel = days[d0 - 1:d1]
    src = model.source
    footer = f"Dados: {src} · Normais: NOAA CPC Global Unified Temperature 1991–2020 (só terra)"
    rng_txt = f"{fmt_day(sel[0])} a {fmt_day(sel[-1])} (dias {d0}–{d1} da previsão)"
    left1 = f"{model.name} {model.resolution}   ·   Run {fmt_time(run)}"
    if kind in ("heat", "cold"):
        key, nkey = ("tmax", "ntmax") if kind == "heat" else ("tmin", "ntmin")
        mask = np.stack([(a[key][d] - a[nkey][d]) >= THRESH if kind == "heat"
                         else (a[key][d] - a[nkey][d]) <= -THRESH for d in sel])
        best, _, _ = longest_run(mask)
        land = np.isfinite(a[nkey][sel[0]])
        data = np.where(land, best.astype(np.float32), np.nan)
        what = ("Tmáx ≥ normal + 5 °C" if kind == "heat" else "Tmín ≤ normal − 5 °C")
        title = "Vaga de calor" if kind == "heat" else "Vaga de frio"
        layers = [{"type": "fill", "var": "t2m", "data": data, "label":
                   f"Máximo de dias consecutivos com {what}",
                   "fill_style": HEAT_FILL if kind == "heat" else COLD_FILL},
                  {"type": "contour", "var": "t2m", "data": data, "interval": 1, "color": "#000000",
                   "width": 2.2, "style": "solid", "levels": [MIN_DAYS - 0.5],
                   "labels": False, "extrema": False}]
        txt = (f"Sombreado: dias consecutivos com {what}  ·  linha preta: {MIN_DAYS} dias "
               f"(critério IPMA/OMM de {title.lower()})")
        return _job(region_id, a["grid"], layers, left1, f"{title}: persistência · {rng_txt}", txt,
                    f"{title}", region.name, footer)
    if kind in ("tmax_anom", "tmin_anom"):
        key, nkey = ("tmax", "ntmax") if kind == "tmax_anom" else ("tmin", "ntmin")
        an = np.nanmean(np.stack([a[key][d] - a[nkey][d] for d in sel]), axis=0)
        nm = "Tmáx" if key == "tmax" else "Tmín"
        layers = [{"type": "fill", "var": "t2m", "data": an.astype(np.float32),
                   "label": f"Anomalia média da {nm} face à normal 1991–2020 (°C)",
                   "fill_style": anom_fill()},
                  {"type": "values", "var": "t2m", "data": an.astype(np.float32), "fmt": "%+.0f"}]
        return _job(region_id, a["grid"], layers, left1, f"Anomalia média da {nm} · {rng_txt}",
                    f"Média de {len(sel)} dia(s) da {nm} prevista menos a normal diária CPC 1991–2020",
                    f"Anomalia {nm}", region.name, footer)
    raise FieldUnavailable(f"produto desconhecido: {kind}")


async def build_ens_job(param: str, region_id: str, run=None, step: int | None = None,
                        d0: int | None = None, d1: int | None = None) -> dict:
    region = REGIONS[region_id]
    if run is None:
        run = (await ENS.list_runs())[0]
    steps = await ENS.list_steps(run)
    if step is not None:
        if step not in steps:
            raise FieldUnavailable(f"ENS: passo +{step}h indisponível")
        use = [step]
    else:
        lo, hi = (d0 or 1) * 24 - 12, (d1 or 15) * 24
        use = [s for s in steps if lo <= s <= hi] or steps
    sem = asyncio.Semaphore(4)

    async def one(s):
        async with sem:
            return await ENS.fetch(run, s, [(param, 0)], region.download_domain)

    res = await asyncio.gather(*(one(s) for s in use))
    fields = [r[(param, 0)] for r in res if (param, 0) in r]
    if not fields:
        raise FieldUnavailable("ENS: sem dados")
    grid = fields[0]
    warm = "gt" in param
    if step is not None:
        data = fields[0].values
        valid = run + dt.timedelta(hours=step)
        l2 = f"Probabilidade de {ENS_LABEL[param]} (51 membros)  ·  +{step} h"
        r1 = f"Válido: {fmt_time(valid)}"
        layers = [{"type": "fill", "var": "t2m", "data": data.astype(np.float32),
                   "label": f"Probabilidade de {ENS_LABEL[param]} (%)",
                   "fill_style": PROB_FILL_WARM if warm else PROB_FILL_COLD},
                  {"type": "contour", "var": "t2m", "data": data.astype(np.float32),
                   "color": "#222222", "width": 0.8, "levels": [50, 75], "labels": True,
                   "fmt": "%d%%", "extrema": False}]
        txt = ("Anomalia padronizada da temperatura a 850 hPa face ao clima do modelo "
               "(reforecasts); isolinhas 50 % e 75 %")
    else:
        # persistence of the signal: days (12-h steps / 2) with probability >= 50 %
        data = (np.stack([f.values for f in fields]) >= 50).sum(axis=0) * 0.5
        data = np.where(data > 0, data, np.nan).astype(np.float32)
        v0 = run + dt.timedelta(hours=use[0])
        v1 = run + dt.timedelta(hours=use[-1])
        l2 = f"Persistência do sinal {ENS_LABEL[param]}: {fmt_time(v0)} → {fmt_time(v1)}"
        r1 = "Vaga de calor (ENS)" if warm else "Vaga de frio (ENS)"
        layers = [{"type": "fill", "var": "t2m", "data": data,
                   "label": f"Dias com P({ENS_LABEL[param]}) ≥ 50 %",
                   "fill_style": Fill([0.5, 1, 2, 3, 4, 5, 6, 8, 10, 15],
                                      HEAT_FILL.colors[:9] if warm else COLD_FILL.colors[:9],
                                      extend="neither", transparent_under=True)},
                  {"type": "contour", "var": "t2m", "data": np.nan_to_num(data), "color": "#000000",
                   "width": 2.0, "levels": [5.75], "labels": False, "extrema": False}]
        txt = (f"Número de dias (em passos de 12 h) em que ≥ 50 % dos 51 membros prevêem {ENS_LABEL[param]}"
               "  ·  linha preta: ≥ 6 dias")
    return _job(region_id, grid, layers, f"ECMWF ENS 0.25°   ·   Run {fmt_time(run)}", l2,
                txt, r1, region.name, "Dados: © ECMWF Open Data (CC BY 4.0)")


# ------------------------------------------------------------------ point product
async def point_data(models: list[Model], lat: float, lon: float, job_prog: dict | None = None,
                     ens_params=("p_gt1", "p_gt2", "p_lt1", "p_lt2")) -> dict:
    await climate.ensure()
    dom = Domain.tile(lat, lon)
    out = {"lat": lat, "lon": lon, "models": []}
    for i, m in enumerate(models):
        if not m.covers(lat, lon):
            continue
        run = await m.best_run()
        if job_prog is not None:
            job_prog.update(model_i=i, models=len(models) + 1)
        try:
            d = await daily_fields(m, run, dom, progress=job_prog)
        except FieldUnavailable:
            continue
        g = d["grid"]
        from .grib import Field
        at = lambda a: Field(g.lats, g.lons, a).at(lat, lon)  # noqa: E731
        out["models"].append({
            "id": m.id, "name": m.name, "run": run, "days": d["days"],
            "tmax": [at(d["tmax"][x]) for x in d["days"]],
            "tmin": [at(d["tmin"][x]) for x in d["days"]],
            "inst": not m.has("tmax"),
        })
    days = sorted({x for mm in out["models"] for x in mm["days"]})
    out["days"] = days
    out["ntmax"] = [climate.normal_at("tmax", x, lat, lon) for x in days]
    out["ntmin"] = [climate.normal_at("tmin", x, lat, lon) for x in days]
    # ENS probabilities
    try:
        run = (await ENS.list_runs())[0]
        steps = await ENS.list_steps(run)
        if job_prog is not None:
            job_prog.update(model_i=len(models), done=0, total=len(steps), model="ECMWF ENS")
        sem = asyncio.Semaphore(4)
        vals = {}

        async def one(s):
            async with sem:
                f = await ENS.fetch(run, s, [(p, 0) for p in ens_params], dom)
                vals[s] = {p: f[(p, 0)].at(lat, lon) for p in ens_params if (p, 0) in f}
                if job_prog is not None:
                    job_prog["done"] += 1

        await asyncio.gather(*(one(s) for s in steps))
        out["ens"] = {"run": run, "valid": [run + dt.timedelta(hours=s) for s in steps],
                      **{p: [vals[s].get(p, np.nan) for s in steps] for p in ens_params}}
    except Exception:  # noqa: BLE001
        out["ens"] = None
    return out


def verdict(days, values, normals, warm: bool):
    """Longest run meeting the criterion -> (length, first day, last day, open at end)."""
    best, start, cur = 0, None, 0
    for i, (v, n) in enumerate(zip(values, normals)):
        ok = np.isfinite(v) and np.isfinite(n) and ((v - n) >= THRESH if warm else (v - n) <= -THRESH)
        cur = cur + 1 if ok else 0
        if cur > best:
            best, start = cur, i - cur + 1
    if not best:
        return None
    return {"n": best, "d0": days[start], "d1": days[start + best - 1],
            "open": start + best == len(days)}
