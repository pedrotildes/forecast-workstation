"""Turn a map request (model/run/step/region + layers) into a render job."""
from __future__ import annotations

import datetime as dt

import numpy as np

from ..models import FieldUnavailable, Model
from ..products import VAR_BY_ID, Ctx, Data, evaluate, gather, wind_needs
from ..regions import REGIONS

WEEKDAYS = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
MONTHS = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]


def fmt_time(t: dt.datetime) -> str:
    return f"{WEEKDAYS[t.weekday()]} {t.day:02d} {MONTHS[t.month - 1]} {t.year} {t:%H}Z"


def level_label(level: int) -> str:
    return f"{level} hPa" if level else ""


# Presets: classic operational chart combinations.
PRESETS: dict[str, dict] = {
    "sfc_precip": {"name": "Pressão MSL + precipitação + vento 10 m", "layers": [
        {"type": "fill", "var": "precip", "accum": 3},
        {"type": "contour", "var": "msl"},
        {"type": "wind", "level": 0, "style": "barbs"}]},
    "z500_t500": {"name": "500 hPa: geopotencial + temperatura + vento", "layers": [
        {"type": "fill", "var": "t", "level": 500},
        {"type": "contour", "var": "gh", "level": 500},
        {"type": "wind", "level": 500, "style": "barbs"}]},
    "z500_msl": {"name": "Geopotencial 500 hPa + pressão MSL", "layers": [
        {"type": "fill", "var": "gh", "level": 500},
        {"type": "contour", "var": "msl", "color": "#ffffff", "width": 0.9}]},
    "t850": {"name": "850 hPa: temperatura + geopotencial + vento", "layers": [
        {"type": "fill", "var": "t", "level": 850},
        {"type": "contour", "var": "gh", "level": 850},
        {"type": "wind", "level": 850, "style": "barbs"}]},
    "thetae850": {"name": "850 hPa: θe + vento", "layers": [
        {"type": "fill", "var": "thetae", "level": 850},
        {"type": "contour", "var": "gh", "level": 850, "color": "#333333"},
        {"type": "wind", "level": 850, "style": "barbs"}]},
    "jet250": {"name": "Corrente de jato 250 hPa", "layers": [
        {"type": "fill", "var": "wspd", "level": 250},
        {"type": "contour", "var": "gh", "level": 250},
        {"type": "wind", "level": 250, "style": "streamlines", "color": "#333333"}]},
    "rh700": {"name": "700 hPa: humidade relativa + ω + vento", "layers": [
        {"type": "fill", "var": "rh", "level": 700},
        {"type": "contour", "var": "gh", "level": 700},
        {"type": "wind", "level": 700, "style": "barbs"}]},
    "thick": {"name": "Espessura 1000–500 hPa + pressão MSL", "layers": [
        {"type": "fill", "var": "thick"},
        {"type": "contour", "var": "msl"}]},
    "t2m": {"name": "Temperatura a 2 m + pressão MSL", "layers": [
        {"type": "fill", "var": "t2m"},
        {"type": "contour", "var": "msl", "color": "#222222"}]},
    "wind10": {"name": "Vento a 10 m (intensidade e direção)", "layers": [
        {"type": "fill", "var": "wind10"},
        {"type": "contour", "var": "msl", "color": "#333333"},
        {"type": "wind", "level": 0, "style": "barbs"}]},
    "cape": {"name": "Convecção: CAPE + cisalhamento + vento 10 m", "layers": [
        {"type": "fill", "var": "mucape"},
        {"type": "contour", "var": "shear"},
        {"type": "wind", "level": 500, "style": "barbs", "color": "#5a189a"}]},
    "vort500": {"name": "500 hPa: vorticidade relativa + geopotencial", "layers": [
        {"type": "fill", "var": "vort", "level": 500},
        {"type": "contour", "var": "gh", "level": 500},
        {"type": "wind", "level": 500, "style": "barbs"}]},
    "tadv850": {"name": "850 hPa: advecção de temperatura", "layers": [
        {"type": "fill", "var": "tadv", "level": 850},
        {"type": "contour", "var": "t", "level": 850},
        {"type": "wind", "level": 850, "style": "barbs"}]},
    "t2m_values": {"name": "Temperatura a 2 m com valores nas cidades", "layers": [
        {"type": "fill", "var": "t2m"},
        {"type": "contour", "var": "msl", "color": "#222222"},
        {"type": "values", "var": "t2m"}]},
    "tmax_day": {"name": "Temperatura máxima do dia", "layers": [
        {"type": "fill", "var": "tmax_day"},
        {"type": "values", "var": "tmax_day"}]},
    "tmin_day": {"name": "Temperatura mínima do dia", "layers": [
        {"type": "fill", "var": "tmin_day"},
        {"type": "values", "var": "tmin_day"}]},
    "tmax_anom": {"name": "Anomalia da Tmáx do dia (normal 1991–2020)", "layers": [
        {"type": "fill", "var": "tmax_anom"},
        {"type": "values", "var": "tmax_anom", "fmt": "%+.0f"}]},
    "tmin_anom": {"name": "Anomalia da Tmín do dia (normal 1991–2020)", "layers": [
        {"type": "fill", "var": "tmin_anom"},
        {"type": "values", "var": "tmin_anom", "fmt": "%+.0f"}]},
    "clouds": {"name": "Nebulosidade total + pressão MSL", "layers": [
        {"type": "fill", "var": "tcc"},
        {"type": "contour", "var": "msl", "color": "#0b3c8c"}]},
}


def preset_layers(preset: str, model: Model) -> list[dict]:
    layers = [dict(x) for x in PRESETS[preset]["layers"]]
    for ly in layers:  # graceful fallbacks for models lacking a field
        if ly.get("var") == "mucape" and not model.has("mucape"):
            ly["var"] = next((v for v in ("cape", "mlcape") if model.has(v)), "lapse")
    return layers


def _layer_label(ly: dict, var, ctx: Ctx) -> str:
    lvl = f" {ly['level']} hPa" if var.upper and ly.get("level") else ""
    name = var.name
    if "day" in ctx.notes:
        d = ctx.notes["day"]
        name = f"{var.name} de {WEEKDAYS[d.weekday()]} {d.day:02d} {MONTHS[d.month - 1]}"
    if var.id == "precip":
        st0 = ctx.notes.get("accum_from", 0)
        h = ctx.step - st0
        name = f"Precipitação acumulada {h} h" if st0 else f"Precipitação total (+{ctx.step} h)"
    return f"{name}{lvl} ({var.units})"


async def build_job(model: Model, run: dt.datetime, step: int, steps: list[int],
                    region_id: str, layers: list[dict]) -> dict:
    region = REGIONS[region_id]
    domain = region.download_domain
    if not model.covers_box(*region.extent):
        raise FieldUnavailable(model.outside_msg())

    plan = []          # (layer, ctx, var|None, needs)
    needs = []
    for ly in layers:
        t = ly.get("type")
        if t not in ("fill", "contour", "wind", "values"):
            continue
        level = int(ly.get("level") or 0)
        ctx = Ctx(model, run, step, steps, level=level, accum=int(ly.get("accum", 3) or 0))
        if t == "wind":
            nd = wind_needs(ctx, level)
            plan.append((ly, ctx, None, nd))
        else:
            var = VAR_BY_ID.get(ly.get("var"))
            if var is None:
                raise FieldUnavailable(f"variável desconhecida: {ly.get('var')}")
            if var.upper and not level:
                ctx.level = level = 500
                ly["level"] = 500
            if not var.is_available(model):
                raise FieldUnavailable(f"{var.name} não disponível no {model.name}")
            nd = var.needs(ctx)
            plan.append((ly, ctx, var, nd))
        needs += nd

    data = await gather(model, run, step, steps, needs, domain, Ctx(model, run, step, steps))
    if not data.f:
        raise FieldUnavailable(f"sem dados {model.name} para {run:%Y-%m-%d %HZ} +{step}h")
    grid = data.grid

    out_layers, labels = [], []
    for ly, ctx, var, _ in plan:
        d = Data(ctx, data.f)
        if var is None:
            lv = ctx.level
            u = d("u10") if lv == 0 else d("u", lv)
            v = d("v10") if lv == 0 else d("v", lv)
            out_layers.append({**ly, "u": (u * 1.943844).astype(np.float32),
                               "v": (v * 1.943844).astype(np.float32)})
            labels.append(f"Vento {'10 m' if lv == 0 else f'{lv} hPa'} (kt)")
        else:
            arr = evaluate(var, d).astype(np.float32)
            label = _layer_label(ly, var, ctx)
            out_layers.append({**ly, "data": arr, "label": label})
            if ly["type"] == "values":
                continue
            labels.append(label if ly["type"] == "fill" else label + " [isolinhas]")

    valid = run + dt.timedelta(hours=step)
    src = model.source
    return {
        "region": region_id,
        "lats": grid.lats, "lons": grid.lons,
        "layers": out_layers,
        "header": {
            "left1": f"{model.name} {model.resolution}   ·   Run {fmt_time(run)}",
            "left2": f"Previsão +{step} h   ·   {region.name}",
            "layers": "  ·  ".join(labels),
            "right1": f"Válido: {fmt_time(valid)}",
            "right2": f"{run:%Y-%m-%d %H} UTC + {step} h",
        },
        "footer": f"Dados: {src}",
    }
