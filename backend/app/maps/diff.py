"""Model difference charts: (A − B) shaded, with both models' isolines on top."""
from __future__ import annotations

import datetime as dt

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ..models import FieldUnavailable, Model
from ..products import VAR_BY_ID, Ctx, evaluate, gather
from ..regions import REGIONS
from .compose import fmt_time
from .styles import Fill, _interp

CONTOUR_VARS = {"gh", "msl", "t", "thick", "thetae", "t2m", "d2m", "td"}
NICE = [0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000]


def diff_fill(diff: np.ndarray) -> Fill:
    m = float(np.nanpercentile(np.abs(diff), 99)) if np.isfinite(diff).any() else 1.0
    step = next((x for x in NICE if x >= max(m, 1e-6) / 7), NICE[-1])
    lv = [round(x, 6) for x in np.arange(-8 * step, 8 * step + step / 2, step)]
    cols = _interp(["#053061", "#2166ac", "#4393c3", "#92c5de", "#f7f7f7", "#f7f7f7",
                    "#f4a582", "#d6604d", "#b2182b", "#67001f"], len(lv) - 1, "diff")
    mid = (len(lv) - 1) // 2
    cols[mid - 1] = cols[mid] = (1, 1, 1, 0)
    return Fill(lv, cols, under="#021b3a", over="#3d0010", tick_every=2)


async def _field(model: Model, run, step, var, level, accum, domain):
    steps = await model.list_steps(run)
    ctx = Ctx(model, run, step, steps, level=level, accum=accum)
    data = await gather(model, run, step, steps, var.needs(ctx), domain, ctx)
    if not data.f:
        raise FieldUnavailable(f"{model.name}: sem dados")
    return evaluate(var, data), data.grid, ctx


async def build_diff_job(a: tuple, b: tuple, var_id: str, level: int, region_id: str,
                         accum: int = 6) -> dict:
    (ma, ra, sa), (mb, rb, sb) = a, b
    var = VAR_BY_ID.get(var_id)
    if var is None:
        raise FieldUnavailable(f"variável desconhecida: {var_id}")
    for m in (ma, mb):
        if not var.is_available(m):
            raise FieldUnavailable(f"{var.name} não disponível no {m.name}")
    if var.upper and not level:
        level = 500
    region = REGIONS[region_id]
    dom = region.download_domain
    fa, ga, ctx_a = await _field(ma, ra, sa, var, level, accum, dom)
    fb, gb, _ = await _field(mb, rb, sb, var, level, accum, dom)
    interp = RegularGridInterpolator((gb.lats, gb.lons), fb, bounds_error=False, fill_value=np.nan)
    LA, LO = np.meshgrid(ga.lats, ga.lons, indexing="ij")
    fb_on_a = interp(np.stack([LA, LO], axis=-1))
    diff = (fa - fb_on_a).astype(np.float32)

    lvl = f" {level} hPa" if var.upper else ""
    name = var.name
    if var.id == "precip":
        st0 = ctx_a.notes.get("accum_from", 0)
        name = f"Precipitação acumulada {sa - st0} h" if st0 else "Precipitação total"
    label = f"{ma.name} − {mb.name}: {name}{lvl} ({var.units})"
    fill = diff_fill(diff)
    step = fill.levels[1] - fill.levels[0]
    shown = np.where(np.abs(diff) < step, np.nan, diff).astype(np.float32)  # no seams near zero
    layers = [{"type": "fill", "var": var.id, "level": level, "data": shown, "label": label,
               "fill_style": fill}]
    if var.id in CONTOUR_VARS:
        layers += [
            {"type": "contour", "var": var.id, "level": level, "data": fa.astype(np.float32),
             "color": "#111111"},
            {"type": "contour", "var": var.id, "level": level, "data": fb_on_a.astype(np.float32),
             "color": "#c1121f", "style": "dashed", "extrema": False},
        ]
    valid = ra + dt.timedelta(hours=sa)
    srcs = {"NOAA/NCEP NOMADS" if m.id == "gfs" else "© ECMWF Open Data (CC BY 4.0)" for m in (ma, mb)}
    iso = (f"  ·  Isolinhas: {ma.name} (preto) e {mb.name} (vermelho tracejado)"
           if var.id in CONTOUR_VARS else "")
    return {
        "region": region_id, "lats": ga.lats, "lons": ga.lons, "layers": layers,
        "header": {
            "left1": f"Diferença {ma.name} − {mb.name}   ·   {name}{lvl}",
            "left2": f"{ma.name}: run {fmt_time(ra)} +{sa} h   ·   {mb.name}: run {fmt_time(rb)} +{sb} h",
            "layers": f"Sombreado: {ma.name} − {mb.name} ({var.units}; vermelho = {ma.name} mais alto){iso}",
            "right1": f"Válido: {fmt_time(valid)}",
            "right2": region.name,
        },
        "footer": "Dados: " + " · ".join(sorted(srcs)),
    }
