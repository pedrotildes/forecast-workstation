"""Catalogue of plottable variables (raw and derived) in display units.

Each :class:`Var` declares which canonical raw fields it needs (possibly at
other forecast steps, e.g. for precipitation accumulations) and how to compute
the displayed 2-D array from them.  This keeps the map renderer completely
model-agnostic.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.ndimage import gaussian_filter

from .grib import Field
from .models import Domain, FieldUnavailable, Key, Model

EARTH_R = 6371000.0
G = 9.80665


# ----------------------------------------------------------------------------
@dataclass
class Ctx:
    model: Model
    run: dt.datetime
    step: int
    steps: list[int]
    level: int = 0                 # hPa for upper-air variables
    accum: int = 0                 # hours, for precipitation (0 = since t0)
    notes: dict = field(default_factory=dict)

    def prev_step(self, hours: int) -> int:
        """Latest available step <= step - hours (0 if none)."""
        target = self.step - hours
        if target <= 0:
            return 0
        cand = [s for s in self.steps if s <= target]
        return max(cand) if cand else 0


Need = tuple[Key, int]                 # (key, step)


class Data:
    def __init__(self, ctx: Ctx, fields: dict[Need, Field]):
        self.ctx, self.f = ctx, fields

    def __call__(self, name: str, level: int = 0, step: int | None = None) -> np.ndarray:
        st = self.ctx.step if step is None else step
        try:
            return self.f[((name, level), st)].values.astype(np.float64)
        except KeyError:
            raise FieldUnavailable(f"{self.ctx.model.name}: campo {name}"
                                   f"{'@' + str(level) if level else ''} indisponível") from None

    def has(self, name, level=0, step=None):
        st = self.ctx.step if step is None else step
        return ((name, level), st) in self.f

    @property
    def grid(self) -> Field:
        return next(iter(self.f.values()))


@dataclass
class Var:
    id: str
    name: str
    units: str
    upper: bool                                     # requires a pressure level
    needs: Callable[[Ctx], list[Need]]
    compute: Callable[[Data], np.ndarray]
    group: str = "Superfície"
    levels: list[int] | None = None                 # restrict selectable levels
    available: Callable[[Model], bool] | None = None
    smooth: float = 0.0                             # gaussian sigma (grid points)

    def is_available(self, m: Model) -> bool:
        return self.available(m) if self.available else True

    def info(self, m: Model | None = None) -> dict:
        lv = self.levels if self.levels else (m.levels if m and self.upper else None)
        return {"id": self.id, "name": self.name, "units": self.units,
                "upper": self.upper, "group": self.group, "levels": lv}


# ----------------------------------------------------------------------------
# helpers
def _k(ctx: Ctx, *names: str, level: int | None = None) -> list[Need]:
    lv = ctx.level if level is None else level
    return [((n, lv), ctx.step) for n in names]


def _s(ctx: Ctx, *names: str) -> list[Need]:
    return [((n, 0), ctx.step) for n in names]


def dewpoint_from_rh(t_k, rh):
    t = t_k - 273.15
    rh = np.clip(rh, 0.5, 100.0)
    a, b = 17.625, 243.04
    g = np.log(rh / 100.0) + a * t / (b + t)
    return b * g / (a - g)                           # °C


def rh_from_q(t_k, q, p_hpa):
    t = t_k - 273.15
    es = 6.112 * np.exp(17.67 * t / (t + 243.5))
    e = q * p_hpa / (0.622 + 0.378 * q)
    return np.clip(100.0 * e / es, 0, 100)


def thetae(t_k, td_c, p_hpa):
    """Bolton (1980) equivalent potential temperature (K)."""
    td_k = td_c + 273.15
    e = 6.112 * np.exp(17.67 * td_c / (td_c + 243.5))
    r = 0.622 * e / (p_hpa - e)
    tl = 1.0 / (1.0 / (td_k - 56.0) + np.log(t_k / td_k) / 800.0) + 56.0
    th_l = t_k * (1000.0 / (p_hpa - e)) ** 0.2854 * (t_k / tl) ** (0.28 * r)
    return th_l * np.exp((3036.0 / tl - 1.78) * r * (1 + 0.448 * r))


def _rh_level(d: Data, lv: int):
    if d.has("r", lv):
        return d("r", lv)
    return rh_from_q(d("t", lv), d("q", lv), lv)


def _needs_rh(ctx: Ctx, lv: int | None = None) -> list[Need]:
    lv = ctx.level if lv is None else lv
    if ctx.model.has("r", lv):
        return _k(ctx, "t", "r", level=lv)
    return _k(ctx, "t", "q", level=lv)


def _metrics(g: Field):
    lat = np.deg2rad(g.lats)[:, None]
    dlon = np.deg2rad(np.gradient(g.lons))[None, :]
    dlat = np.deg2rad(np.gradient(g.lats))[:, None]
    dx = EARTH_R * np.cos(lat) * dlon
    dy = EARTH_R * dlat * np.ones_like(dx)
    return dx, dy, lat


def ddx(a, dx):
    return np.gradient(a, axis=1) / dx


def ddy(a, dy):
    return np.gradient(a, axis=0) / dy


def vorticity(u, v, g: Field):
    dx, dy, lat = _metrics(g)
    return ddx(v, dx) - ddy(u, dy) + u * np.tan(lat) / EARTH_R


def divergence(u, v, g: Field):
    dx, dy, lat = _metrics(g)
    return ddx(u, dx) + ddy(v, dy) - v * np.tan(lat) / EARTH_R


def advection(s, u, v, g: Field):
    dx, dy, _ = _metrics(g)
    return -(u * ddx(s, dx) + v * ddy(s, dy))


KT = 1.943844


# ----------------------------------------------------------------------------
def _precip_needs(ctx: Ctx) -> list[Need]:
    st0 = 0 if ctx.accum == 0 else ctx.prev_step(ctx.accum)
    ctx.notes["accum_from"] = st0
    need = [(("tp", 0), ctx.step)]
    if st0 > 0:
        need.append((("tp", 0), st0))
    return need


def _precip(d: Data):
    st0 = d.ctx.notes.get("accum_from", 0)
    cur = d("tp")
    if st0 <= 0:
        return np.maximum(cur, 0)
    return np.maximum(cur - d("tp", step=st0), 0)


def _wind_speed_needs(ctx):
    return _k(ctx, "u", "v")


VARS: list[Var] = [
    # --------------------------------------------------------------- surface
    Var("msl", "Pressão ao nível médio do mar", "hPa", False,
        lambda c: _s(c, "msl"), lambda d: d("msl") / 100.0, smooth=1.0),
    Var("t2m", "Temperatura a 2 m", "°C", False,
        lambda c: _s(c, "t2m"), lambda d: d("t2m") - 273.15),
    Var("d2m", "Ponto de orvalho a 2 m", "°C", False,
        lambda c: _s(c, "d2m"), lambda d: d("d2m") - 273.15),
    Var("rh2m", "Humidade relativa a 2 m", "%", False,
        lambda c: _s(c, "t2m", "d2m"),
        lambda d: np.clip(100 * np.exp(17.625 * (d("d2m") - 273.15) / (d("d2m") - 30.11)
                                       - 17.625 * (d("t2m") - 273.15) / (d("t2m") - 30.11)), 0, 100)),
    Var("wind10", "Vento a 10 m", "kt", False,
        lambda c: _s(c, "u10", "v10"), lambda d: np.hypot(d("u10"), d("v10")) * KT),
    Var("gust", "Rajada máxima a 10 m", "km/h", False,
        lambda c: _s(c, "gust"), lambda d: d("gust") * 3.6,
        available=lambda m: m.has("gust")),
    Var("precip", "Precipitação acumulada", "mm", False, _precip_needs, _precip,
        group="Precipitação e nuvens"),
    Var("tcc", "Nebulosidade total", "%", False, lambda c: _s(c, "tcc"), lambda d: d("tcc"),
        group="Precipitação e nuvens", available=lambda m: m.has("tcc")),
    Var("lcc", "Nuvens baixas", "%", False, lambda c: _s(c, "lcc"), lambda d: d("lcc"),
        group="Precipitação e nuvens", available=lambda m: m.has("lcc")),
    Var("mcc", "Nuvens médias", "%", False, lambda c: _s(c, "mcc"), lambda d: d("mcc"),
        group="Precipitação e nuvens", available=lambda m: m.has("mcc")),
    Var("hcc", "Nuvens altas", "%", False, lambda c: _s(c, "hcc"), lambda d: d("hcc"),
        group="Precipitação e nuvens", available=lambda m: m.has("hcc")),
    Var("pwat", "Água precipitável", "mm", False, lambda c: _s(c, "pwat"), lambda d: d("pwat"),
        group="Precipitação e nuvens"),
    Var("refc", "Refletividade composta simulada", "dBZ", False,
        lambda c: _s(c, "refc"), lambda d: d("refc"),
        group="Precipitação e nuvens", available=lambda m: m.has("refc")),
    # ----------------------------------------------------------- convection
    Var("cape", "CAPE (superfície)", "J/kg", False, lambda c: _s(c, "cape"), lambda d: d("cape"),
        group="Convecção", available=lambda m: m.has("cape")),
    Var("mlcape", "MLCAPE (90 hPa)", "J/kg", False, lambda c: _s(c, "mlcape"), lambda d: d("mlcape"),
        group="Convecção", available=lambda m: m.has("mlcape")),
    Var("mucape", "MUCAPE", "J/kg", False, lambda c: _s(c, "mucape"), lambda d: d("mucape"),
        group="Convecção", available=lambda m: m.has("mucape")),
    Var("cin", "CIN (superfície)", "J/kg", False, lambda c: _s(c, "cin"), lambda d: d("cin"),
        group="Convecção", available=lambda m: m.has("cin")),
    Var("shear", "Cisalhamento 10 m – 500 hPa (≈0–6 km)", "kt", False,
        lambda c: _s(c, "u10", "v10") + _k(c, "u", "v", level=500),
        lambda d: np.hypot(d("u", 500) - d("u10"), d("v", 500) - d("v10")) * KT,
        group="Convecção"),
    Var("lapse", "Gradiente térmico vertical 700–500 hPa", "°C/km", False,
        lambda c: _k(c, "t", "gh", level=700) + _k(c, "t", "gh", level=500),
        lambda d: (d("t", 700) - d("t", 500)) / ((d("gh", 500) - d("gh", 700)) / 1000.0),
        group="Convecção", smooth=1.0),
    Var("tt", "Índice Total Totals", "°C", False,
        lambda c: _needs_rh(c, 850) + _k(c, "t", level=500),
        lambda d: (d("t", 850) - 273.15) + dewpoint_from_rh(d("t", 850), _rh_level(d, 850))
        - 2 * (d("t", 500) - 273.15),
        group="Convecção", smooth=0.7),
    Var("kindex", "Índice K", "°C", False,
        lambda c: _needs_rh(c, 850) + _needs_rh(c, 700) + _k(c, "t", level=500),
        lambda d: (d("t", 850) - d("t", 500))
        + dewpoint_from_rh(d("t", 850), _rh_level(d, 850))
        - ((d("t", 700) - 273.15) - dewpoint_from_rh(d("t", 700), _rh_level(d, 700))),
        group="Convecção", smooth=0.7),
    Var("frzl", "Altitude da isotérmica de 0 °C", "m", False,
        lambda c: _s(c, "frzl"), lambda d: d("frzl"),
        group="Convecção", available=lambda m: m.has("frzl")),
    # --------------------------------------------------------------- upper air
    Var("gh", "Geopotencial", "dam", True, lambda c: _k(c, "gh"), lambda d: d("gh", d.ctx.level) / 10.0,
        group="Altitude", smooth=1.0),
    Var("t", "Temperatura", "°C", True, lambda c: _k(c, "t"), lambda d: d("t", d.ctx.level) - 273.15,
        group="Altitude", smooth=0.5),
    Var("rh", "Humidade relativa", "%", True, lambda c: _needs_rh(c),
        lambda d: _rh_level(d, d.ctx.level), group="Altitude"),
    Var("td", "Ponto de orvalho", "°C", True, lambda c: _needs_rh(c),
        lambda d: dewpoint_from_rh(d("t", d.ctx.level), _rh_level(d, d.ctx.level)), group="Altitude"),
    Var("tdd", "Depressão do ponto de orvalho (T−Td)", "°C", True, lambda c: _needs_rh(c),
        lambda d: (d("t", d.ctx.level) - 273.15)
        - dewpoint_from_rh(d("t", d.ctx.level), _rh_level(d, d.ctx.level)), group="Altitude"),
    Var("thetae", "Temperatura potencial equivalente θe", "K", True, lambda c: _needs_rh(c),
        lambda d: thetae(d("t", d.ctx.level),
                         dewpoint_from_rh(d("t", d.ctx.level), _rh_level(d, d.ctx.level)),
                         float(d.ctx.level)), group="Altitude", smooth=0.8),
    Var("wspd", "Velocidade do vento", "kt", True, _wind_speed_needs,
        lambda d: np.hypot(d("u", d.ctx.level), d("v", d.ctx.level)) * KT, group="Altitude"),
    Var("w", "Velocidade vertical ω", "Pa/s", True, lambda c: _k(c, "w"),
        lambda d: d("w", d.ctx.level), group="Altitude", smooth=1.0),
    Var("vort", "Vorticidade relativa", "10⁻⁵ s⁻¹", True, _wind_speed_needs,
        lambda d: vorticity(d("u", d.ctx.level), d("v", d.ctx.level), d.grid) * 1e5,
        group="Altitude", smooth=1.2),
    Var("div", "Divergência", "10⁻⁵ s⁻¹", True, _wind_speed_needs,
        lambda d: divergence(d("u", d.ctx.level), d("v", d.ctx.level), d.grid) * 1e5,
        group="Altitude", smooth=1.5),
    Var("tadv", "Advecção de temperatura", "°C/3h", True, lambda c: _k(c, "t", "u", "v"),
        lambda d: advection(gaussian_filter(d("t", d.ctx.level), 1.5), d("u", d.ctx.level),
                            d("v", d.ctx.level), d.grid) * 3 * 3600,
        group="Altitude", smooth=1.0),
    Var("q", "Humidade específica", "g/kg", True, lambda c: _k(c, "q"),
        lambda d: d("q", d.ctx.level) * 1000.0, group="Altitude",
        available=lambda m: m.has("q")),
    Var("thick", "Espessura 1000–500 hPa", "dam", False,
        lambda c: _k(c, "gh", level=500) + _k(c, "gh", level=1000),
        lambda d: (d("gh", 500) - d("gh", 1000)) / 10.0, group="Altitude", smooth=1.0),
]


# ----------------------------------------------------------------- daily extremes
def _daily_needs(kind: str, anomaly: bool):
    def needs(ctx: Ctx) -> list[Need]:
        import asyncio as _aio

        from . import climate
        from .daily import daily_plan, keys_for
        if anomaly and not climate.ready():
            _aio.get_running_loop().create_task(climate.ensure())
            raise FieldUnavailable("A descarregar a climatologia CPC 1991–2020 (só na primeira vez, "
                                   "~1 min). Tente de novo dentro de instantes.")
        day = (ctx.run + dt.timedelta(hours=ctx.step)).date()
        p = daily_plan(ctx.model, ctx.run, ctx.steps).get(day)
        nm = "Tmáx" if kind == "tmax" else "Tmín"
        if not p or not p[f"{kind}_ok"]:
            raise FieldUnavailable(f"{nm} de {day:%d/%m} incompleta nesta run "
                                   "(escolha um passo de outro dia)")
        ctx.notes.update(day=day, steps=p[kind], key=keys_for(ctx.model, kind))
        return [(ctx.notes["key"], st) for st in p[kind]]
    return needs


def _daily_compute(kind: str, anomaly: bool):
    def compute(d: Data) -> np.ndarray:
        k = d.ctx.notes["key"]
        st = np.stack([d(k[0], 0, step=s) for s in d.ctx.notes["steps"]])
        v = (np.nanmax(st, 0) if kind == "tmax" else np.nanmin(st, 0)) - 273.15
        if anomaly:
            from . import climate
            g = d.grid
            v = v - climate.normal_on(kind, d.ctx.notes["day"], g.lats, g.lons)
        return v
    return compute


VARS += [
    Var("tmax_day", "Temperatura máxima diária", "°C", False, _daily_needs("tmax", False),
        _daily_compute("tmax", False), group="Temperatura diária"),
    Var("tmin_day", "Temperatura mínima diária", "°C", False, _daily_needs("tmin", False),
        _daily_compute("tmin", False), group="Temperatura diária"),
    Var("tmax_anom", "Anomalia da Tmáx (normal 1991–2020)", "°C", False, _daily_needs("tmax", True),
        _daily_compute("tmax", True), group="Temperatura diária"),
    Var("tmin_anom", "Anomalia da Tmín (normal 1991–2020)", "°C", False, _daily_needs("tmin", True),
        _daily_compute("tmin", True), group="Temperatura diária"),
]
VAR_BY_ID = {v.id: v for v in VARS}


def wind_needs(ctx: Ctx, level: int) -> list[Need]:
    if level == 0:
        return _s(ctx, "u10", "v10")
    return _k(ctx, "u", "v", level=level)


async def gather(model: Model, run, step: int, steps: list[int], needs: list[Need],
                 domain: Domain, ctx: Ctx) -> Data:
    by_step: dict[int, list[Key]] = {}
    for key, st in needs:
        by_step.setdefault(st, []).append(key)
    out: dict[Need, Field] = {}
    for st, keys in by_step.items():
        got = await model.fetch(run, st, keys, domain)
        for k, f in got.items():
            out[(k, st)] = f
    return Data(ctx, out)


def evaluate(var: Var, data: Data) -> np.ndarray:
    arr = var.compute(data)
    if var.smooth:
        lats = data.grid.lats
        res = abs(float(lats[1] - lats[0])) if len(lats) > 1 else 0.25
        sigma = var.smooth * max(1.0, 0.25 / res)   # smoothing scales are set for 0.25° grids
        nan = np.isnan(arr)
        arr = gaussian_filter(np.where(nan, np.nanmean(arr), arr), sigma)
        arr[nan] = np.nan
    return arr
