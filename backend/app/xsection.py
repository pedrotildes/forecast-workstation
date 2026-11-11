"""Vertical cross-sections along a line A→B (great circle) — forecasts and case studies.

Pressure-level fields are fetched over a box around the line; derived quantities are
computed on the grid (potential and equivalent potential temperature, relative humidity,
isobaric Ertel potential vorticity) and interpolated bilinearly to points along the great
circle. Wind is decomposed into the component along the section (A→B) and the component
normal to it (positive = into the figure, i.e. towards the left of the direction A→B).
The figure is drawn by :func:`render` in the worker pool.
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from .models import FieldUnavailable
from .models.base import Domain

R_EARTH = 6.371e6
G = 9.80665
OMEGA = 7.292e-5
KT = 1.943844

KINDS = {
    "dyn": "Vento, θ e tropopausa dinâmica (2 PVU)",
    "normal": "Vento normal ao corte, θ e 2 PVU",
    "moist": "θe, humidade relativa e subida (ω)",
    "omega": "Movimento vertical (ω), θ e HR 90 %",
    "pv": "Vorticidade potencial, θ e isotácas",
    "temp": "Temperatura, isotérmica 0 °C e HR 90 %",
}
TOP = {"dyn": 100, "normal": 100, "pv": 100, "moist": 200, "omega": 200, "temp": 200}

MIN_KM, MAX_KM = 150.0, 6000.0


# ------------------------------------------------------------------ geometry
def _xyz(lat, lon):
    la, lo = math.radians(lat), math.radians(lon)
    return np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])


def length_km(lat0, lon0, lat1, lon1) -> float:
    c = float(np.clip(np.dot(_xyz(lat0, lon0), _xyz(lat1, lon1)), -1, 1))
    return math.acos(c) * R_EARTH / 1000


def great_circle(lat0, lon0, lat1, lon1, n: int):
    """n points along the great circle: lats, lons, distance from A (km), azimuth (deg)."""
    p0, p1 = _xyz(lat0, lon0), _xyz(lat1, lon1)
    ang = math.acos(float(np.clip(np.dot(p0, p1), -1, 1)))
    f = np.linspace(0, 1, n)
    if ang < 1e-9:
        pts = np.repeat(p0[None], n, 0)
    else:
        pts = (np.sin((1 - f) * ang)[:, None] * p0 + np.sin(f * ang)[:, None] * p1) / math.sin(ang)
    lats = np.degrees(np.arcsin(np.clip(pts[:, 2], -1, 1)))
    lons = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    dist = f * ang * R_EARTH / 1000
    la, lo = np.radians(lats), np.radians(lons)
    la2, lo2 = np.r_[la[1:], la[-1]], np.r_[lo[1:], lo[-1]]
    az = np.degrees(np.arctan2(np.sin(lo2 - lo) * np.cos(la2),
                               np.cos(la) * np.sin(la2) - np.sin(la) * np.cos(la2) * np.cos(lo2 - lo)))
    az[-1] = az[-2] if n > 1 else az[-1]
    return lats, lons, dist, az


def line_domain(lat0, lon0, lat1, lon1, margin=3.0, clip: Domain | None = None) -> Domain:
    la, lo, _, _ = great_circle(lat0, lon0, lat1, lon1, 60)
    # keep the box roughly square (in km) so the location map is readable
    cl = max(0.2, math.cos(math.radians(float(la.mean()))))
    w, h = float(lo.max() - lo.min()) * cl, float(la.max() - la.min())
    side = max(w, h)
    mh = margin + max(0.0, 0.65 * side - h) / 2
    mw = margin + max(0.0, 0.65 * side - w) / 2 / cl
    d = Domain(float(lo.min() - mw), float(lo.max() + mw),
               float(la.min() - mh), float(la.max() + mh)).rounded()
    if clip is not None:
        d = Domain(max(d.lon0, clip.lon0), min(d.lon1, clip.lon1),
                   max(d.lat0, clip.lat0), min(d.lat1, clip.lat1))
    return Domain(max(-180.0, d.lon0), min(179.75, d.lon1), max(-89.0, d.lat0), min(89.0, d.lat1))


def check_line(lat0, lon0, lat1, lon1):
    for la, lo in ((lat0, lon0), (lat1, lon1)):
        if not (-85 <= la <= 85 and -180 <= lo <= 180):
            raise ValueError("coordenadas inválidas")
    L = length_km(lat0, lon0, lat1, lon1)
    if L < MIN_KM:
        raise ValueError(f"linha demasiado curta ({L:.0f} km; mínimo {MIN_KM:.0f} km)")
    if L > MAX_KM:
        raise ValueError(f"linha demasiado longa ({L:.0f} km; máximo {MAX_KM:.0f} km)")
    if abs(lon1 - lon0) > 180:
        raise ValueError("a linha não pode atravessar o antimeridiano")
    return L


# ------------------------------------------------------------------ data
async def fetch_grids(model, run: dt.datetime, step: int, dom: Domain) -> dict:
    """3-D cubes (level, lat, lon) of gh, t, u, v, rh, w and 2-D sp/msl over dom."""
    from .products import rh_from_q
    lv = sorted([p for p in model.levels if p >= 100], reverse=True)
    if len(lv) < 5:
        raise FieldUnavailable(f"{model.name} não tem níveis de pressão suficientes para um corte")
    hum = "r" if model.has("r") else "q"
    names = ["gh", "t", "u", "v", hum] + (["w"] if model.has("w") else [])
    keys = [(n, p) for p in lv for n in names] + [(n, 0) for n in ("sp", "msl") if model.has(n)]
    f = await model.fetch(run, step, keys, dom)
    upper = [f[k] for k in keys if k in f and k[1]]
    if not upper:
        raise FieldUnavailable(f"{model.name}: sem campos em altitude para este corte")
    # fields may come on different extents (event archive crops vs. fresh downloads):
    # use the most common grid and resample the others onto it
    shapes = [fl.values.shape for fl in upper]
    shape = max(set(shapes), key=shapes.count)
    ref = next(fl for fl in upper if fl.values.shape == shape)

    def on_ref(fl):
        if fl.values.shape == shape and np.allclose(fl.lats, ref.lats) and np.allclose(fl.lons, ref.lons):
            return fl.values
        fi = (np.asarray(ref.lats) - fl.lats[0]) / (fl.lats[1] - fl.lats[0])
        fj = (np.asarray(ref.lons) - fl.lons[0]) / (fl.lons[1] - fl.lons[0])
        FI, FJ = np.meshgrid(fi, fj, indexing="ij")
        return map_coordinates(np.asarray(fl.values, np.float64), [FI, FJ], order=1, mode="nearest",
                               cval=np.nan).astype(np.float32)

    def cube(n):
        a = np.full((len(lv),) + shape, np.nan, np.float32)
        for i, p in enumerate(lv):
            fl = f.get((n, p))
            if fl is not None:
                a[i] = on_ref(fl)
        return a

    def flat(n):
        fl = f.get((n, 0))
        return on_ref(fl) / 100.0 if fl is not None else None

    t = cube("t")
    ok = ~np.isnan(t).all(axis=(1, 2)) & ~np.isnan(cube("u")).all(axis=(1, 2))
    if ok.sum() < 5:
        raise FieldUnavailable(f"{model.name}: níveis em falta para este corte")
    P = np.array(lv, dtype=float)
    hm = cube(hum)
    rh = hm if hum == "r" else rh_from_q(t, hm, P[:, None, None]).astype(np.float32)
    out = {"lats": ref.lats, "lons": ref.lons, "levels": P[ok], "t": t[ok], "u": cube("u")[ok],
           "v": cube("v")[ok], "gh": cube("gh")[ok], "rh": rh[ok],
           "w": cube("w")[ok] if "w" in names else None, "sp": flat("sp"), "msl": flat("msl")}
    return out


def _fill_nan(a):
    if not np.isnan(a).any():
        return a
    a = a.copy()
    for k in range(a.shape[0]):
        m = np.isnan(a[k])
        if m.all():
            continue
        a[k][m] = np.nanmean(a[k])
    return a


def compute(g: dict, lat0, lon0, lat1, lon1) -> dict:
    """Derived fields on the grid, interpolated to the section."""
    lats, lons, P = np.asarray(g["lats"], float), np.asarray(g["lons"], float), g["levels"]
    dlat, dlon = float(lats[1] - lats[0]), float(lons[1] - lons[0])
    Pc = P[:, None, None]
    T = g["t"].astype(np.float64)
    th = T * (1000.0 / Pc) ** 0.2857
    rh = np.clip(np.nan_to_num(g["rh"], nan=50.0), 1, 100)
    # equivalent potential temperature (Bolton 1980)
    tc = T - 273.15
    e = rh / 100 * 6.112 * np.exp(17.67 * tc / (tc + 243.5))
    r = 622.0 * e / (Pc - e)                                   # g/kg
    TL = 1.0 / (1.0 / (T - 55.0) - np.log(rh / 100) / 2840.0) + 55.0
    the = T * (1000.0 / Pc) ** (0.2854 * (1 - 0.00028 * r)) * np.exp((3.376 / TL - 0.00254) * r * (1 + 0.00081 * r))
    # isobaric Ertel potential vorticity (PVU), fields smoothed to ~0.25° first
    sig = max(0.6, 0.25 / abs(dlat))
    sm = lambda a: gaussian_filter(_fill_nan(a), (0, sig, sig), mode="nearest")  # noqa: E731
    us, vs, ths = sm(g["u"].astype(np.float64)), sm(g["v"].astype(np.float64)), sm(th)
    phi = np.radians(lats)[None, :, None]
    dy = R_EARTH * math.radians(dlat)
    dx = R_EARTH * np.cos(phi) * math.radians(dlon)
    pa = P * 100.0
    dth_dp = np.gradient(ths, pa, axis=0)
    du_dp = np.gradient(us, pa, axis=0)
    dv_dp = np.gradient(vs, pa, axis=0)
    dth_dy = np.gradient(ths, axis=1) / dy
    dth_dx = np.gradient(ths, axis=2) / dx
    zeta = np.gradient(vs, axis=2) / dx - np.gradient(us, axis=1) / dy + us * np.tan(phi) / R_EARTH
    fcor = 2 * OMEGA * np.sin(phi)
    pv = -G * ((zeta + fcor) * dth_dp + du_dp * dth_dy - dv_dp * dth_dx) * 1e6

    # ---- section points
    L = length_km(lat0, lon0, lat1, lon1)
    res_km = abs(dlat) * 111.2
    n = int(np.clip(L / (0.6 * res_km), 80, 320))
    la, lo, dist, az = great_circle(lat0, lon0, lat1, lon1, n)
    fi = (la - lats[0]) / dlat
    fj = (lo - lons[0]) / dlon
    if fi.min() < -0.5 or fi.max() > len(lats) - 0.5 or fj.min() < -0.5 or fj.max() > len(lons) - 0.5:
        raise FieldUnavailable("a linha sai da área com dados deste modelo")

    def sec(c):
        return np.stack([map_coordinates(np.asarray(c[k], np.float64), [fi, fj], order=1,
                                         mode="nearest") for k in range(c.shape[0])]).astype(np.float32)

    def line(a):
        return None if a is None else map_coordinates(np.asarray(a, np.float64), [fi, fj],
                                                       order=1, mode="nearest").astype(np.float32)

    W = g["w"]
    if W is not None and abs(dlat) < 0.2:
        # km-scale models: smooth ω to ~0.2° so the synoptic/mesoscale ascent stands out
        W = gaussian_filter(_fill_nan(W.astype(np.float64)), (0, 0.2 / abs(dlat), 0.2 / abs(dlat)), mode="nearest")
    U, V = sec(g["u"]), sec(g["v"])
    azr = np.radians(az)[None, :]
    ut = U * np.sin(azr) + V * np.cos(azr)                 # along A→B
    un = -U * np.cos(azr) + V * np.sin(azr)                # left of A→B = into the figure

    # ---- small map (inset): MSLP + 300 hPa wind / 850 hPa θe, coarsened
    k = max(1, int(math.ceil(max(len(lats), len(lons)) / 140)))
    i300 = int(np.argmin(np.abs(P - 300)))
    i850 = int(np.argmin(np.abs(P - 850)))
    inset = {"lats": lats[::k], "lons": lons[::k],
             "msl": None if g["msl"] is None else g["msl"][::k, ::k].astype(np.float32),
             "jet": (np.hypot(g["u"][i300], g["v"][i300]) * KT)[::k, ::k].astype(np.float32),
             "the850": the[i850][::k, ::k].astype(np.float32),
             "jet_p": int(P[i300]), "the_p": int(P[i850])}
    return {"p": P.astype(np.float32), "dist": dist.astype(np.float32), "lat": la.astype(np.float32),
            "lon": lo.astype(np.float32), "length": L,
            "t": (sec(T) - 273.15), "th": sec(th), "the": sec(the), "rh": sec(rh),
            "u": U * KT, "v": V * KT, "ut": ut * KT, "un": un * KT,
            "spd": np.hypot(U, V) * KT, "w": None if W is None else sec(W),
            "pv": sec(pv), "sp": line(g["sp"]), "inset": inset}


async def section(model, run, step, lat0, lon0, lat1, lon1, clip: Domain | None = None) -> dict:
    import asyncio
    dom = line_domain(lat0, lon0, lat1, lon1, clip=clip)
    g = await fetch_grids(model, run, step, dom)
    return await asyncio.to_thread(compute, g, lat0, lon0, lat1, lon1)


# ------------------------------------------------------------------ rendering (worker pool)
def _ll(lat, lon):
    return f"{abs(lat):.1f}°{'N' if lat >= 0 else 'S'} {abs(lon):.1f}°{'E' if lon >= 0 else 'W'}"


def render(sec: dict, meta: dict) -> bytes:
    """meta: kind, title, subtitle, source, a_name, b_name"""
    import io
    import warnings

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import patheffects as pe
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.figure import Figure
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator

    from .maps.styles import THETAE_COLORS, THETAE_LEVELS, WIND_COLORS, WIND_LEVELS, _interp, temp_fill

    warnings.filterwarnings("ignore")
    kind = meta["kind"]
    P, X = np.asarray(sec["p"], float), np.asarray(sec["dist"], float)
    top = max(TOP[kind], float(P.min()))
    sp = sec.get("sp")
    bottom = 1000.0
    if sp is not None and np.isfinite(sp).any():
        bottom = float(np.clip(np.nanmax(sp) + 8, 1000, 1050))

    fig = Figure(figsize=(14.5, 7.9), dpi=110, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0.055, 0.17, 0.635, 0.68])
    cax = fig.add_axes([0.055, 0.065, 0.635, 0.022])

    def shade(data, levels, colors, extend="both", label="", under=None, over=None, ticks=None):
        cmap = ListedColormap(colors)
        if under:
            cmap.set_under(under)
        if over:
            cmap.set_over(over)
        norm = BoundaryNorm(levels, cmap.N, extend="neither")
        cf = ax.contourf(X, P, data, levels=levels, cmap=cmap, norm=norm, extend=extend, zorder=1)
        cb = fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=ticks or levels)
        cb.ax.tick_params(labelsize=7.5)
        cb.set_label(label, fontsize=8.5)
        return cf

    def lines(data, levels, color, lw=0.8, fmt="%d", ls="solid", every=1, z=3, labels=True):
        if data is None or not np.isfinite(data).any():
            return None
        lv = [x for x in levels if np.nanmin(data) <= x <= np.nanmax(data)]
        if not lv:
            return None
        cs = ax.contour(X, P, data, levels=lv, colors=color, linewidths=lw, linestyles=ls, zorder=z)
        if labels:
            ax.clabel(cs, lv[::every], fmt=fmt, fontsize=6.8, inline_spacing=2)
        return cs

    th, the, rh, pv = sec["th"], sec["the"], sec["rh"], sec["pv"]
    w = sec.get("w")
    legend = []
    theta_lv = list(range(240, 500, 4))

    if kind == "dyn":
        shade(sec["spd"], WIND_LEVELS[2:], WIND_COLORS[2:], "max", "Velocidade do vento (kt)",
              over="#ffe0ff")
        lines(th, theta_lv, "#1f2937", 0.7, every=2)
        lines(pv, [2], "#7c2d12", 2.6, labels=False, z=5)
        legend = ["Sombreado: velocidade do vento horizontal (kt)",
                  "Linhas pretas: temperatura potencial θ (K, 4 em 4)",
                  "Castanho grosso: 2 PVU = tropopausa dinâmica",
                  "Barbelas: vento (kt), norte para cima"]
    elif kind == "normal":
        lv = [-120, -100, -80, -60, -40, -20, -10, 10, 20, 40, 60, 80, 100, 120]
        cols = _interp(["#08306b", "#2171b5", "#6baed6", "#c6dbef"], 6) + ["#ffffff"] + \
            _interp(["#fcbba1", "#fb6a4a", "#cb181d", "#67000d"], 6)
        shade(sec["un"], lv, cols, "both", "Componente do vento normal ao corte (kt): + para dentro da figura ⊗",
              under="#041a40", over="#3b0007")
        lines(th, theta_lv, "#1f2937", 0.7, every=2)
        lines(pv, [2], "#7c2d12", 2.6, labels=False, z=5)
        legend = ["Sombreado: vento normal ao corte (kt)",
                  "  vermelho ⊗ = para dentro da figura (à esquerda de A→B)",
                  "  azul ⊙ = para fora da figura",
                  "Linhas pretas: θ (K) · castanho: 2 PVU"]
    elif kind == "moist":
        lv = [50, 60, 70, 80, 90, 95, 100.1]
        shade(rh, lv, ["#edf6e9", "#c7e9c0", "#86c98a", "#41a35a", "#1b7a3c", "#0b4d25"], "neither",
              "Humidade relativa (%)", ticks=[50, 60, 70, 80, 90, 95, 100])
        lines(the, list(range(250, 400, 3)), "#b91c1c", 0.8, every=2)
        if w is not None:
            lines(w, [-5, -3, -2, -1.5, -1, -0.5], "#6d28d9", 1.2, fmt="%.1f", z=4)
        legend = ["Sombreado: humidade relativa ≥ 50 %",
                  "Linhas vermelhas: θe (K, 3 em 3)",
                  "Linhas violeta: ω < 0 — subida (Pa/s)" if w is not None else "(ω indisponível neste modelo)",
                  "Barbelas: vento (kt)"]
    elif kind == "omega":
        if w is None:
            raise ValueError("este modelo não fornece velocidade vertical (ω)")
        lv = [-3, -2, -1.5, -1, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 1, 1.5, 2, 3]
        cols = _interp(["#67000d", "#cb181d", "#fb6a4a", "#fcbba1"], 6) + ["#ffffff"] + \
            _interp(["#c6dbef", "#6baed6", "#2171b5", "#08306b"], 6)
        shade(w, lv, cols, "both", "ω (Pa/s): negativo = movimento ascendente (vermelho)",
              under="#3b0007", over="#041a40")
        lines(th, theta_lv, "#1f2937", 0.7, every=2)
        lines(rh, [90], "#15803d", 1.6, labels=False, z=4)
        legend = ["Sombreado: velocidade vertical ω (Pa/s)",
                  "  vermelho = subida · azul = descida",
                  "Linhas pretas: θ (K) · verde: HR = 90 % (nuvem)",
                  "Barbelas: vento (kt)"]
    elif kind == "pv":
        lv = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 10]
        shade(pv, lv, ["#f1f5f9", "#dbeafe", "#bfdbfe", "#93c5fd", "#fde68a", "#f59e0b", "#ea580c",
                       "#b91c1c", "#7f1d1d", "#450a0a"], "max", "Vorticidade potencial (PVU)",
              over="#1c0303")
        lines(th, theta_lv, "#1f2937", 0.7, every=2)
        lines(sec["spd"], [60, 80, 100, 120, 140, 160, 180], "#1d4ed8", 1.0, z=4)
        lines(pv, [2], "#111111", 2.2, labels=False, z=5)
        legend = ["Sombreado: vorticidade potencial isobárica (PVU)",
                  "Linhas pretas: θ (K) · preto grosso: 2 PVU",
                  "Linhas azuis: isotácas ≥ 60 kt",
                  "PV calculada dos campos do modelo (suavizados ~25 km)"]
    elif kind == "temp":
        tf = temp_fill(-64, 40, 4)
        shade(sec["t"], tf.levels, tf.colors, "both", "Temperatura (°C)", under=tf.under, over=tf.over,
              ticks=tf.levels[::2])
        lines(sec["t"], [x for x in range(-64, 41, 4) if x != 0], "#374151", 0.5, every=2)
        lines(sec["t"], [0], "#1d4ed8", 2.4, labels=False, z=5)
        lines(rh, [90], "#15803d", 1.6, labels=False, z=4)
        legend = ["Sombreado: temperatura (°C)",
                  "Linha azul grossa: isotérmica de 0 °C",
                  "Linha verde: HR = 90 % (nuvem)",
                  "Barbelas: vento (kt)"]

    # wind barbs (map-like: north up)
    if kind in ("dyn", "moist", "omega", "temp"):
        ncol = 22
        cols_i = np.unique(np.linspace(0, len(X) - 1, ncol).round().astype(int))
        lev_i = [i for i, p in enumerate(P) if p >= top and (len(P) <= 14 or p in
                 (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100))]
        XX, PP = np.meshgrid(X[cols_i], P[lev_i])
        ax.barbs(XX, PP, sec["u"][np.ix_(lev_i, cols_i)], sec["v"][np.ix_(lev_i, cols_i)],
                 length=5.0, linewidth=0.45, color="#111", zorder=6, sizes={"emptybarb": 0.04})

    # ground
    if sp is not None and np.isfinite(sp).any():
        ax.fill_between(X, np.nan_to_num(sp, nan=1013.0), 1100, color="#6d4c41", zorder=7, lw=0)
    ax.set_yscale("log")
    ax.set_ylim(bottom, top)
    ax.set_xlim(X[0], X[-1])
    yt = [p for p in (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100) if top <= p <= bottom]
    ax.set_yticks(yt)
    ax.set_yticklabels([str(p) for p in yt], fontsize=8)
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("Pressão (hPa)", fontsize=8.5)
    ax.grid(True, axis="y", color="#000", alpha=0.08, lw=0.6)
    sec_ax = ax.secondary_yaxis("right", functions=(
        lambda p: 44.3308 * (1 - (np.clip(p, 1, None) / 1013.25) ** 0.190263),
        lambda z: 1013.25 * (1 - np.clip(z, None, 44.3) / 44.3308) ** (1 / 0.190263)))
    sec_ax.yaxis.set_major_locator(FixedLocator([0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16]))
    sec_ax.yaxis.set_major_formatter(FuncFormatter(lambda z, _: f"{z:.0f}"))
    sec_ax.yaxis.set_minor_locator(NullLocator())
    sec_ax.tick_params(labelsize=7.5)
    sec_ax.set_ylabel("Altitude na atmosfera padrão (km)", fontsize=8)
    # x: distance + coordinates
    lat, lon = sec["lat"], sec["lon"]
    idx = np.linspace(0, len(X) - 1, 5).round().astype(int)
    ax.set_xticks(X[idx])
    ax.set_xticklabels([f"{X[i]:.0f} km\n{_ll(lat[i], lon[i])}" for i in idx], fontsize=7.8)
    for x, s, ha in ((0.006, "A", "left"), (0.994, "B", "right")):
        ax.text(x, 0.985, s, transform=ax.transAxes, ha=ha, va="top", fontsize=15, fontweight="bold",
                color="#0d1b2a", zorder=9, path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    # ---- inset map
    ins = sec["inset"]
    mid = len(lat) // 2
    proj = ccrs.LambertConformal(central_longitude=float(lon[mid]), central_latitude=float(lat[mid]),
                                 standard_parallels=(float(lat[mid]) - 5, float(lat[mid]) + 5))
    PC = ccrs.PlateCarree()
    mx = fig.add_axes([0.725, 0.37, 0.26, 0.48], projection=proj)
    ilats, ilons = np.asarray(ins["lats"], float), np.asarray(ins["lons"], float)
    lo0 = max(ilons.min(), min(ilons.min() + 1.0, float(lon.min()) - 1.0))
    lo1 = min(ilons.max(), max(ilons.max() - 1.0, float(lon.max()) + 1.0))
    la0 = max(ilats.min(), min(ilats.min() + 0.6, float(lat.min()) - 0.8))
    la1 = min(ilats.max(), max(ilats.max() - 0.6, float(lat.max()) + 0.8))
    mx.set_extent([lo0, lo1, la0, la1], PC)
    mx.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#efe9dc", edgecolor="none", zorder=0)
    mx.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#dbe9f4", edgecolor="none", zorder=0)
    if kind in ("dyn", "normal", "pv"):
        jl = [60, 80, 100, 120, 140, 160, 180]
        mx.contourf(ilons, ilats, ins["jet"], levels=jl, transform=PC, zorder=1, alpha=0.75,
                    cmap=ListedColormap(_interp(["#bfdbfe", "#60a5fa", "#7c3aed", "#c026d3"], len(jl) - 1)),
                    extend="max")
        inset_lbl = f"Mapa: vento ≥ 60 kt a {ins['jet_p']} hPa e pressão ao nível do mar"
    else:
        mx.contourf(ilons, ilats, ins["the850"], levels=THETAE_LEVELS, colors=THETAE_COLORS,
                    transform=PC, zorder=1, alpha=0.7, extend="both")
        inset_lbl = f"Mapa: θe a {ins['the_p']} hPa e pressão ao nível do mar"
    mx.coastlines("50m", lw=0.6, color="#333", zorder=2)
    mx.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.4, edgecolor="#555", zorder=2)
    if ins["msl"] is not None:
        cs = mx.contour(ilons, ilats, ins["msl"], levels=np.arange(940, 1060, 4), colors="#111",
                        linewidths=0.6, transform=PC, zorder=3)
        mx.clabel(cs, fmt="%d", fontsize=5.5, inline_spacing=1)
    mx.plot(lon, lat, color="#111", lw=3.4, transform=PC, zorder=4)
    mx.plot(lon, lat, color="#facc15", lw=1.6, transform=PC, zorder=5)
    mx.plot(lon[idx[1:-1]], lat[idx[1:-1]], "o", ms=3.5, color="#111", transform=PC, zorder=6)
    for i, s in ((0, "A"), (-1, "B")):
        mx.text(float(lon[i]), float(lat[i]), s, transform=PC, fontsize=12, fontweight="bold",
                ha="center", va="center", zorder=7, color="#0d1b2a",
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    fig.text(0.725, 0.345, inset_lbl, fontsize=7.5, color="#333", va="top")

    # ---- legend / info block
    y = 0.30
    info = [f"A: {meta.get('a_name') or _ll(lat[0], lon[0])}",
            f"B: {meta.get('b_name') or _ll(lat[-1], lon[-1])}",
            f"Comprimento: {sec['length']:.0f} km (círculo máximo)", ""] + legend
    for s in info:
        fig.text(0.725, y, s, fontsize=8, color="#1b263b", va="top")
        y -= 0.027

    fig.text(0.055, 0.965, meta["title"], fontsize=13, fontweight="bold", color="#0d1b2a", va="top")
    fig.text(0.055, 0.925, meta["subtitle"], fontsize=9, color="#1b263b", va="top")
    fig.text(0.985, 0.012, f"Dados: {meta['source']} · Forecast Workstation", fontsize=7, color="#666",
             ha="right")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor="white")
    return buf.getvalue()
