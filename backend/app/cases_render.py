"""Matplotlib renderers for case studies (run in the worker pool)."""
from __future__ import annotations

import datetime as dt
import io
import math
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
from matplotlib import patheffects as pe  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

from .meteogram import MODEL_COLORS, WEEK, MON, _time_axis, _panel_label  # noqa: E402
from .regions import REGIONS  # noqa: E402

warnings.filterwarnings("ignore")
PC = ccrs.PlateCarree()
NAMES = {"gfs": "GFS", "ecmwf": "ECMWF IFS", "aifs": "ECMWF AIFS", "icon_eu": "ICON-EU"}


def _t(s):
    return dt.datetime.fromisoformat(s)


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor="white")
    return buf.getvalue()


def _fmt(t):
    return f"{WEEK[t.weekday()]} {t.day:02d} {MON[t.month - 1]} {t:%H}Z"


# ------------------------------------------------------------------ central pressure
def render_pressure(data: dict) -> bytes:
    """data = {event, tracks: {system: {model: [pts]}}, forecasts: {system: {run: [pts]}},
    fc_model}"""
    systems = list(data["tracks"])
    n = max(1, len(systems))
    fig_h = 3.6 * n + 1.3
    fig = Figure(figsize=(14.5, fig_h), dpi=110, facecolor="white")
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(n, 1, hspace=0.32, left=0.06, right=0.97, top=1 - 1.0 / fig_h,
                          bottom=0.5 / fig_h)
    t_all = [_t(p["time"]) for per in data["tracks"].values() for tr in per.values() for p in tr]
    for per in data.get("forecasts", {}).values():
        t_all += [_t(p["time"]) for tr in per.values() for p in tr]
    if not t_all:
        raise ValueError("sem trajetórias")
    t0, t1 = min(t_all), max(t_all)
    axes = []
    for i, sysname in enumerate(systems):
        ax = fig.add_subplot(gs[i])
        axes.append(ax)
        fcs = data.get("forecasts", {}).get(sysname, {})
        cmap = matplotlib.colormaps["Greys"]
        for k, (r, tr) in enumerate(sorted(fcs.items())):
            shade = 0.35 + 0.55 * (k + 1) / max(1, len(fcs))
            ax.plot([_t(p["time"]) for p in tr], [p["p"] for p in tr], color=cmap(shade), lw=1.0,
                    ls=(0, (4, 2)), zorder=3,
                    label=f"previsões {NAMES.get(data['fc_model'], '')} (mais escuro = mais recente)" if k == len(fcs) - 1 else None)
        for mid, tr in data["tracks"][sysname].items():
            if not tr:
                continue
            ax.plot([_t(p["time"]) for p in tr], [p["p"] for p in tr], color=MODEL_COLORS.get(mid, "#333"),
                    lw=2.2, marker="o", ms=3.5, zorder=5, label=f"{NAMES.get(mid, mid)} (análises)")
            j = int(np.argmin([p["p"] for p in tr]))
            mi = list(data["tracks"][sysname]).index(mid)
            ax.annotate(f"{NAMES.get(mid, mid)}: {tr[j]['p']:.1f} hPa · {_fmt(_t(tr[j]['time']))}",
                        xy=(0.012, 0.30 - 0.075 * mi), xycoords="axes fraction", fontsize=7.8,
                        color=MODEL_COLORS.get(mid, "#333"), fontweight="bold")
        ax.set_ylabel("hPa", fontsize=8)
        ax.invert_yaxis()
        ax.legend(loc="upper right", fontsize=7.5, framealpha=0.92, ncol=2)
        _panel_label(ax, f"Pressão central — {sysname}")
    _time_axis(axes, t0 - dt.timedelta(hours=3), t1 + dt.timedelta(hours=3), axes[0])
    for ax in axes:
        ax.tick_params(labelbottom=True)
    fig.text(0.06, 1 - 0.25 / fig_h, f"Evolução da pressão central — evento «{data['event']}»", fontsize=13,
             fontweight="bold", color="#0d1b2a", va="top")
    fig.text(0.06, 1 - 0.55 / fig_h, "Linhas contínuas: análises de cada modelo (3/3 h); tracejado: "
             "previsões das runs 00/12Z anteriores (aprofundamento previsto vs. analisado)",
             fontsize=9, color="#1b263b", va="top")
    return _png(fig)


# ------------------------------------------------------------------ synoptic sequence
PR_LV = [0.5, 2, 5, 10, 15, 20, 30, 40, 60, 80]
PR_COL = ["#c6e9fb", "#6cb6ef", "#1f5fc7", "#14a33a", "#a5dc2b", "#f2ec2c", "#f99d1c",
          "#e2231a", "#b02fa3"]


def render_sequence(data: dict) -> bytes:
    panels = data["panels"]
    if not panels:
        raise ValueError("sem painéis")
    region = REGIONS[data["region"]]
    ncol = 3 if len(panels) > 4 else 2
    nrow = math.ceil(len(panels) / ncol)
    pw = 14.5 / ncol
    ph = pw * region.aspect + 0.35
    fig_h = nrow * ph + 1.5
    fig = Figure(figsize=(14.5, fig_h), dpi=100, facecolor="white")
    FigureCanvasAgg(fig)
    cmap = ListedColormap(PR_COL)
    norm = BoundaryNorm(PR_LV, cmap.N)
    land = cfeature.NaturalEarthFeature("physical", "land", "50m")
    coast = cfeature.NaturalEarthFeature("physical", "coastline", "50m")
    border = cfeature.NaturalEarthFeature("cultural", "admin_0_boundary_lines_land", "50m")
    stroke = [pe.withStroke(linewidth=2.4, foreground="white")]
    for k, p in enumerate(panels):
        r, c = divmod(k, ncol)
        left = c * pw / 14.5 + 0.004
        bottom = 1 - (1.15 + (r + 1) * ph) / fig_h
        ax = fig.add_axes([left, bottom, pw / 14.5 - 0.008, (ph - 0.35) / fig_h], projection=region.crs)
        ax.set_extent(region.proj_extent, crs=region.crs)
        ax.add_feature(land, facecolor="#efebe3", edgecolor="none", zorder=0)
        ax.set_facecolor("#d6e4ef")
        lats, lons = p["lats"], p["lons"]
        if p["pr"] is not None:
            ax.contourf(p["pr_lons"], p["pr_lats"], np.ma.masked_less(p["pr"], PR_LV[0]), levels=PR_LV,
                        cmap=cmap, norm=norm, extend="max", transform=PC, zorder=1)
        res = abs(float(lats[1] - lats[0])) if len(lats) > 1 else 0.25
        msl = gaussian_filter(p["msl"].astype(float), max(1.0, 0.25 / res))
        lv = np.arange(np.floor(np.nanmin(msl) / 4) * 4, np.nanmax(msl) + 4, 4)
        cs = ax.contour(lons, lats, msl, levels=lv, colors="#111", linewidths=0.7, transform=PC, zorder=3)
        ax.clabel(cs, fmt="%d", fontsize=6, inline_spacing=1)
        ax.add_feature(coast, facecolor="none", edgecolor="#333", linewidth=0.5, zorder=4)
        ax.add_feature(border, facecolor="none", edgecolor="#555", linewidth=0.4, zorder=4)
        for lon, lat, name, pc in p["centres"]:
            ax.text(lon, lat, "B", transform=PC, color="#b3001b", fontsize=12, fontweight="bold",
                    ha="center", va="center", zorder=6, path_effects=stroke, clip_on=True)
            ax.text(lon, lat, f"\n\n{name} {pc:.0f}", transform=PC, color="#b3001b", fontsize=7,
                    fontweight="bold", ha="center", va="center", zorder=6, path_effects=stroke,
                    clip_on=True)
        t = p["time"]
        ax.set_title(_fmt(t), fontsize=9.5, fontweight="bold", pad=3)
    fig.text(0.01, 1 - 0.25 / fig_h, f"Sequência sinótica — evento «{data['event']}» · {data['model']}",
             fontsize=13, fontweight="bold", color="#0d1b2a", va="top")
    fig.text(0.01, 1 - 0.6 / fig_h, "Isóbaras de 4 em 4 hPa (análises) · sombreado: precipitação nas 6 h "
             "anteriores (previsão 0–6 h) · B: centro seguido", fontsize=9, color="#1b263b", va="top")
    cax = fig.add_axes([0.62, 1 - 0.95 / fig_h, 0.36, 0.13 / fig_h])
    from matplotlib.cm import ScalarMappable
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                      ticks=PR_LV, extend="max")
    cb.ax.tick_params(labelsize=7)
    cb.set_label("mm / 6 h", fontsize=8)
    fig.text(0.99, 0.15 / fig_h, f"Dados: {data['source']} · Forecast Workstation", fontsize=7,
             color="#666", ha="right")
    return _png(fig)


# ------------------------------------------------------------------ point series
def render_point(data: dict) -> bytes:
    """data = {event, place, lat, lon, series: {model: point_chain dict}}"""
    panels = [("msl", "Pressão ao nível médio do mar", "hPa"),
              ("wind", "Vento médio a 10 m (contínuo) e rajada (pontilhado)", "km/h"),
              ("precip", "Intensidade média da precipitação em cada intervalo (barras) e acumulada (linhas)", "mm/h"),
              ("t2m", "Temperatura a 2 m", "°C")]
    fig_h = 2.3 * len(panels) + 1.4
    fig = Figure(figsize=(14.5, fig_h), dpi=110, facecolor="white")
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(len(panels), 1, hspace=0.12, left=0.06, right=0.95,
                          top=1 - 1.0 / fig_h, bottom=0.45 / fig_h)
    axes = []
    for i in range(len(panels)):
        axes.append(fig.add_subplot(gs[i], sharex=axes[0] if axes else None))
    times_all = []
    models = list(data["series"])
    ax2 = axes[2].twinx()
    width_h = {m: (1 if m in ("gfs", "icon_eu") else 3 if m == "ecmwf" else 6) for m in models}
    for k, (mid, s) in enumerate(data["series"].items()):
        t = [x if isinstance(x, dt.datetime) else _t(x) for x in s["time"]]
        if not t:
            continue
        times_all += t
        col = MODEL_COLORS.get(mid, "#333")
        axes[0].plot(t, s["msl"], color=col, lw=1.8, label=NAMES.get(mid, mid))
        axes[1].plot(t, s["wind"], color=col, lw=1.6)
        if np.isfinite(s["gust"]).any():
            axes[1].plot(t, s["gust"], color=col, lw=1.0, ls=":")
        pr = np.nan_to_num(np.array(s["precip"], dtype=float))
        hrs = np.array([h if h else width_h[mid] for h in s.get("precip_h", [width_h[mid]] * len(t))], float)
        rate = pr / hrs
        # each bar spans its accumulation interval (ending at t), so different models compare
        tn = mdates.date2num(t)
        left = tn - hrs / 24
        axes[2].bar(left, rate, width=hrs / 24 * 0.92, align="edge", color=col, alpha=0.35 + 0.5 / len(models),
                    edgecolor=col, linewidth=0.6)
        ax2.plot(t, np.cumsum(pr), color=col, lw=1.6)
        axes[3].plot(t, s["t2m"], color=col, lw=1.6)
    if not times_all:
        raise ValueError("sem dados")
    ax2.set_ylabel("acumulado (mm)", fontsize=8)
    ax2.tick_params(labelsize=7.5)
    for ax, (key, title, unit) in zip(axes, panels):
        ax.set_ylabel(unit, fontsize=8)
        _panel_label(ax, title)
    axes[0].legend(loc="upper right", fontsize=8, ncol=len(models), framealpha=0.92)
    _time_axis(axes, min(times_all), max(times_all), axes[0])
    axes[-1].set_xlabel("Hora (UTC)", fontsize=8)
    ns, ew = ("N" if data["lat"] >= 0 else "S"), ("E" if data["lon"] >= 0 else "W")
    loc = f"{data['place'] + ' · ' if data['place'] else ''}{abs(data['lat']):.2f}°{ns} {abs(data['lon']):.2f}°{ew}"
    fig.text(0.06, 1 - 0.22 / fig_h, f"Evolução num local — {loc} · evento «{data['event']}»",
             fontsize=13, fontweight="bold", color="#0d1b2a", va="top")
    fig.text(0.06, 1 - 0.52 / fig_h, "Cadeia de curto prazo: primeiras horas de cada run de 6 em 6 h "
             "(GFS/ICON-EU horário, ECMWF 3 h, AIFS 6 h) — o mais próximo de uma reanálise disponível em tempo real",
             fontsize=8.8, color="#1b263b", va="top")
    return _png(fig)
