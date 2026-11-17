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
NAMES = {"gfs": "GFS", "ecmwf": "ECMWF IFS", "aifs": "ECMWF AIFS", "icon_eu": "ICON-EU", "era5": "ERA5"}


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
    gs = fig.add_gridspec(n, 1, hspace=0.32, left=0.06, right=0.9, top=1 - 1.0 / fig_h,
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
        leads = data.get("leads", {}).get(sysname, {})
        for k, (r, tr) in enumerate(sorted(fcs.items())):
            shade = 0.35 + 0.55 * (k + 1) / max(1, len(fcs))
            ax.plot([_t(p["time"]) for p in tr], [p["p"] for p in tr], color=cmap(shade), lw=1.0,
                    ls=(0, (4, 2)), zorder=3,
                    label=f"previsões {NAMES.get(data['fc_model'], '')} (mais escuro = mais recente)" if k == len(fcs) - 1 else None)
            if tr and r in leads:
                rt = _t(r)
                ax.annotate(f"run {rt.day:02d}/{rt:%H}Z (−{leads[r]} h)", xy=(_t(tr[-1]["time"]), tr[-1]["p"]),
                            xytext=(4, 0), textcoords="offset points", fontsize=7, color=cmap(shade),
                            va="center", zorder=4)
        ob = data.get("observed", {}).get(sysname)
        if ob:
            ax.plot([_t(p["time"]) for p in ob], [p["p"] for p in ob], color="#000", lw=2.8, marker="o", ms=4,
                    zorder=6, label="NHC: best track (observado)")
            j = int(np.argmin([p["p"] for p in ob]))
            ax.annotate(f"NHC: {ob[j]['p']:.0f} hPa · {_fmt(_t(ob[j]['time']))}", xy=(0.012, 0.38),
                        xycoords="axes fraction", fontsize=7.8, color="#000", fontweight="bold")
        fz = data.get("focus", {}).get(sysname)
        if fz:
            ax.axvline(_t(fz), color="#b45309", lw=1.0, ls=":", zorder=2)
            ax.text(_t(fz), 0.98, " referência das previsões", transform=ax.get_xaxis_transform(),
                    fontsize=7.5, color="#b45309", va="top")
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


# ------------------------------------------------------------------ case meteogram
def render_case_meteogram(data: dict) -> bytes:
    prof, chain = data["profile"], data["chain"]
    if prof is None and not chain["time"]:
        raise ValueError("sem dados")
    panels = ([("th", 3.6)] if prof is not None else []) + [("msl", 1.4), ("wind", 1.7),
                                                             ("precip", 1.7), ("t2m", 1.4)]
    fig_h = sum(h for _, h in panels) + 1.6
    fig = Figure(figsize=(14.5, fig_h), dpi=110, facecolor="white")
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(len(panels), 1, height_ratios=[h for _, h in panels], hspace=0.1,
                          left=0.06, right=0.93, top=1 - 1.0 / fig_h, bottom=0.45 / fig_h)
    axes, first = {}, None
    for i, (name, _) in enumerate(panels):
        axes[name] = fig.add_subplot(gs[i], sharex=first)
        first = first or axes[name]
    ax_list = [axes[n] for n, _ in panels]
    col = MODEL_COLORS.get(data["model_id"], "#333")
    times_all = list(chain["time"]) + list(data["times"])

    if prof is not None:
        ax = axes["th"]
        tt = mdates.date2num(data["times"])
        lv = np.array(data["levels"], dtype=float)
        rh_lv = [60, 70, 80, 90, 95, 100.1]
        ax.contourf(tt, lv, prof["r"], levels=rh_lv,
                    cmap=ListedColormap(["#d8f3dc", "#95d5b2", "#52b788", "#2d6a4f", "#1b4332"]),
                    norm=BoundaryNorm(rh_lv, 5), zorder=2)
        tc = prof["t"] - 273.15
        tlev = np.arange(-60, 41, 4)
        cs = ax.contour(tt, lv, tc, levels=tlev, colors="#b91c1c", linewidths=0.8, zorder=3,
                        linestyles=["--" if x < 0 else "-" for x in tlev])
        ax.clabel(cs, fmt="%d", fontsize=6.5, inline_spacing=1)
        c0 = ax.contour(tt, lv, tc, levels=[0], colors="#1d4ed8", linewidths=2.0, zorder=4)
        ax.clabel(c0, fmt="0°C", fontsize=7)
        if np.isfinite(prof["sp"]).any():
            ax.fill_between(tt, prof["sp"], 1050, color="#8d6e63", alpha=0.85, zorder=5, lw=0)
        for i, p in enumerate(lv):
            if p in (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200) or len(lv) <= 14:
                ax.barbs(tt, np.full(len(tt), p), prof["u"][i] * 1.943844, prof["v"][i] * 1.943844,
                         length=5.2, linewidth=0.5, zorder=6, sizes={"emptybarb": 0.05})
        for t in data["obs"]["times"]:
            ax.axvline(t, color="#111", lw=0.8, ls=":", zorder=7)
            ax.text(t, 0.995, "▼", transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=8, color="#111", zorder=8)
        ax.set_yscale("log")
        ax.set_ylim(1030, lv.min() - 5)
        ticks = [1000, 925, 850, 700, 500, 400, 300, 200]
        ax.set_yticks([x for x in ticks if x >= lv.min()])
        ax.set_yticklabels([str(x) for x in ticks if x >= lv.min()])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylabel("hPa", fontsize=8)
        extra = (f" · ▼ radiossonda de {data['obs']['station']} ({data['obs']['km']} km)"
                 if data["obs"]["times"] else "")
        _panel_label(ax, "Análises de 6 em 6 h: HR ≥60 % (verde), T (°C; 0 °C azul), vento (kt)" + extra)

    t = chain["time"]
    if t:
        axes["msl"].plot(t, chain["msl"], color=col, lw=1.8)
        axes["wind"].plot(t, chain["wind"], color="#0f766e", lw=1.8, label="Vento médio 10 m")
        if np.isfinite(chain["gust"]).any():
            axes["wind"].plot(t, chain["gust"], color="#ea580c", lw=1.2, marker=".", ms=3, label="Rajada")
        axes["wind"].legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.9)
        pr = np.nan_to_num(np.array(chain["precip"], dtype=float))
        hrs = np.array([h or 1 for h in chain["precip_h"]], float)
        tn = mdates.date2num(t)
        axes["precip"].bar(tn - hrs / 24, pr / hrs, width=hrs / 24 * 0.92, align="edge", color="#2563eb")
        ax2 = axes["precip"].twinx()
        ax2.plot(t, np.cumsum(pr), color="#0b3c8c", lw=1.5)
        ax2.set_ylabel("acumulado (mm)", fontsize=8, color="#0b3c8c")
        ax2.tick_params(labelsize=7.5, colors="#0b3c8c")
        axes["t2m"].plot(t, chain["t2m"], color="#d00000", lw=1.8)
        tot = float(np.nansum(pr))
        _panel_label(axes["precip"], f"Precipitação: intensidade (mm/h) e acumulada — total {tot:.0f} mm")
    _panel_label(axes["msl"], "Pressão ao nível médio do mar (hPa)")
    _panel_label(axes["wind"], "Vento a 10 m e rajada (km/h)")
    _panel_label(axes["t2m"], "Temperatura a 2 m (°C)")
    _time_axis(ax_list, min(times_all), max(times_all), ax_list[0])
    ax_list[-1].set_xlabel("Hora (UTC)", fontsize=8)
    ns, ew = ("N" if data["lat"] >= 0 else "S"), ("E" if data["lon"] >= 0 else "W")
    loc = f"{data['place'] + ' · ' if data['place'] else ''}{abs(data['lat']):.2f}°{ns} {abs(data['lon']):.2f}°{ew}"
    fig.text(0.06, 1 - 0.22 / fig_h, f"Meteograma do caso «{data['event']}» — {data['model']} · {loc}",
             fontsize=13, fontweight="bold", color="#0d1b2a", va="top")
    fig.text(0.06, 1 - 0.52 / fig_h, "Secção tempo-altura das análises arquivadas; superfície pela cadeia "
             "de curto prazo (primeiras horas de cada run de 6 em 6 h)", fontsize=9, color="#1b263b", va="top")
    fig.text(0.93, 0.08 / fig_h, f"Dados: {data['source']} · Forecast Workstation", fontsize=7,
             color="#666", ha="right")
    return _png(fig)


# ------------------------------------------------------------------ NHC / ATCF forecast verification
PLACES_PT = [(36.98, "Sagres"), (37.95, "Sines"), (38.70, "Lisboa"), (39.36, "Peniche"), (39.60, "Nazaré"),
             (40.15, "Figueira da Foz"), (40.64, "Aveiro"), (41.15, "Porto"), (41.69, "Viana do Castelo"),
             (42.88, "Finisterra"), (43.37, "Corunha"), (48.45, "Brest"), (49.96, "Lizard Point"),
             (51.45, "Mizen Head"), (51.90, "Cork"), (52.27, "Tralee"), (52.66, "Limerick"), (53.27, "Galway"),
             (54.27, "Sligo"), (54.65, "Donegal"), (55.37, "Malin Head")]


def render_atcf_tracks(data: dict) -> bytes:
    """data = {storm, name, best: [pts], observed: {lat, lon, time}, tracks: [{base, tech, label,
    colour, ls, pts, landfall}], extent: [lon0, lon1, lat0, lat1]}"""
    lon0, lon1, lat0, lat1 = data["extent"]
    proj = ccrs.LambertConformal(central_longitude=(lon0 + lon1) / 2, central_latitude=(lat0 + lat1) / 2,
                                 standard_parallels=(lat0 + 3, lat1 - 3))
    # figure height from the map's aspect ratio (map 66 % of the width)
    xs, ys = zip(*[proj.transform_point(lo, la, PC) for lo, la in
                   ((lon0, lat0), (lon1, lat0), (lon0, lat1), (lon1, lat1), ((lon0 + lon1) / 2, lat1))])
    asp = (max(ys) - min(ys)) / (max(xs) - min(xs))
    fig_h = min(11.5, max(6.5, 14.5 * 0.66 * asp + 1.6))
    fig = Figure(figsize=(14.5, fig_h), dpi=110, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0.03, 0.5 / fig_h, 0.66, 1 - 1.6 / fig_h], projection=proj)
    ax.set_extent([lon0, lon1, lat0, lat1], PC)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#efe9dc", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#dbe9f4", edgecolor="none", zorder=0)
    ax.coastlines("50m", lw=0.7, color="#333", zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.5, edgecolor="#555", zorder=2)
    gl = ax.gridlines(draw_labels=True, lw=0.4, color="#999", alpha=0.6, x_inline=False, y_inline=False)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 8}
    bases = sorted({tr["base"] for tr in data["tracks"]})
    shades = {b: c for b, c in zip(bases, _shades(len(bases)))}
    for tr in data["tracks"]:
        pts = tr["pts"]
        col = tr["colour"] if len(bases) == 1 else shades[tr["base"]]
        ls = tr["ls"] if len(bases) == 1 else ("-" if tr["tech"] == "OFCL" else "--")
        ax.plot([p["lon"] for p in pts], [p["lat"] for p in pts], color=col, lw=1.8 if tr["tech"] == "OFCL" else 1.2,
                ls=ls, transform=PC, zorder=4)
        lf = tr.get("landfall")
        if lf:
            ax.plot(lf["lon"], lf["lat"], marker="X", ms=9, mec="white", mew=0.8, color=col, transform=PC, zorder=6)
    b = data["best"]
    ax.plot([p["lon"] for p in b], [p["lat"] for p in b], color="#000", lw=3.2, transform=PC, zorder=5)
    for p in b:
        if p["time"].hour in (0, 12):
            ax.plot(p["lon"], p["lat"], "o", ms=5, color="#000", transform=PC, zorder=6)
            if lon0 < p["lon"] < lon1 and lat0 < p["lat"] < lat1:
                ax.text(p["lon"] + 0.25, p["lat"] + 0.25, f"{p['time'].day:02d}/{p['time']:%H}Z {p['vmax']} kt",
                        transform=PC, fontsize=7.5, fontweight="bold", zorder=7,
                        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    o = data.get("observed")
    if o:
        ax.plot(o["lon"], o["lat"], marker="*", ms=20, color="#facc15", mec="#000", mew=1.2, transform=PC, zorder=8)
    # legend / info
    X, dy = 0.71, 0.27 / fig_h
    y = 1 - 1.15 / fig_h
    fig.text(X, y, "Trajetória observada (best track do NHC)", fontsize=9, fontweight="bold", va="top")
    y -= dy
    if o:
        fig.text(X, y, f"★ chegada a terra: {o['lat']:.2f}°N {abs(o['lon']):.2f}°W, "
                 f"{o['time'].day:02d}/{o['time']:%m} {o['time']:%H:%M} UTC", fontsize=8.5, va="top")
        y -= 1.4 * dy
    if len(bases) == 1:
        fig.text(X, y, f"Previsões feitas a {_fmt(bases[0])} (✕ = chegada a terra)", fontsize=9,
                 fontweight="bold", va="top")
        y -= dy
        seen = set()
        for tr in data["tracks"]:
            if tr["tech"] in seen:
                continue
            seen.add(tr["tech"])
            lf = tr.get("landfall")
            txt = f"{tr['label']}: " + (f"{lf['lat']:.2f}°N {abs(lf['lon']):.2f}°W · {lf['time'].day:02d} "
                                         f"{lf['time']:%H:%M}" if lf else "sem chegada a terra")
            fig.text(X, y, "━━ ", color=tr["colour"], fontsize=9, va="top", fontweight="bold")
            fig.text(X + 0.03, y, txt, fontsize=8, va="top")
            y -= dy
    else:
        fig.text(X, y, "Previsões por hora de emissão", fontsize=9, fontweight="bold", va="top")
        y -= dy
        fig.text(X, y, "✕ = chegada a terra prevista", fontsize=8, va="top", color="#444")
        y -= dy
        for bb in bases:
            ofcl = next((t for t in data["tracks"] if t["base"] == bb and t["tech"] == "OFCL"), None)
            lf = ofcl.get("landfall") if ofcl else None
            lead = (o["time"] - bb).total_seconds() / 3600 if o else 0
            txt = f"{_fmt(bb)} (−{lead:.0f} h)" + (f" · NHC: {lf['lat']:.2f}°N" if lf else "")
            fig.text(X, y, "━━ ", color=shades[bb], fontsize=9, va="top", fontweight="bold")
            fig.text(X + 0.03, y, txt, fontsize=8, va="top")
            y -= dy
        y -= 0.3 * dy
        fig.text(X, y, "contínuo = NHC (oficial) · tracejado = GFS", fontsize=8, va="top", color="#444")
    fig.text(0.03, 1 - 0.25 / fig_h, f"{data['name']}: trajetória observada e previsões do NHC e dos modelos",
             fontsize=14, fontweight="bold", color="#0d1b2a", va="top")
    fig.text(0.03, 1 - 0.6 / fig_h, data.get("subtitle", ""), fontsize=9.5, color="#1b263b", va="top")
    fig.text(0.97, 0.015, f"Dados: NHC/NOAA, arquivo ATCF ({data['storm']}) · Forecast Workstation",
             fontsize=7.5, color="#666", ha="right")
    return _png(fig)


def _shades(n):
    from matplotlib import colormaps
    cm = colormaps["plasma"]
    return [cm(0.08 + 0.8 * i / max(1, n - 1)) for i in range(n)]


def render_atcf_landfall(data: dict) -> bytes:
    """data = {name, storm, observed, rows: [{base, tech, lead_h, lat, time}], techs: {tech: (label, colour, ls)}}"""
    o = data["observed"]
    fig = Figure(figsize=(14.5, 8.4), dpi=110, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0.07, 0.36, 0.72, 0.52])
    ax2 = fig.add_axes([0.07, 0.08, 0.72, 0.20], sharex=ax)
    rows = data["rows"]
    for tech, (label, col, ls) in data["techs"].items():
        rs = sorted([r for r in rows if r["tech"] == tech], key=lambda r: -r["lead_h"])
        if not rs:
            continue
        lw = 2.6 if tech == "OFCL" else 1.4
        ax.plot([-r["lead_h"] for r in rs], [r["lat"] for r in rs], color=col, ls=ls, lw=lw, marker="o",
                ms=4 if tech != "OFCL" else 6, label=label, zorder=5 if tech == "OFCL" else 3)
        ax2.plot([-r["lead_h"] for r in rs], [(r["time"] - o["time"]).total_seconds() / 3600 for r in rs],
                 color=col, ls=ls, lw=lw, marker="o", ms=3.5, zorder=5 if tech == "OFCL" else 3)
    ax.axhline(o["lat"], color="#b45309", lw=2, zorder=2)
    ax.text(0.005, o["lat"] + 0.05, f"observado: {o['lat']:.2f}°N", transform=ax.get_yaxis_transform(),
            color="#b45309", fontsize=9, fontweight="bold", va="bottom")
    lats = [r["lat"] for r in rows if r["tech"] in data["techs"]] + [o["lat"]]
    lo, hi = min(lats) - 0.4, max(lats) + 0.4
    ax.set_ylim(lo, hi)
    for la, name in PLACES_PT:
        if lo < la < hi:
            ax.axhline(la, color="#000", alpha=0.08, lw=0.8, zorder=1)
            ax.text(1.005, la, name, transform=ax.get_yaxis_transform(), fontsize=8, va="center", color="#444")
    ax.set_ylabel("Latitude prevista da chegada a terra (°N)", fontsize=9)
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, ncol=3, framealpha=0.92)
    ax2.axhline(0, color="#b45309", lw=1.5)
    ax2.set_ylabel("Erro da hora (h)", fontsize=9)
    ax2.set_xlabel("Antecedência em relação à chegada a terra observada (h)", fontsize=9)
    ax2.grid(True, alpha=0.25)
    ax.tick_params(labelbottom=False)
    fig.text(0.07, 0.965, f"{data['name']}: onde e quando as previsões punham a chegada a terra",
             fontsize=14, fontweight="bold", color="#0d1b2a", va="top")
    fig.text(0.07, 0.93, f"Observado (best track do NHC): {o['lat']:.2f}°N {abs(o['lon']):.2f}°W às "
             f"{o['time']:%H:%M} UTC de {o['time'].day:02d}/{o['time']:%m}/{o['time']:%Y} · cada ponto é uma "
             f"previsão, situada pela antecedência", fontsize=9.5, color="#1b263b", va="top")
    fig.text(0.97, 0.015, f"Dados: NHC/NOAA, arquivo ATCF ({data['storm']}) · Forecast Workstation",
             fontsize=7.5, color="#666", ha="right")
    return _png(fig)
