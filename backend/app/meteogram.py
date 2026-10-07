"""Meteogram rendering: single-model technical meteogram and multi-model comparison."""
from __future__ import annotations

import datetime as dt
import io
import warnings

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
from matplotlib import patheffects as pe  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

warnings.filterwarnings("ignore")

WEEK = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
MON = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
MODEL_COLORS = {"gfs": "#2563eb", "ecmwf": "#dc2626", "aifs": "#16a34a", "icon_eu": "#9333ea"}
KMH = 3.6


def fmt_time(t):
    return f"{WEEK[t.weekday()]} {t.day:02d} {MON[t.month - 1]} {t:%H}Z"


def _wind_dir(u, v):
    return (np.degrees(np.arctan2(-u, -v)) + 360) % 360


def _time_axis(axes, t0, t1, top_ax):
    """Day shading, 00Z separators and weekday labels."""
    d = dt.datetime(t0.year, t0.month, t0.day)
    k = 0
    while d <= t1:
        nxt = d + dt.timedelta(days=1)
        for ax in axes:
            if k % 2 == 0:
                ax.axvspan(d, nxt, color="#f1f4f8", zorder=0, lw=0)
            ax.axvline(d, color="#9aa5b1", lw=0.8, zorder=1)
        mid = d + dt.timedelta(hours=12)
        if t0 <= mid <= t1:
            top_ax.text(mid, 1.02, f"{WEEK[d.weekday()]} {d.day:02d} {MON[d.month - 1]}",
                        transform=top_ax.get_xaxis_transform(), ha="center", va="bottom",
                        fontsize=8.5, fontweight="bold", color="#1b263b", clip_on=False)
        d, k = nxt, k + 1
    span_h = (t1 - t0).total_seconds() / 3600
    hours = [0, 12] if span_h > 200 else [0, 6, 12, 18]
    for ax in axes:
        ax.set_xlim(t0, t1)
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=hours))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
        ax.grid(True, axis="y", color="#cbd2d9", lw=0.5)
        ax.tick_params(labelsize=7.5, length=2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)


def _panel_label(ax, text, color="#0d1b2a"):
    ax.text(0.003, 0.97, text, transform=ax.transAxes, fontsize=8.6, fontweight="bold", va="top",
            color=color, zorder=20,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cbd2d9", lw=0.6, alpha=0.92))


def _interval_precip(tp):
    tp = np.where(np.isnan(tp), np.nan, tp)
    d = np.diff(tp, prepend=0.0)
    d[0] = 0.0
    return np.clip(d, 0, None)


def render_meteogram(data: dict, place: str) -> bytes:
    s = data["s"]
    t = np.array(data["valid"])
    tt = mdates.date2num(t)
    prof = data.get("profile")
    has_lmh = all(k in s and np.isfinite(s[k]).any() for k in ("lcc", "mcc", "hcc"))
    cape_key = next((k for k in ("mucape", "cape", "mlcape") if k in s and np.isfinite(s[k]).any()), None)

    panels = []
    if prof is not None:
        panels.append(("th", 3.4))
    panels += [("temp", 2.1), ("precip", 1.7), ("cloud", 0.75 if has_lmh else 0.45),
               ("wind", 1.8), ("msl", 1.3)]
    if cape_key:
        panels.append(("cape", 1.0))
    heights = [h for _, h in panels]
    fig_h = sum(heights) + 1.6
    fig = Figure(figsize=(14.5, fig_h), dpi=110, facecolor="white")
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(len(panels), 1, height_ratios=heights, hspace=0.09,
                          left=0.06, right=0.93, top=1 - 0.95 / fig_h, bottom=0.45 / fig_h)
    axes = {}
    first = None
    for i, (name, _) in enumerate(panels):
        axes[name] = fig.add_subplot(gs[i], sharex=first)
        first = first or axes[name]
    ax_list = [axes[n] for n, _ in panels]

    # ---------------------------------------------------- time-height section
    if prof is not None:
        ax = axes["th"]
        lv = np.array(data["levels"], dtype=float)
        rh = prof["r"]
        rh_lv = [60, 70, 80, 90, 95, 100.1]
        cmap = ListedColormap(["#d8f3dc", "#95d5b2", "#52b788", "#2d6a4f", "#1b4332"])
        ax.contourf(tt, lv, rh, levels=rh_lv, cmap=cmap, norm=BoundaryNorm(rh_lv, 5), zorder=2)
        tc = prof["t"] - 273.15
        tlev = np.arange(-60, 41, 4)
        cs = ax.contour(tt, lv, tc, levels=tlev, colors="#b91c1c", linewidths=0.8, zorder=3,
                        linestyles=["--" if x < 0 else "-" for x in tlev])
        ax.clabel(cs, fmt="%d", fontsize=6.5, inline_spacing=1)
        c0 = ax.contour(tt, lv, tc, levels=[0], colors="#1d4ed8", linewidths=2.0, zorder=4)
        ax.clabel(c0, fmt="0°C", fontsize=7)
        sp = s.get("sp")
        if sp is not None and np.isfinite(sp).any():
            ax.fill_between(tt, sp / 100.0, 1050, color="#8d6e63", alpha=0.85, zorder=5, lw=0)
        nb = max(1, int(np.ceil(len(t) / 55)))
        lsel = [i for i, p in enumerate(lv) if p in (1000, 950, 900, 850, 800, 700, 600, 500,
                                                       400, 300, 250, 200)]
        for i in lsel:
            ax.barbs(tt[::nb], np.full(len(tt[::nb]), lv[i]), prof["u"][i, ::nb] * 1.943844,
                     prof["v"][i, ::nb] * 1.943844, length=5.0, linewidth=0.45, zorder=6,
                     sizes={"emptybarb": 0.05}, clip_on=True)
        ax.set_yscale("log")
        ax.set_ylim(1030, lv.min() - 5)
        ticks = [1000, 925, 850, 700, 500, 400, 300, 200]
        ax.set_yticks([x for x in ticks if x >= lv.min()])
        ax.set_yticklabels([str(x) for x in ticks if x >= lv.min()])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylabel("hPa", fontsize=8)
        _panel_label(ax, "Secção tempo-altura: HR ≥60 % (verde), T (°C, vermelho; 0 °C azul), vento (kt)")

    # ---------------------------------------------------- temperature
    ax = axes["temp"]
    T2 = s["t2m"] - 273.15
    D2 = s["d2m"] - 273.15
    ax.plot(t, T2, color="#d00000", lw=2.0, label="T 2 m", zorder=5)
    ax.plot(t, D2, color="#008a3e", lw=1.6, label="Td 2 m", zorder=5)
    ax.fill_between(t, D2, T2, color="#fde2e4", alpha=0.6, zorder=2)
    if "t850" in s:
        ax.plot(t, s["t850"] - 273.15, color="#7c3aed", lw=1.2, ls="--", label="T 850 hPa", zorder=4)
    ax.axhline(0, color="#1d4ed8", lw=0.9, zorder=3)
    for i in _daily_extrema(t, T2):
        ax.annotate(f"{T2[i]:.0f}°", (t[i], T2[i]), textcoords="offset points",
                    xytext=(0, 5 if i in _max_idx(t, T2) else -11), ha="center", fontsize=7,
                    color="#9d0208", fontweight="bold")
    ax.set_ylabel("°C", fontsize=8)
    ax.legend(loc="upper right", fontsize=7, ncol=3, framealpha=0.9)
    _panel_label(ax, "Temperatura e ponto de orvalho")

    # ---------------------------------------------------- precipitation
    ax = axes["precip"]
    pr = _interval_precip(s["tp"])
    dts = np.diff(tt, prepend=tt[0] - (tt[1] - tt[0]) if len(tt) > 1 else 0.125)
    ax.bar(tt - dts / 2, pr, width=dts * 0.9, color="#2563eb", align="center", zorder=4)
    ax.set_ylabel("mm/intervalo", fontsize=8)
    ax.set_ylim(0, max(2.0, np.nanmax(pr) * 1.25 if np.isfinite(pr).any() else 2))
    ax2 = ax.twinx()
    ax2.plot(t, np.nan_to_num(s["tp"]), color="#0b3c8c", lw=1.4, ls="-", zorder=5)
    ax2.set_ylim(0, max(5, np.nanmax(s["tp"]) * 1.1 if np.isfinite(s["tp"]).any() else 5))
    ax2.set_ylabel("acumulado (mm)", fontsize=8, color="#0b3c8c")
    ax2.tick_params(labelsize=7.5, colors="#0b3c8c")
    ax2.spines["top"].set_visible(False)
    total = np.nanmax(s["tp"]) if np.isfinite(s["tp"]).any() else 0
    _panel_label(ax, f"Precipitação (barras: por intervalo · linha: acumulada, total {total:.1f} mm)")

    # ---------------------------------------------------- clouds
    ax = axes["cloud"]
    gray = ListedColormap(["#ffffff", "#e5e7eb", "#c7ccd3", "#9ca3af", "#6b7280", "#4b5563"])
    cl_norm = BoundaryNorm([0, 10, 30, 50, 70, 90, 100.1], 6)
    rows = [("hcc", "Altas"), ("mcc", "Médias"), ("lcc", "Baixas")] if has_lmh else [("tcc", "Total")]
    edges = np.concatenate([[tt[0] - (tt[1] - tt[0]) / 2], (tt[:-1] + tt[1:]) / 2,
                            [tt[-1] + (tt[-1] - tt[-2]) / 2]]) if len(tt) > 1 else tt
    for j, (k, lab) in enumerate(rows):
        arr = s.get(k, np.full(len(t), np.nan))
        ax.pcolormesh(edges, [j, j + 1], arr[None, :], cmap=gray, norm=cl_norm, zorder=2)
    ax.set_ylim(0, len(rows))
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels([r[1] for r in rows], fontsize=7)
    ax.grid(False)
    ax.set_title("Nebulosidade (%)", fontsize=8, loc="left", pad=2, fontweight="bold")

    # ---------------------------------------------------- wind
    ax = axes["wind"]
    spd = np.hypot(s["u10"], s["v10"]) * KMH
    ax.plot(t, spd, color="#0f766e", lw=1.8, label="Vento médio 10 m", zorder=5)
    if "gust" in s and np.isfinite(s["gust"]).any():
        ax.plot(t, s["gust"] * KMH, color="#ea580c", lw=1.2, ls="-", marker=".", ms=3,
                label="Rajada", zorder=5)
    top = np.nanmax([np.nanmax(spd), np.nanmax(s["gust"] * KMH) if "gust" in s else 0]) * 1.35 + 10
    ax.set_ylim(0, top)
    nb = max(1, int(np.ceil(len(t) / 60)))
    ax.barbs(tt[::nb], np.full(len(tt[::nb]), top * 0.88), s["u10"][::nb] * 1.943844,
             s["v10"][::nb] * 1.943844, length=5.2, linewidth=0.5, zorder=6,
             sizes={"emptybarb": 0.05})
    ax.set_ylabel("km/h", fontsize=8)
    ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.9, bbox_to_anchor=(1, 0.8))
    _panel_label(ax, "Vento a 10 m (barbelas em kt)")

    # ---------------------------------------------------- MSLP
    ax = axes["msl"]
    ax.plot(t, s["msl"] / 100.0, color="#111827", lw=1.6)
    ax.set_ylabel("hPa", fontsize=8)
    _panel_label(ax, "Pressão ao nível médio do mar")

    if cape_key:
        ax = axes["cape"]
        c = np.nan_to_num(s[cape_key])
        ax.bar(tt, c, width=(tt[1] - tt[0]) * 0.85 if len(tt) > 1 else 0.1,
               color=np.where(c > 1000, "#dc2626", np.where(c > 250, "#f59e0b", "#a3a3a3")), zorder=4)
        ax.set_ylim(0, max(300, c.max() * 1.2))
        ax.set_ylabel("J/kg", fontsize=8)
        _panel_label(ax, {"mucape": "MUCAPE", "cape": "CAPE (superfície)", "mlcape": "MLCAPE (90 hPa)"}[cape_key])

    _time_axis(ax_list, t[0], t[-1], ax_list[0])
    ax_list[-1].set_xlabel("Hora (UTC)", fontsize=8)

    run = data["run"]
    ns, ew = ("N" if data["lat"] >= 0 else "S"), ("E" if data["lon"] >= 0 else "W")
    loc = f"{place + ' · ' if place else ''}{abs(data['lat']):.2f}°{ns} {abs(data['lon']):.2f}°{ew}"
    if data.get("orog") is not None:
        loc += f" · altitude do modelo {data['orog']:.0f} m"
    fig.text(0.06, 1 - 0.22 / fig_h, f"Meteograma {data['model_name']}  —  {loc}", fontsize=13,
             fontweight="bold", color="#0d1b2a", va="top")
    fig.text(0.06, 1 - 0.5 / fig_h, f"Run {fmt_time(run)} {run.year}   ·   até +{data['steps'][-1]} h",
             fontsize=9.5, color="#1b263b", va="top")
    fig.text(0.93, 0.08 / fig_h, "Forecast Workstation", fontsize=7, color="#666", ha="right",
             style="italic")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor="white")
    return buf.getvalue()


def _days(t):
    keys = {}
    for i, x in enumerate(t):
        keys.setdefault((x.year, x.month, x.day), []).append(i)
    return [v for v in keys.values() if len(v) >= 4]


def _max_idx(t, y):
    return {max(ix, key=lambda i: -np.inf if np.isnan(y[i]) else y[i]) for ix in _days(t)}


def _daily_extrema(t, y):
    out = set(_max_idx(t, y))
    for ix in _days(t):
        out.add(min(ix, key=lambda i: np.inf if np.isnan(y[i]) else y[i]))
    return sorted(out)


# ============================================================ multi-model
def render_multi(datas: list[dict], place: str) -> bytes:
    panels = [("t2m", "Temperatura a 2 m", "°C"), ("t850", "Temperatura a 850 hPa", "°C"),
              ("precip", "Precipitação acumulada desde o início de cada run", "mm"),
              ("wind", "Vento médio a 10 m", "km/h"), ("msl", "Pressão ao nível médio do mar", "hPa"),
              ("gh500", "Geopotencial 500 hPa", "dam")]
    fig_h = 2.0 * len(panels) + 1.4
    fig = Figure(figsize=(14.5, fig_h), dpi=110, facecolor="white")
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(len(panels), 1, hspace=0.1, left=0.06, right=0.97,
                          top=1 - 1.0 / fig_h, bottom=0.45 / fig_h)
    axes = []
    for i in range(len(panels)):
        axes.append(fig.add_subplot(gs[i], sharex=axes[0] if axes else None))

    t0 = max(d["valid"][0] for d in datas)
    t1 = max(d["valid"][-1] for d in datas)
    for d in datas:
        s, t = d["s"], np.array(d["valid"])
        col = MODEL_COLORS.get(d["model"], "#333")
        lab = f"{d['model_name']} ({d['run']:%d/%m %H}Z)"
        series = {
            "t2m": s["t2m"] - 273.15,
            "t850": s.get("t850", np.full(len(t), np.nan)) - 273.15,
            "precip": s["tp"] - (np.interp(mdates.date2num(t0), mdates.date2num(t), s["tp"])
                                 if t[0] < t0 else 0),
            "wind": np.hypot(s["u10"], s["v10"]) * KMH,
            "msl": s["msl"] / 100.0,
            "gh500": s.get("gh500", np.full(len(t), np.nan)) / 10.0,
        }
        m = t >= t0
        for ax, (key, _, _) in zip(axes, panels):
            y = series[key]
            if key == "precip":
                y = np.clip(y, 0, None)
            ax.plot(t[m], y[m], color=col, lw=1.9 if d["model"] != "aifs" else 1.6,
                    ls="-" if d["model"] != "aifs" else (0, (5, 2)), label=lab, zorder=5)
            if key == "wind" and "gust" in s and np.isfinite(s["gust"]).any():
                ax.plot(t[m], (s["gust"] * KMH)[m], color=col, lw=0.8, ls=":", zorder=4)
    for ax, (key, title, unit) in zip(axes, panels):
        ax.set_ylabel(unit, fontsize=8)
        _panel_label(ax, title + (" (pontilhado: rajadas)" if key == "wind" else ""))
        if key in ("t2m", "t850"):
            ax.axhline(0, color="#1d4ed8", lw=0.8, zorder=3)
    axes[0].legend(loc="upper right", fontsize=8, ncol=len(datas), framealpha=0.92)
    _time_axis(axes, t0, t1, axes[0])
    axes[-1].set_xlabel("Hora (UTC)", fontsize=8)
    ns, ew = ("N" if datas[0]["lat"] >= 0 else "S"), ("E" if datas[0]["lon"] >= 0 else "W")
    loc = f"{place + ' · ' if place else ''}{abs(datas[0]['lat']):.2f}°{ns} {abs(datas[0]['lon']):.2f}°{ew}"
    fig.text(0.06, 1 - 0.22 / fig_h, f"Comparação de modelos  —  {loc}", fontsize=13,
             fontweight="bold", color="#0d1b2a", va="top")
    fig.text(0.06, 1 - 0.52 / fig_h, "   ·   ".join(
        f"{d['model_name']}: run {fmt_time(d['run'])}" for d in datas), fontsize=9.5,
        color="#1b263b", va="top")
    fig.text(0.97, 0.08 / fig_h, "Forecast Workstation", fontsize=7, color="#666", ha="right",
             style="italic")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor="white")
    return buf.getvalue()
