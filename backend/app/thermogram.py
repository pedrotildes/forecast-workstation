"""Heat/cold-wave thermogram for a point: daily Tmax/Tmin per model vs normals, ENS signal."""
from __future__ import annotations

import datetime as dt
import io

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .heatwave import MIN_DAYS, THRESH, fmt_day, verdict  # noqa: E402
from .meteogram import MODEL_COLORS, _panel_label, _time_axis, fmt_time  # noqa: E402


def _classify(v, warm):
    if v is None:
        return "sem sinal", "#6b7280"
    if v["n"] >= MIN_DAYS:
        return ("ONDA DE CALOR" if warm else "ONDA DE FRIO"), ("#b91c1c" if warm else "#1d4ed8")
    if v["n"] >= 3:
        return ("episódio quente" if warm else "episódio frio"), ("#ea580c" if warm else "#2563eb")
    return ("dias isolados quentes" if warm else "dias isolados frios"), "#6b7280"


def render_thermogram(data: dict, place: str) -> bytes:
    days = data["days"]
    if not days:
        raise ValueError("sem dias")
    x = [dt.datetime(d.year, d.month, d.day, 12) for d in days]
    ntx, ntn = np.array(data["ntmax"]), np.array(data["ntmin"])
    ens = data.get("ens")
    n_panels = 3 if ens else 2
    fig_h = 2.6 * n_panels + 2.6
    fig = Figure(figsize=(14.5, fig_h), dpi=110, facecolor="white")
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(n_panels, 1, hspace=0.12, left=0.06, right=0.97,
                          top=1 - 2.2 / fig_h, bottom=0.45 / fig_h,
                          height_ratios=[1.2, 1.2] + ([1.0] if ens else []))
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    axes = [ax1, ax2]
    if ens:
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        axes.append(ax3)

    def shade_days(ax, warm):
        for i, d in enumerate(days):
            hits = 0
            for m in data["models"]:
                if d in m["days"]:
                    j = m["days"].index(d)
                    v = (m["tmax"] if warm else m["tmin"])[j]
                    n = (ntx if warm else ntn)[i]
                    if np.isfinite(v) and np.isfinite(n) and (
                            (v - n) >= THRESH if warm else (v - n) <= -THRESH):
                        hits += 1
            if hits:
                ax.axvspan(x[i] - dt.timedelta(hours=12), x[i] + dt.timedelta(hours=12),
                           color="#fecaca" if warm else "#bfdbfe",
                           alpha=0.35 + 0.2 * hits, zorder=1, lw=0)

    for ax, key, norm, warm, title in ((ax1, "tmax", ntx, True, "Temperatura máxima diária (06–18 UTC)"),
                                       (ax2, "tmin", ntn, False, "Temperatura mínima diária (18–06 UTC)")):
        shade_days(ax, warm)
        ax.plot(x, norm, color="#111827", lw=1.6, label="Normal 1991–2020", zorder=4)
        thr = norm + (THRESH if warm else -THRESH)
        ax.plot(x, thr, color="#b91c1c" if warm else "#1d4ed8", lw=1.4, ls="--",
                label=f"Normal {'+' if warm else '−'} {THRESH:.0f} °C (limiar)", zorder=4)
        for m in data["models"]:
            xs = [dt.datetime(d.year, d.month, d.day, 12) for d in m["days"]]
            ax.plot(xs, m[key], color=MODEL_COLORS.get(m["id"], "#333"), lw=2.0, marker="o", ms=4,
                    ls="-" if m["id"] != "aifs" else (0, (5, 2)),
                    label=f"{m['name']}" + (" (T2m 6h)" if m["inst"] else ""), zorder=5)
        ax.set_ylabel("°C", fontsize=8)
        ax.legend(loc="upper right", fontsize=7.5, ncol=3, framealpha=0.92)
        _panel_label(ax, title + ("  ·  dias sombreados: algum modelo cumpre o critério"))

    if ens:
        ev = ens["valid"]
        w = 0.4
        ax3.bar(ev, ens["p_gt1"], width=w, color="#fca5a5", label="P(T850 > +1σ)", zorder=3)
        ax3.bar(ev, ens["p_gt2"], width=w, color="#b91c1c", label="P(T850 > +2σ)", zorder=4)
        ax3.bar(ev, -np.array(ens["p_lt1"]), width=w, color="#93c5fd", label="P(T850 < −1σ)", zorder=3)
        ax3.bar(ev, -np.array(ens["p_lt2"]), width=w, color="#1d4ed8", label="P(T850 < −2σ)", zorder=4)
        ax3.axhline(0, color="#111", lw=0.8)
        for y in (50, -50):
            ax3.axhline(y, color="#555", lw=0.7, ls=":")
        ax3.set_ylim(-100, 100)
        ax3.set_yticks([-100, -50, 0, 50, 100])
        ax3.set_yticklabels(["100", "50", "0", "50", "100"])
        ax3.set_ylabel("% membros", fontsize=8)
        ax3.legend(loc="upper right", fontsize=7.2, ncol=4, framealpha=0.92)
        _panel_label(ax3, f"ECMWF ENS (run {fmt_time(ens['run'])}): anomalia padronizada de T850 "
                          "(quente ↑ / frio ↓)")
    t0 = x[0] - dt.timedelta(hours=12)
    t1 = x[-1] + dt.timedelta(hours=12)
    if ens:
        t1 = max(t1, ens["valid"][-1])
    _time_axis(axes, t0, t1, axes[0])
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))

    # ---- verdict box
    ns, ew = ("N" if data["lat"] >= 0 else "S"), ("E" if data["lon"] >= 0 else "W")
    loc = f"{place + ' · ' if place else ''}{abs(data['lat']):.2f}°{ns} {abs(data['lon']):.2f}°{ew}"
    fig.text(0.06, 1 - 0.22 / fig_h, f"Vagas de calor e de frio  —  {loc}", fontsize=13,
             fontweight="bold", color="#0d1b2a", va="top")
    fig.text(0.06, 1 - 0.50 / fig_h,
             f"Critério IPMA/OMM: ≥ {MIN_DAYS} dias consecutivos com Tmáx ≥ normal + {THRESH:.0f} °C "
             f"(calor) ou Tmín ≤ normal − {THRESH:.0f} °C (frio). Normais CPC 1991–2020.",
             fontsize=9, color="#1b263b", va="top")
    y = 1 - 0.85 / fig_h
    if not np.isfinite(ntx).any():
        fig.text(0.06, y, "Ponto sobre o mar: não há normais climatológicas (CPC só cobre terra).",
                 fontsize=10, color="#b91c1c", va="top", fontweight="bold")
    for m in data["models"]:
        idx = [days.index(d) for d in m["days"]]
        hv = verdict(m["days"], m["tmax"], ntx[idx], True)
        cv = verdict(m["days"], m["tmin"], ntn[idx], False)
        hl, hc = _classify(hv, True)
        cl, cc = _classify(cv, False)

        def desc(v):
            if not v:
                return ""
            s = f" — {v['n']} d, {v['d0']:%d/%m}→{v['d1']:%d/%m}"
            return s + (" (pode continuar)" if v["open"] else "")
        fig.text(0.06, y, f"{m['name']} ({m['run']:%d/%m %H}Z):", fontsize=9.5, fontweight="bold",
                 color=MODEL_COLORS.get(m["id"], "#333"), va="top")
        fig.text(0.22, y, f"Calor: {hl}{desc(hv)}", fontsize=9.5, color=hc, va="top",
                 fontweight="bold" if hv and hv["n"] >= MIN_DAYS else "normal")
        fig.text(0.62, y, f"Frio: {cl}{desc(cv)}", fontsize=9.5, color=cc, va="top",
                 fontweight="bold" if cv and cv["n"] >= MIN_DAYS else "normal")
        y -= 0.26 / fig_h
    if ens:
        pg = np.nanmax(ens["p_gt1"]) if np.isfinite(ens["p_gt1"]).any() else np.nan
        pl = np.nanmax(ens["p_lt1"]) if np.isfinite(ens["p_lt1"]).any() else np.nan
        ig, il = int(np.nanargmax(ens["p_gt1"])), int(np.nanargmax(ens["p_lt1"]))
        fig.text(0.06, y, "ECMWF ENS:", fontsize=9.5, fontweight="bold", color="#111", va="top")
        fig.text(0.22, y, f"P máx(T850 > +1σ) = {pg:.0f}% ({fmt_time(ens['valid'][ig])})",
                 fontsize=9.5, color="#b91c1c", va="top")
        fig.text(0.62, y, f"P máx(T850 < −1σ) = {pl:.0f}% ({fmt_time(ens['valid'][il])})",
                 fontsize=9.5, color="#1d4ed8", va="top")
    fig.text(0.97, 0.08 / fig_h, "Forecast Workstation", fontsize=7, color="#666", ha="right",
             style="italic")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor="white")
    return buf.getvalue()
