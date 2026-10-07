"""Multi-model forecast sounding: profiles overlaid on one Skew-T + indices table."""
from __future__ import annotations

import io

import numpy as np
from matplotlib import patheffects as pe
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from metpy.plots import Hodograph
from metpy.units import units

from ..meteogram import MODEL_COLORS
from .skewt import _f, skewt_background


def render_skewt_multi(items: list[dict], title: str, subtitle: str) -> bytes:
    fig = Figure(figsize=(14.5, 10.2), dpi=110, facecolor="white")
    FigureCanvasAgg(fig)
    skew = skewt_background(fig, (0.055, 0.07, 0.52, 0.84))
    ax = skew.ax
    stroke = [pe.withStroke(linewidth=3.2, foreground="white")]

    hax = fig.add_axes((0.70, 0.52, 0.28, 0.38))
    rmax = 30
    for it in items:
        prof = it["snd"]["profile"]
        z = prof["z"] - prof["z"][0]
        m = z <= 12000
        rmax = max(rmax, float(np.nanmax(np.hypot(prof["u"][m], prof["v"][m]))))
    rmax = int(np.ceil(rmax / 10) * 10) + 10
    h = Hodograph(hax, component_range=rmax)
    h.add_grid(increment=10, color="#bbb", lw=0.5)
    hax.set_facecolor("#fbfbf8")

    for k, it in enumerate(items):
        prof = it["snd"]["profile"]
        col = MODEL_COLORS.get(it["model"], "#333")
        p = prof["p"] * units.hPa
        lab = f"{it['name']} ({it['run']:%d/%m %H}Z +{it['step']}h)"
        skew.plot(p, prof["t"] * units.degC, color=col, lw=2.3, zorder=8, path_effects=stroke,
                  label=lab)
        skew.plot(p, prof["td"] * units.degC, color=col, lw=1.7, ls=(0, (5, 2)), zorder=8,
                  path_effects=stroke)
        tr = it["an"].get("_parcel_traces", {}).get("ML")
        if tr is not None:
            skew.plot(p, tr * units.degC, color=col, lw=0.9, ls=":", zorder=6)
        sel = prof["p"] >= 100
        xloc = 1.03 + 0.04 * k
        skew.plot_barbs(p[sel], prof["u"][sel] * units.knot, prof["v"][sel] * units.knot,
                        xloc=xloc, length=5.2, linewidth=0.7, color=col,
                        sizes={"emptybarb": 0.06})
        z = prof["z"] - prof["z"][0]
        m = z <= 12000
        hax.plot(prof["u"][m], prof["v"][m], color=col, lw=2.0, label=it["name"])
        for km in (1, 3, 6):
            if km * 1000 <= z[m][-1]:
                hax.plot(np.interp(km * 1000, z, prof["u"]), np.interp(km * 1000, z, prof["v"]),
                         "o", ms=3.5, color=col)
        rm = it["an"].get("kinematics", {}).get("rm")
        if rm and None not in rm:
            hax.plot(*rm, marker="s", ms=6, color=col, mfc="white", mew=1.6)

    ax.legend(loc="upper left", fontsize=8, framealpha=0.92,
              title="T (contínuo) · Td (tracejado) · parcela ML (pontilhado)", title_fontsize=7.5)
    hax.legend(loc="upper left", fontsize=7, framealpha=0.85)
    hax.set_title("Hodógrafas 0–12 km (kt) · ■ Bunkers RM", fontsize=9.5, fontweight="bold",
                  loc="left")
    hax.tick_params(labelsize=7)
    hax.set_xlabel("")
    hax.set_ylabel("")

    # ---- indices table
    tax = fig.add_axes((0.70, 0.05, 0.28, 0.42))
    tax.axis("off")
    tax.set_xlim(0, 1)
    tax.set_ylim(0, 1)
    rows = [
        ("MLCAPE (J/kg)", lambda a: _f(a["parcels"].get("ML", {}).get("cape"))),
        ("MLCIN (J/kg)", lambda a: _f(a["parcels"].get("ML", {}).get("cin"))),
        ("MUCAPE (J/kg)", lambda a: _f(a["parcels"].get("MU", {}).get("cape"))),
        ("SBCAPE (J/kg)", lambda a: _f(a["parcels"].get("SB", {}).get("cape"))),
        ("LI ML (°C)", lambda a: _f(a["parcels"].get("ML", {}).get("li"), "{:.1f}")),
        ("LCL ML (m)", lambda a: _f(a["parcels"].get("ML", {}).get("lcl_z"))),
        ("K", lambda a: _f(a["indices"].get("k"))),
        ("Total Totals", lambda a: _f(a["indices"].get("tt"))),
        ("Água precip. (mm)", lambda a: _f(a["indices"].get("pw"), "{:.1f}")),
        ("Iso 0 °C (m)", lambda a: _f(a["indices"].get("frz_msl"))),
        ("Γ 700–500 (°C/km)", lambda a: _f(a["indices"].get("lr_700_500"), "{:.1f}")),
        ("Shear 0–6 km (kt)", lambda a: _f(a["kinematics"].get("shear_0_6"))),
        ("SRH 0–3 km (m²/s²)", lambda a: _f(a["kinematics"].get("srh_0_3"))),
    ]
    n = len(items)
    colx = [0.44 + i * (0.56 / n) + 0.56 / n - 0.02 for i in range(n)]
    y = 0.98
    for i, it in enumerate(items):
        tax.text(colx[i], y, it["short"], ha="right", va="top", fontsize=8.6, fontweight="bold",
                 color=MODEL_COLORS.get(it["model"], "#333"))
    y -= 0.065
    for name, fn in rows:
        tax.text(0.0, y, name, fontsize=8.3, va="top", fontweight="bold")
        for i, it in enumerate(items):
            tax.text(colx[i], y, fn(it["an"]), fontsize=8.3, va="top", ha="right",
                     family="DejaVu Sans Mono")
        y -= 0.06

    fig.text(0.055, 0.965, title, fontsize=13.5, fontweight="bold", color="#0d1b2a", va="center")
    fig.text(0.055, 0.935, subtitle, fontsize=10, color="#1b263b", va="center")
    fig.text(0.975, 0.012, "Forecast Workstation", fontsize=7, color="#666", ha="right",
             style="italic")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor="white")
    return buf.getvalue()
