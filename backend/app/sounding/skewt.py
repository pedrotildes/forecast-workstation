"""RAOB-style Skew-T log-p diagram with hodograph and indices panel."""
from __future__ import annotations

import io
import warnings

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import patheffects as pe  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from metpy.plots import Hodograph, SkewT  # noqa: E402
from metpy.units import units  # noqa: E402

warnings.filterwarnings("ignore")

HODO_COLORS = [(0, 1000, "#d0006f", "0–1 km"), (1000, 3000, "#e8410b", "1–3 km"),
               (3000, 6000, "#2b9348", "3–6 km"), (6000, 9000, "#c9a227", "6–9 km"),
               (9000, 12000, "#1d4ed8", "9–12 km")]


def _f(x, fmt="{:.0f}", unit=""):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return fmt.format(x) + unit


def _p_at_height(prof, zagl):
    z = prof["z"] - prof["z"][0]
    if zagl > z[-1]:
        return None
    return float(np.exp(np.interp(zagl, z, np.log(prof["p"]))))


def skewt_background(fig, rect) -> SkewT:
    """Skew-T axes with dry/moist adiabats, mixing-ratio lines and key isotherms."""
    skew = SkewT(fig, rotation=45, rect=rect)
    ax = skew.ax
    ax.set_ylim(1050, 100)
    ax.set_xlim(-40, 50)
    ax.set_facecolor("#fbfbf8")

    # ---- background lines
    pp = np.linspace(1050, 100, 60) * units.hPa
    skew.plot_dry_adiabats(t0=np.arange(-40, 200, 10) * units.degC, pressure=pp,
                           colors="#c8a27a", linewidths=0.6, alpha=0.75, linestyles="-")
    skew.plot_moist_adiabats(t0=np.arange(-20, 44, 4) * units.degC, pressure=pp,
                             colors="#4f9d69", linewidths=0.6, alpha=0.7, linestyles="--")
    mr = np.array([0.4, 1, 2, 3, 5, 8, 12, 16, 20, 28]).reshape(-1, 1) / 1000.0
    skew.plot_mixing_lines(mixing_ratio=mr, pressure=np.linspace(1050, 600, 20) * units.hPa,
                           colors="#7b5ea7", linewidths=0.6, alpha=0.7, linestyles=":")
    for w in mr.ravel():
        tx = (np.log((w * 600 / (0.622 + w)) / 6.112) * 243.5 /
              (17.67 - np.log((w * 600 / (0.622 + w)) / 6.112)))
        ax.text(tx, 600, f"{w * 1000:g}", fontsize=6.5, color="#7b5ea7", ha="center", va="bottom",
                transform=ax.transData, clip_on=True)
    for iso, col in ((0, "#1d4ed8"), (-20, "#0ea5e9")):
        ax.axvline(iso, color=col, lw=1.1, ls="-", alpha=0.85)
    ax.grid(True, which="major", axis="both", color="#999", lw=0.5, alpha=0.5)
    ax.set_yticks([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100])
    ax.set_yticklabels(["1000", "925", "850", "700", "600", "500", "400", "300", "250",
                        "200", "150", "100"])
    ax.set_xlabel("Temperatura (°C)", fontsize=9)
    ax.set_ylabel("Pressão (hPa)", fontsize=9)
    ax.tick_params(labelsize=8)

    return skew


def render_skewt(snd: dict, an: dict, title: str, subtitle: str, parcel: str = "ML") -> bytes:
    prof = snd["profile"]
    p, t, td = prof["p"], prof["t"], prof["td"]
    u, v = prof["u"], prof["v"]

    fig = Figure(figsize=(14.5, 10.2), dpi=110, facecolor="white")
    FigureCanvasAgg(fig)
    skew = skewt_background(fig, (0.055, 0.07, 0.54, 0.84))
    ax = skew.ax

    # ---- parcel and CAPE/CIN shading
    traces = an.get("_parcel_traces", {})
    sel = parcel if parcel in traces else next(iter(traces), None)
    pcol = {"SB": "#111111", "ML": "#111111", "MU": "#111111"}
    if sel:
        off = an["_mu_idx"] if sel == "MU" else 0
        ps, ts, tds = p[off:] * units.hPa, t[off:] * units.degC, td[off:] * units.degC
        tr = traces[sel] * units.degC
        skew.plot(ps, tr, color=pcol[sel], lw=1.6, ls=(0, (5, 3)), zorder=6)
        try:
            skew.shade_cin(ps, ts, tr, tds, alpha=0.25, color="#2563eb")
            skew.shade_cape(ps, ts, tr, alpha=0.28, color="#dc2626")
        except Exception:  # noqa: BLE001
            pass
        for other in traces:
            if other != sel:
                o = an["_mu_idx"] if other == "MU" else 0
                skew.plot(p[o:] * units.hPa, traces[other] * units.degC, color="#6b7280",
                          lw=0.8, ls=":", zorder=5)

    # ---- environment
    wb = an.get("wetbulb")
    if wb is not None:
        skew.plot(p * units.hPa, np.array(wb) * units.degC, color="#06b6d4", lw=1.0, zorder=7)
    tv = None
    try:
        import metpy.calc as mpcalc
        tv = mpcalc.virtual_temperature_from_dewpoint(p * units.hPa, t * units.degC,
                                                      td * units.degC).to("degC")
        skew.plot(p * units.hPa, tv, color="#dc2626", lw=0.8, ls="--", zorder=7, alpha=0.8)
    except Exception:  # noqa: BLE001
        pass
    stroke = [pe.withStroke(linewidth=3.6, foreground="white")]
    skew.plot(p * units.hPa, t * units.degC, color="#d00000", lw=2.6, zorder=8, path_effects=stroke)
    skew.plot(p * units.hPa, td * units.degC, color="#008a3e", lw=2.6, zorder=8, path_effects=stroke)
    ax.plot(t[0], p[0], "o", color="#d00000", ms=4, zorder=9)
    ax.plot(td[0], p[0], "o", color="#008a3e", ms=4, zorder=9)
    ax.text(t[0] + 0.8, p[0], f"{t[0]:.1f}°", color="#d00000", fontsize=8, va="top",
            fontweight="bold", zorder=10, path_effects=stroke)
    ax.text(td[0] - 0.8, p[0], f"{td[0]:.1f}°", color="#008a3e", fontsize=8, va="top",
            ha="right", fontweight="bold", zorder=10, path_effects=stroke)

    # ---- wind barbs on the right edge
    mask = p >= 100
    skew.plot_barbs(p[mask] * units.hPa, u[mask] * units.knot, v[mask] * units.knot, xloc=1.045,
                    length=6.2, linewidth=0.7, sizes={"emptybarb": 0.08}, color="#111")

    # ---- height (km AGL) labels on the left
    tr_axes = ax.get_yaxis_transform()
    for km in range(1, 16):
        pk = _p_at_height(prof, km * 1000)
        if pk is None or pk < 100:
            continue
        ax.plot([0, 0.018], [pk, pk], color="#444", lw=0.9, transform=tr_axes, zorder=9)
        ax.text(0.022, pk, f"{km} km", transform=tr_axes, fontsize=6.8, va="center",
                color="#333", zorder=9)
    ax.text(0.022, p[0], "SFC" if prof["z"][0] < 50 else f"{prof['z'][0]:.0f} m",
            transform=tr_axes, fontsize=6.8, va="bottom", color="#333")

    # ---- LCL / LFC / EL markers
    par = an["parcels"].get(sel or "", {})
    for key, label, col in (("lcl_p", "LCL", "#0b7a2b"), ("lfc_p", "LFC", "#b45309"),
                            ("el_p", "EL", "#7c3aed")):
        pv = par.get(key)
        if pv and 100 < pv <= p[0]:
            ax.plot([0.86, 0.9], [pv, pv], color=col, lw=2, transform=tr_axes, zorder=9)
            ax.text(0.905, pv, label, transform=tr_axes, color=col, fontsize=8,
                    fontweight="bold", va="center", zorder=9, path_effects=stroke)
    idx = an.get("indices", {})
    for key, label, col in (("frz", "0 °C", "#1d4ed8"), ("wbz", "WBZ", "#0891b2")):
        zz = idx.get(key)
        if zz is not None:
            pk = _p_at_height(prof, zz)
            if pk:
                ax.plot([0.0, 0.05], [pk, pk], color=col, lw=1.6, transform=tr_axes, zorder=9)
                ax.text(0.052, pk, label, transform=tr_axes, color=col, fontsize=7,
                        va="bottom", fontweight="bold", zorder=9, path_effects=stroke)

    # ---- titles
    fig.text(0.055, 0.965, title, fontsize=13.5, fontweight="bold", color="#0d1b2a", va="center")
    fig.text(0.055, 0.935, subtitle, fontsize=10, color="#1b263b", va="center")

    # ---- hodograph
    hax = fig.add_axes((0.665, 0.52, 0.31, 0.40))
    zagl = prof["z"] - prof["z"][0]
    spd = np.hypot(u, v)
    sel12 = zagl <= 12000
    rmax = max(20, int(np.ceil(np.nanmax(spd[sel12]) / 10.0) * 10) + 10) if sel12.any() else 60
    h = Hodograph(hax, component_range=rmax)
    h.add_grid(increment=10, color="#bbb", lw=0.5)
    hax.set_facecolor("#fbfbf8")
    zi = np.arange(0, min(12000, zagl[-1]) + 1, 100)
    ui, vi = np.interp(zi, zagl, u), np.interp(zi, zagl, v)
    for z0, z1, col, lab in HODO_COLORS:
        m = (zi >= z0) & (zi <= z1)
        if m.sum() > 1:
            seg = np.stack([ui[m], vi[m]], axis=1)
            hax.add_collection(LineCollection([seg], colors=col, linewidths=2.4, label=lab))
    for km in (1, 3, 6, 9):
        if km * 1000 <= zi[-1]:
            uu, vv = np.interp(km * 1000, zi, ui), np.interp(km * 1000, zi, vi)
            hax.plot(uu, vv, "o", ms=3.5, color="#111")
            hax.text(uu + rmax * 0.015, vv + rmax * 0.015, str(km), fontsize=7)
    kin = an.get("kinematics", {})
    for key, lab, col in (("rm", "RM", "#b91c1c"), ("lm", "LM", "#1d4ed8"), ("mean", "MW", "#555")):
        val = kin.get(key)
        if val and None not in val:
            hax.plot(*val, marker="s" if key != "mean" else "^", ms=6, color=col, mfc="white",
                     mew=1.6)
            hax.text(val[0] + rmax * 0.03, val[1], lab, color=col, fontsize=7.5, fontweight="bold")
    hax.legend(loc="upper left", fontsize=6.8, frameon=True, framealpha=0.85)
    hax.set_title("Hodógrafa (kt)", fontsize=9.5, fontweight="bold", loc="left")
    hax.tick_params(labelsize=7)
    hax.set_xlabel("")
    hax.set_ylabel("")

    # ---- indices table
    tax = fig.add_axes((0.665, 0.05, 0.31, 0.43))
    tax.axis("off")
    tax.set_xlim(0, 1)
    tax.set_ylim(0, 1)
    tax.set_autoscale_on(False)
    pr = an["parcels"]

    def col(key, fmt="{:.0f}"):
        return [_f(pr.get(k, {}).get(key), fmt) for k in ("SB", "ML", "MU")]

    rows = [("Parcela", ["SB", "ML", "MU"]),
            ("CAPE (J/kg)", col("cape")),
            ("CIN (J/kg)", col("cin")),
            ("LI (°C)", col("li", "{:.1f}")),
            ("LCL (m AGL)", col("lcl_z")),
            ("LFC (m AGL)", col("lfc_z")),
            ("EL (hPa)", col("el_p"))]
    y = 0.98
    for i, (name, vals) in enumerate(rows):
        bold = "bold" if i == 0 else "normal"
        tax.text(0.0, y, name, fontsize=8.6, fontweight="bold", va="top", family="DejaVu Sans")
        for j, vv in enumerate(vals):
            color = "#9d0208" if (i == 0 and vals[j] == (sel or "")) else "#111"
            tax.text(0.42 + j * 0.2, y, vv, fontsize=8.6, va="top", ha="right" if i else "right",
                     fontweight=bold, family="DejaVu Sans Mono", color=color)
        y -= 0.058
    tax.plot([0, 1], [y + 0.02, y + 0.02], color="#999", lw=0.6)
    y -= 0.01

    left = [("K", _f(idx.get("k"), "{:.0f}")), ("Total Totals", _f(idx.get("tt"), "{:.0f}")),
            ("Showalter", _f(idx.get("showalter"), "{:.1f}")),
            ("SWEAT", _f(idx.get("sweat"), "{:.0f}")),
            ("Água precip.", _f(idx.get("pw"), "{:.1f}", " mm")),
            ("DCAPE", _f(idx.get("dcape"), "{:.0f}", " J/kg")),
            ("Iso 0 °C", _f(idx.get("frz_msl"), "{:.0f}", " m")),
            ("WBZ", _f(idx.get("wbz_msl"), "{:.0f}", " m")),
            ("Γ 850–500", _f(idx.get("lr_850_500"), "{:.1f}", " °C/km")),
            ("Γ 700–500", _f(idx.get("lr_700_500"), "{:.1f}", " °C/km"))]
    right = [("Shear 0–1", _f(kin.get("shear_0_1"), "{:.0f}", " kt")),
             ("Shear 0–3", _f(kin.get("shear_0_3"), "{:.0f}", " kt")),
             ("Shear 0–6", _f(kin.get("shear_0_6"), "{:.0f}", " kt")),
             ("SRH 0–1", _f(kin.get("srh_0_1"), "{:.0f}", " m²/s²")),
             ("SRH 0–3", _f(kin.get("srh_0_3"), "{:.0f}", " m²/s²")),
             ("Bunkers RM", _wind_txt(kin.get("rm"))),
             ("Bunkers LM", _wind_txt(kin.get("lm"))),
             ("SCP", _f(idx.get("scp"), "{:.1f}")),
             ("STP", _f(idx.get("stp"), "{:.1f}")),
             ("Γ 0–3 km", _f(idx.get("lr_0_3"), "{:.1f}", " °C/km"))]
    y0 = y
    for (a, av), (b, bv) in zip(left, right):
        tax.text(0.0, y0, a, fontsize=8.2, va="top", fontweight="bold")
        tax.text(0.49, y0, av, fontsize=8.2, va="top", ha="right", family="DejaVu Sans Mono")
        tax.text(0.53, y0, b, fontsize=8.2, va="top", fontweight="bold")
        tax.text(1.0, y0, bv, fontsize=8.2, va="top", ha="right", family="DejaVu Sans Mono")
        y0 -= 0.052

    leg = ("— T   — Td   — Tw (bolbo húmido)   -- Tv   -- parcela " + (sel or "") +
           "   ░ CAPE/CIN")
    fig.text(0.055, 0.025, leg, fontsize=7.5, color="#333")
    fig.text(0.975, 0.012, "Forecast Workstation", fontsize=7, color="#666", ha="right",
             style="italic")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor="white")
    return buf.getvalue()


def _wind_txt(uv):
    if not uv or None in uv:
        return "—"
    u, v = uv
    spd = float(np.hypot(u, v))
    d = (np.degrees(np.arctan2(-u, -v)) + 360) % 360
    return f"{d:03.0f}°/{spd:.0f} kt"
