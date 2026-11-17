"""Ophelia (15–16/10/2017): observações horárias das estações (NOAA ISD — METAR/SYNOP) contra a
reanálise ERA5 arquivada pela Forecast Workstation:
  • Portugal, 15/10: temperatura, humidade relativa, vento e rajada (o dia dos incêndios);
  • Irlanda, 16/10: pressão, vento e rajada na chegada do Ophelia.

Corre a partir de backend/:  ../.venv/bin/python ../docs/casos/ophelia/make_obs_ophelia.py
"""
import asyncio
import csv
import datetime as dt
import io
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "backend"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from app import atcf  # noqa: E402
from app.models import MODELS  # noqa: E402
from app.models.base import Domain  # noqa: E402

OUT = Path(__file__).resolve().parent / "figuras"
DATA = Path(__file__).resolve().parent / "dados"
OUT.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
ISD = "https://noaa-global-hourly-pds.s3.amazonaws.com/2017/{}.csv"
H = dt.timedelta(hours=1)
PT = [("08545099999", "Porto", 41.248, -8.681), ("08548099999", "Coimbra/Cernache", 40.15, -8.467),
      ("08540099999", "Monte Real (Leiria)", 39.831, -8.887), ("08560099999", "Viseu", 40.717, -7.883),
      ("08570099999", "Castelo Branco", 39.833, -7.483), ("08536099999", "Lisboa", 38.781, -9.136)]
IE = [("03953099999", "Valentia", 51.933, -10.25), ("03951099999", "Sherkin Island", 51.467, -9.417),
      ("03955099999", "Cork (aeroporto)", 51.841, -8.491), ("03962099999", "Shannon", 52.702, -8.925),
      ("03963099999", "Mace Head", 53.317, -9.9), ("03969099999", "Dublin (aeroporto)", 53.421, -6.27)]


def rh(t, td):
    es = lambda x: 6.112 * math.exp(17.67 * x / (x + 243.5))  # noqa: E731
    return 100 * es(td) / es(t)


def isd(sid, t0, t1):
    p = DATA / f"{sid}.csv"
    if not p.exists():
        p.write_bytes(urllib.request.urlopen(ISD.format(sid), timeout=120).read())
    out = []
    for r in csv.DictReader(io.StringIO(p.read_text())):
        t = dt.datetime.fromisoformat(r["DATE"])
        if not (t0 <= t <= t1):
            continue
        rec = {"time": t}
        v = (r.get("SLP") or "").split(",")
        if v[0] not in ("", "99999"):
            rec["slp"] = int(v[0]) / 10
        w = (r.get("WND") or "").split(",")
        if len(w) > 3 and w[3] != "9999":
            rec["wind"] = int(w[3]) / 10 * 3.6
        g = (r.get("OC1") or "").split(",")
        if g[0] not in ("", "9999"):
            rec["gust"] = int(g[0]) / 10 * 3.6
        tt = (r.get("TMP") or "").split(",")
        dd = (r.get("DEW") or "").split(",")
        if tt[0] not in ("", "+9999"):
            rec["t"] = int(tt[0]) / 10
            if dd[0] not in ("", "+9999"):
                rec["rh"] = rh(rec["t"], int(dd[0]) / 10)
        out.append(rec)
    return out


async def era5_point(lat, lon, t0, t1):
    """Hourly ERA5 at a point from the event archive (analyses: steps 0–5 of each 6-h run;
    gusts: the maximum in the hour ending at t, steps 1–6)."""
    m = MODELS["era5"]
    dom = Domain.tile(lat, lon)
    out = []
    t = t0
    while t <= t1:
        r = t.replace(hour=(t.hour // 6) * 6)
        s = int((t - r) / H)
        rec = {"time": t}
        try:
            f = await m.fetch(r, s, [("msl", 0), ("t2m", 0), ("d2m", 0), ("u10", 0), ("v10", 0)], dom)
            rec["slp"] = f[("msl", 0)].at(lat, lon) / 100
            tc, dc = f[("t2m", 0)].at(lat, lon) - 273.15, f[("d2m", 0)].at(lat, lon) - 273.15
            rec["t"], rec["rh"] = tc, rh(tc, dc)
            rec["wind"] = math.hypot(f[("u10", 0)].at(lat, lon), f[("v10", 0)].at(lat, lon)) * 3.6
        except Exception:  # noqa: BLE001
            pass
        rg, sg = (r, s) if s else (r - 6 * H, 6)
        try:
            g = await m.fetch(rg, sg, [("gust", 0)], dom)
            rec["gust"] = g[("gust", 0)].at(lat, lon) * 3.6
        except Exception:  # noqa: BLE001
            pass
        out.append(rec)
        t += H
    return out


def _axes(a):
    a.grid(alpha=0.3)
    a.xaxis.set_major_formatter(mdates.DateFormatter("%d %Hh"))
    a.tick_params(labelsize=7.5)


def _series(a, recs, key, **kw):
    rs = [r for r in recs if key in r and np.isfinite(r[key])]
    return a.plot([r["time"] for r in rs], [r[key] for r in rs], **kw)


async def portugal():
    t0, t1 = dt.datetime(2017, 10, 14, 12), dt.datetime(2017, 10, 16, 12)
    fig = Figure(figsize=(14.5, 13.2), dpi=110, facecolor="white")
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(len(PT), 2, hspace=0.38, wspace=0.16, left=0.06, right=0.94, top=0.91, bottom=0.05)
    rows = []
    for i, (sid, name, la, lo) in enumerate(PT):
        o, e = isd(sid, t0, t1), await era5_point(la, lo, t0, t1)
        a1, a2 = fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1])
        _series(a1, e, "t", color="#dc2626", lw=1.8, label="ERA5 T")
        _series(a1, o, "t", color="#7f1d1d", marker="o", ms=3, lw=1, label="obs. T")
        b1 = a1.twinx()
        _series(b1, e, "rh", color="#2563eb", lw=1.8, label="ERA5 HR")
        _series(b1, o, "rh", color="#1e3a8a", marker="o", ms=3, lw=1, label="obs. HR")
        b1.set_ylim(0, 100)
        b1.set_ylabel("HR (%)", fontsize=8, color="#1e3a8a")
        b1.tick_params(labelsize=7.5)
        _series(a2, e, "wind", color="#0f766e", lw=1.8, label="ERA5 vento médio")
        _series(a2, e, "gust", color="#0f766e", lw=1.1, ls="--", label="ERA5 rajada")
        _series(a2, o, "wind", color="#111", marker="o", ms=3, lw=1, label="obs. vento médio")
        og = [r for r in o if "gust" in r]
        a2.plot([r["time"] for r in og], [r["gust"] for r in og], "^", color="#dc2626", ms=6, label="obs. rajada")
        for a in (a1, a2):
            _axes(a)
            a.axvspan(dt.datetime(2017, 10, 15, 12), dt.datetime(2017, 10, 16, 0), color="#f59e0b", alpha=0.08)
        a1.set_ylabel("°C", fontsize=8, color="#7f1d1d")
        a2.set_ylabel("km/h", fontsize=8)
        a1.set_title(f"{name}: temperatura e humidade relativa", fontsize=9, loc="left", fontweight="bold")
        a2.set_title(f"{name}: vento e rajada", fontsize=9, loc="left", fontweight="bold")
        if i == 0:
            h1, l1 = a1.get_legend_handles_labels()
            h2, l2 = b1.get_legend_handles_labels()
            a1.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left", ncol=2)
            a2.legend(fontsize=7, loc="upper left", ncol=2)
        d15 = [r for r in o if dt.datetime(2017, 10, 15) <= r["time"] < dt.datetime(2017, 10, 16)]
        tx = max((r["t"] for r in d15 if "t" in r), default=np.nan)
        hn = min((r["rh"] for r in d15 if "rh" in r), default=np.nan)
        gx = max((r["gust"] for r in d15 if "gust" in r), default=np.nan)
        wx = max((r["wind"] for r in d15 if "wind" in r), default=np.nan)
        e15 = [r for r in e if dt.datetime(2017, 10, 15) <= r["time"] < dt.datetime(2017, 10, 16)]
        rows.append((name, tx, hn, wx, gx, max((r.get("t", np.nan) for r in e15)), min((r.get("rh", np.nan) for r in e15)),
                     max((r.get("gust", np.nan) for r in e15))))
    fig.text(0.06, 0.975, "Ophelia: o dia 15 de outubro de 2017 nas estações de Portugal vs. ERA5", fontsize=14,
             fontweight="bold", color="#0d1b2a", va="top")
    fig.text(0.06, 0.95, "Observações (NOAA ISD: METAR e SYNOP) com pontos; ERA5 no ponto da estação com linha contínua; "
             "faixa laranja = tarde e noite de 15/10 (maior proximidade do Ophelia e propagação dos incêndios)",
             fontsize=9, color="#1b263b", va="top")
    fig.text(0.94, 0.012, "Dados: NOAA NCEI ISD · ERA5 (ECMWF/C3S) · Forecast Workstation", fontsize=7.5, color="#666",
             ha="right")
    fig.savefig(OUT / "s3_portugal_15out.png", dpi=110, facecolor="white")
    print("ok s3_portugal_15out.png")
    for r in rows:
        print("PT", r[0], "obs Tmax %.1f HRmin %.0f vento %.0f rajada %.0f | ERA5 Tmax %.1f HRmin %.0f rajada %.0f" % r[1:])


async def ireland():
    t0, t1 = dt.datetime(2017, 10, 16, 0), dt.datetime(2017, 10, 17, 0)
    lf = atcf.landfall(atcf.best_track("AL172017"), after=dt.datetime(2017, 10, 15))
    fig = Figure(figsize=(14.5, 13.2), dpi=110, facecolor="white")
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(len(IE), 2, hspace=0.38, wspace=0.14, left=0.06, right=0.98, top=0.91, bottom=0.05)
    rows = []
    for i, (sid, name, la, lo) in enumerate(IE):
        o, e = isd(sid, t0, t1), await era5_point(la, lo, t0, t1)
        a1, a2 = fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1])
        _series(a1, e, "slp", color="#0f766e", lw=2, label="ERA5")
        _series(a1, o, "slp", color="#111", marker="o", ms=3, lw=1, label="observado")
        if sid == "03953099999":     # the hourly record has a gap at the minimum: NHC report value
            a1.plot([dt.datetime(2017, 10, 16, 10, 59)], [959.3], "*", ms=13, color="#b45309",
                    label="959,3 hPa às 10:59 (NHC)")
            a1.legend(fontsize=7.5, loc="lower left")
        _series(a2, e, "wind", color="#0f766e", lw=1.8, label="ERA5 vento médio")
        _series(a2, e, "gust", color="#0f766e", lw=1.1, ls="--", label="ERA5 rajada")
        _series(a2, o, "wind", color="#111", marker="o", ms=3, lw=1, label="obs. vento médio")
        og = [r for r in o if "gust" in r]
        a2.plot([r["time"] for r in og], [r["gust"] for r in og], "^", color="#dc2626", ms=6, label="obs. rajada")
        for a in (a1, a2):
            _axes(a)
            if lf:
                a.axvline(lf["time"], color="#b45309", lw=1, ls=":")
        a1.set_ylabel("hPa", fontsize=8)
        a2.set_ylabel("km/h", fontsize=8)
        a1.set_title(f"{name}: pressão ao nível do mar", fontsize=9, loc="left", fontweight="bold")
        a2.set_title(f"{name}: vento e rajada", fontsize=9, loc="left", fontweight="bold")
        if i == 0:
            a1.legend(fontsize=7.5, loc="lower left")
            a2.legend(fontsize=7, loc="upper left", ncol=2)
        smin = min((r for r in o if "slp" in r), key=lambda r: r["slp"], default=None)
        gmax = max((r for r in o if "gust" in r), key=lambda r: r["gust"], default=None)
        wmax = max((r["wind"] for r in o if "wind" in r), default=np.nan)
        emin = min((r for r in e if "slp" in r), key=lambda r: r["slp"])
        egst = max((r.get("gust", np.nan) for r in e))
        ewnd = max((r.get("wind", np.nan) for r in e))
        rows.append((name, smin, gmax, wmax, emin, egst, ewnd))
    fig.text(0.06, 0.975, "Ophelia: a chegada à Irlanda a 16 de outubro de 2017 — estações vs. ERA5", fontsize=14,
             fontweight="bold", color="#0d1b2a", va="top")
    fig.text(0.06, 0.95, "Observações (NOAA ISD: METAR e SYNOP) a preto, rajadas a vermelho; ERA5 no ponto da estação "
             "a verde; tracejado laranja = chegada a terra observada (NHC)", fontsize=9, color="#1b263b", va="top")
    fig.text(0.98, 0.012, "Dados: NOAA NCEI ISD · ERA5 (ECMWF/C3S) · NHC · Forecast Workstation", fontsize=7.5,
             color="#666", ha="right")
    fig.savefig(OUT / "s4_irlanda_16out.png", dpi=110, facecolor="white")
    print("ok s4_irlanda_16out.png")
    for name, smin, gmax, wmax, emin, egst, ewnd in rows:
        print("IE", name, "obs pmin", smin and (smin["slp"], smin["time"].strftime("%H:%M")), "rajada",
              gmax and (round(gmax["gust"]), gmax["time"].strftime("%H:%M")), "vento máx %.0f" % wmax,
              "| ERA5 pmin %.1f %s rajada %.0f vento %.0f" % (emin["slp"], emin["time"].strftime("%H"), egst, ewnd))


async def main():
    await portugal()
    await ireland()


asyncio.run(main())
