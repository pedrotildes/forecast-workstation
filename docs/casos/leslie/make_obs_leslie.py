"""Leslie (13–14/10/2018): observações horárias das estações (NOAA ISD — METAR/SYNOP) contra a
reanálise ERA5 arquivada pela Forecast Workstation, e tabela das observações.

Corre a partir de backend/:  ../.venv/bin/python ../docs/casos/leslie/make_obs_leslie.py
"""
import asyncio
import csv
import datetime as dt
import io
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

from app import archive, atcf  # noqa: E402
from app.cases import point_chain  # noqa: E402
from app.models import MODELS  # noqa: E402

OUT = Path(__file__).resolve().parent / "figuras"
DATA = Path(__file__).resolve().parent / "dados"
OUT.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
ISD = "https://noaa-global-hourly-pds.s3.amazonaws.com/2018/{}.csv"
STATIONS = [("08540099999", "Monte Real", 39.831, -8.887), ("08548099999", "Coimbra/Cernache", 40.15, -8.467),
            ("08544099999", "Ovar/Maceda", 40.917, -8.633), ("08545099999", "Porto", 41.248, -8.681),
            ("08567099999", "Vila Real", 41.274, -7.72), ("08560099999", "Viseu", 40.717, -7.883),
            ("08536099999", "Lisboa", 38.781, -9.136), ("08541099999", "Sines", 37.95, -8.867)]
T0, T1 = dt.datetime(2018, 10, 13, 6), dt.datetime(2018, 10, 14, 12)


def isd(sid):
    p = DATA / f"{sid}.csv"
    if not p.exists():
        p.write_bytes(urllib.request.urlopen(ISD.format(sid), timeout=120).read())
    out = []
    for r in csv.DictReader(io.StringIO(p.read_text())):
        t = dt.datetime.fromisoformat(r["DATE"])
        if not (T0 <= t <= T1):
            continue
        rec = {"time": t}
        if r.get("SLP") and r["SLP"].split(",")[0] != "99999":
            rec["slp"] = int(r["SLP"].split(",")[0]) / 10
        w = (r.get("WND") or "").split(",")
        if len(w) > 3 and w[3] != "9999":
            rec["wind"] = int(w[3]) / 10 * 3.6
        g = (r.get("OC1") or "").split(",")
        if g and g[0] not in ("", "9999"):
            rec["gust"] = int(g[0]) / 10 * 3.6
        out.append(rec)
    return out


async def main():
    ev = archive.get_event("furaco-leslie")
    m = MODELS["era5"]
    obs = {sid: isd(sid) for sid, *_ in STATIONS}
    lf = atcf.landfall(atcf.best_track("AL132018"), after=dt.datetime(2018, 10, 13))
    sel = STATIONS[:6]
    fig = Figure(figsize=(14.5, 12.8), dpi=110, facecolor="white")
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(len(sel), 2, hspace=0.35, wspace=0.14, left=0.06, right=0.98, top=0.91, bottom=0.05)
    rows = []
    for i, (sid, name, la, lo) in enumerate(sel):
        ch = await point_chain(m, ev, la, lo)
        tt = [t for t in ch["time"] if T0 <= t <= T1]
        ix = [k for k, t in enumerate(ch["time"]) if T0 <= t <= T1]
        o = obs[sid]
        a1, a2 = fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1])
        a1.plot(tt, [ch["msl"][k] for k in ix], color="#0f766e", lw=2, label="ERA5")
        os_ = [r for r in o if "slp" in r]
        a1.plot([r["time"] for r in os_], [r["slp"] for r in os_], "o-", color="#111", ms=3.5, lw=1, label="observado")
        a2.plot(tt, [ch["wind"][k] for k in ix], color="#0f766e", lw=2, label="ERA5 vento médio")
        a2.plot(tt, [ch["gust"][k] for k in ix], color="#0f766e", lw=1.2, ls="--", label="ERA5 rajada")
        ow = [r for r in o if "wind" in r]
        og = [r for r in o if "gust" in r]
        a2.plot([r["time"] for r in ow], [r["wind"] for r in ow], "o-", color="#111", ms=3, lw=1, label="obs. vento médio")
        a2.plot([r["time"] for r in og], [r["gust"] for r in og], "^", color="#dc2626", ms=6, label="obs. rajada")
        for a in (a1, a2):
            if lf:
                a.axvline(lf["time"], color="#b45309", lw=1, ls=":")
            a.grid(alpha=0.3)
            a.xaxis.set_major_formatter(mdates.DateFormatter("%d %Hh"))
            a.tick_params(labelsize=7.5)
        a1.set_ylabel("hPa", fontsize=8)
        a2.set_ylabel("km/h", fontsize=8)
        a1.set_title(f"{name}: pressão ao nível do mar", fontsize=9, loc="left", fontweight="bold")
        a2.set_title(f"{name}: vento e rajada", fontsize=9, loc="left", fontweight="bold")
        if i == 0:
            a1.legend(fontsize=7.5, loc="lower left")
            a2.legend(fontsize=7.5, loc="upper left", ncol=2)
        smin = min(os_, key=lambda r: r["slp"]) if os_ else None
        gmax = max(og, key=lambda r: r["gust"]) if og else None
        emin = min((ch["msl"][k], ch["time"][k]) for k in ix)
        egst = max((ch["gust"][k] for k in ix if np.isfinite(ch["gust"][k])), default=np.nan)
        rows.append((name, smin, gmax, emin, egst))
    fig.text(0.06, 0.975, "Leslie: observações das estações vs. reanálise ERA5 — 13 a 14/10/2018", fontsize=14,
             fontweight="bold", color="#0d1b2a", va="top")
    fig.text(0.06, 0.95, "Observações horárias/sinóticas (NOAA ISD: METAR e SYNOP) a preto, rajadas a vermelho; ERA5 no ponto "
             "da estação (cadeia horária) a verde; tracejado laranja = chegada a terra observada (NHC)", fontsize=9,
             color="#1b263b", va="top")
    fig.text(0.98, 0.012, "Dados: NOAA NCEI ISD · ERA5 (ECMWF/C3S) · NHC · Forecast Workstation", fontsize=7.5,
             color="#666", ha="right")
    fig.savefig(OUT / "s3_estacoes_vs_era5.png", dpi=110, facecolor="white")
    print("ok s3_estacoes_vs_era5.png")
    for name, smin, gmax, emin, egst in rows:
        print(f"{name}: obs SLP mín {smin and (smin['slp'], smin['time'].strftime('%d %H:%M'))} · rajada máx "
              f"{gmax and (round(gmax['gust']), gmax['time'].strftime('%d %H:%M'))} · ERA5 mín {emin[0]:.1f} "
              f"{emin[1]:%d %H}h · ERA5 rajada máx {egst:.0f}")


asyncio.run(main())
