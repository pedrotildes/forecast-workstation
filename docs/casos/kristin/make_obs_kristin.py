"""Kristin (28/01/2026): observações METAR dos aeródromos (Iowa Environmental Mesonet) contra a
reanálise ERA5 arquivada pela Forecast Workstation.

Corre a partir de backend/:  ../.venv/bin/python ../docs/casos/kristin/make_obs_kristin.py
"""
import asyncio
import csv
import datetime as dt
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "backend"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from app import archive  # noqa: E402
from app.cases import _use, point_chain  # noqa: E402
from app.models import MODELS  # noqa: E402

OUT = Path(__file__).resolve().parent / "figuras"
DATA = Path(__file__).resolve().parent / "dados"
OUT.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
IEM = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station={}&data=mslp&data=sknt&data=gust"
       "&data=metar&year1=2026&month1=1&day1=27&hour1=12&year2=2026&month2=1&day2=29&hour2=0&tz=Etc/UTC"
       "&format=onlycomma&latlon=yes&missing=M&trace=T&direct=no&report_type=3&report_type=4")
STATIONS = [("LPMR", "Monte Real (BA5)", 39.831, -8.887), ("LPPR", "Porto", 41.248, -8.681),
            ("LPPT", "Lisboa", 38.781, -9.136), ("LPMT", "Montijo (BA6)", 38.704, -9.036),
            ("LPBJ", "Beja (BA11)", 38.079, -7.932), ("LPFR", "Faro", 37.014, -7.966)]
T0, T1 = dt.datetime(2026, 1, 27, 18), dt.datetime(2026, 1, 28, 12)
LANDFALL = dt.datetime(2026, 1, 28, 4, 45)       # Wikipedia / IPMA: ~04h45 UTC, distrito de Leiria
KT = 1.852
RX_W = re.compile(r"\b(\d{3}|VRB)(\d{2,3})(?:G(\d{2,3}))?KT\b")
RX_Q = re.compile(r"Q(\d{4})\b")


def metars(icao):
    p = DATA / f"{icao}.csv"
    if not p.exists() or p.stat().st_size < 200:
        p.write_bytes(urllib.request.urlopen(IEM.format(icao), timeout=120).read())
        time.sleep(5)
    out = []
    for r in csv.DictReader(p.open()):
        t = dt.datetime.fromisoformat(r["valid"])
        if not (T0 <= t <= T1):
            continue
        rec = {"time": t}
        m = r.get("metar") or ""
        w = RX_W.search(m)
        if w:
            rec["wind"] = int(w.group(2)) * KT
            if w.group(3):
                rec["gust"] = int(w.group(3)) * KT
        q = RX_Q.search(m)
        if q:
            rec["slp"] = int(q.group(1))          # QNH ≈ pressão ao nível do mar (aeródromos baixos)
        out.append(rec)
    return out


async def main():
    ev = archive.get_event("depresso-kristin")
    _use(ev)
    m = MODELS["era5"]
    fig = Figure(figsize=(14.5, 12.8), dpi=110, facecolor="white")
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(len(STATIONS), 2, hspace=0.38, wspace=0.14, left=0.06, right=0.98, top=0.91, bottom=0.05)
    rows = []
    for i, (icao, name, la, lo) in enumerate(STATIONS):
        o = metars(icao)
        ch = await point_chain(m, ev, la, lo)
        ix = [k for k, t in enumerate(ch["time"]) if T0 <= t <= T1]
        tt = [ch["time"][k] for k in ix]
        a1, a2 = fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1])
        a1.plot(tt, [ch["msl"][k] for k in ix], color="#0f766e", lw=2, label="ERA5")
        os_ = [r for r in o if "slp" in r]
        a1.plot([r["time"] for r in os_], [r["slp"] for r in os_], "o-", color="#111", ms=3.2, lw=1, label="METAR (QNH)")
        a2.plot(tt, [ch["wind"][k] for k in ix], color="#0f766e", lw=2, label="ERA5 vento médio")
        a2.plot(tt, [ch["gust"][k] for k in ix], color="#0f766e", lw=1.2, ls="--", label="ERA5 rajada")
        ow = [r for r in o if "wind" in r]
        og = [r for r in o if "gust" in r]
        a2.plot([r["time"] for r in ow], [r["wind"] for r in ow], "o-", color="#111", ms=3, lw=1, label="METAR vento médio")
        a2.plot([r["time"] for r in og], [r["gust"] for r in og], "^", color="#dc2626", ms=6, label="METAR rajada")
        for a in (a1, a2):
            a.axvline(LANDFALL, color="#b45309", lw=1, ls=":")
            a.grid(alpha=0.3)
            a.set_xlim(T0, T1)
            a.xaxis.set_major_formatter(mdates.DateFormatter("%d %Hh"))
            a.tick_params(labelsize=7.5)
        a1.set_ylabel("hPa", fontsize=8)
        a2.set_ylabel("km/h", fontsize=8)
        a1.set_title(f"{name}: pressão", fontsize=9, loc="left", fontweight="bold")
        a2.set_title(f"{name}: vento e rajada", fontsize=9, loc="left", fontweight="bold")
        if i == 0:
            a1.legend(fontsize=7.5, loc="lower left")
            a2.legend(fontsize=7.5, loc="upper left", ncol=2)
        smin = min(os_, key=lambda r: r["slp"]) if os_ else None
        gmax = max(og, key=lambda r: r["gust"]) if og else None
        wmax = max(ow, key=lambda r: r["wind"]) if ow else None
        emin = min((ch["msl"][k], ch["time"][k]) for k in ix)
        egst = max(((ch["gust"][k], ch["time"][k]) for k in ix if np.isfinite(ch["gust"][k])), default=(np.nan, None))
        ewnd = max((ch["wind"][k] for k in ix if np.isfinite(ch["wind"][k])), default=np.nan)
        rows.append((name, smin, gmax, wmax, emin, egst, ewnd))
    fig.text(0.06, 0.975, "Kristin: observações dos aeródromos vs. reanálise ERA5 — 27 a 28/01/2026", fontsize=14,
             fontweight="bold", color="#0d1b2a", va="top")
    fig.text(0.06, 0.95, "METAR/SPECI (Iowa Environmental Mesonet) a preto, rajadas a vermelho; ERA5 no ponto do aeródromo "
             "(cadeia horária) a verde; pontilhado laranja = entrada em terra (~04h45 UTC, distrito de Leiria)", fontsize=9,
             color="#1b263b", va="top")
    fig.text(0.98, 0.012, "Dados: METAR via Iowa Environmental Mesonet · ERA5 (ECMWF/C3S) · Forecast Workstation", fontsize=7.5,
             color="#666", ha="right")
    fig.savefig(OUT / "k11_aerodromos_vs_era5.png", dpi=110, facecolor="white")
    print("ok k11_aerodromos_vs_era5.png")
    for name, smin, gmax, wmax, emin, egst, ewnd in rows:
        print(f"{name}: QNH mín {smin and (smin['slp'], smin['time'].strftime('%d %H:%M'))} · rajada máx "
              f"{gmax and (round(gmax['gust']), gmax['time'].strftime('%d %H:%M'))} · vento máx "
              f"{wmax and (round(wmax['wind']), wmax['time'].strftime('%d %H:%M'))} · ERA5 mín {emin[0]:.1f} "
              f"{emin[1]:%d %H}h · ERA5 rajada máx {egst[0]:.0f} {egst[1] and egst[1].strftime('%d %H')}h · ERA5 vento máx {ewnd:.0f}")


asyncio.run(main())
