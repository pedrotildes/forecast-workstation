"""Kristin: rajada máxima prevista para a madrugada de 28/01/2026 (00–09 UTC) pelo ECMWF IFS e pelo
GFS, nas corridas de 26 e 27/01, com a trajetória prevista do centro, a da ERA5 e as rajadas
observadas (IPMA, Monte Real). Imprime também as métricas usadas no texto.

Corre a partir de backend/:  ../.venv/bin/python ../docs/casos/kristin/make_prev_kristin.py
"""
import asyncio
import datetime as dt
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "backend"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from app import archive  # noqa: E402
from app.cases import _use, all_tracks, track_forecast  # noqa: E402
from app.maps.styles import GUST_COLORS, GUST_LEVELS  # noqa: E402
from app.models import MODELS, FieldUnavailable  # noqa: E402
from app.models.base import Domain  # noqa: E402

OUT = Path(__file__).resolve().parent / "figuras"
H = dt.timedelta(hours=1)
T = dt.datetime
RUNS = [T(2026, 1, 26, 0), T(2026, 1, 26, 12), T(2026, 1, 27, 0), T(2026, 1, 27, 12)]
LANDFALL = T(2026, 1, 28, 5)
W0, W1 = T(2026, 1, 28, 0), T(2026, 1, 28, 9)          # janela da rajada prevista
DOM = Domain(-13, -3, 35.5, 43.5)
EXT = (-11.2, -6.2, 37.4, 42.2)
BOX = (39.3, 40.4, -9.5, -7.3)                          # Leiria–Ansião–Tomar–Castelo Branco, só em terra (lat0, lat1, lon0, lon1)
# (nome, lat, lon, rajada km/h, deslocamento do rótulo dx, dy, alinhamento)
OBS = [("Monte Real*", 39.83, -8.89, 176, -0.09, 0.10, "right"), ("Ansião", 39.92, -8.44, 172, 0.07, 0.06, "left"),
       ("Leiria", 39.78, -8.82, 156, 0.07, -0.20, "left"), ("C. Carvoeiro", 39.36, -9.41, 149, -0.08, 0.0, "right"),
       ("C. Branco", 39.84, -7.48, 137, 0.07, 0.05, "left"), ("Tomar", 39.60, -8.41, 133, 0.07, -0.12, "left"),
       ("Lisboa", 38.77, -9.15, 110, 0.07, 0.03, "left")]
NAME = {"ecmwf": "ECMWF IFS", "gfs": "GFS"}
COL = {"ecmwf": "#7c3aed", "gfs": "#2563eb", "aifs": "#db2777"}


_LAND = None


def _land_mask(LA, LO):
    """True on land (Natural Earth 10 m) — the gust maxima quoted are over land only."""
    global _LAND
    if _LAND is None:
        import shapely.ops
        from cartopy.io import shapereader
        from shapely.prepared import prep
        _LAND = prep(shapely.ops.unary_union(list(
            shapereader.Reader(shapereader.natural_earth("10m", "physical", "land")).geometries())))
    from shapely.geometry import Point
    out = np.zeros(LA.shape, bool)
    sel = (LA >= 36) & (LA <= 43) & (LO >= -10.5) & (LO <= -6)
    for i in zip(*np.where(sel)):
        out[i] = _LAND.contains(Point(float(LO[i]), float(LA[i])))
    return out


def km(a, b):
    return 111.2 * math.hypot(a[0] - b[0], (a[1] - b[1]) * math.cos(math.radians((a[0] + b[0]) / 2)))


def interp(track, t):
    pts = [(archive.parse_t(p["time"]), p) for p in track]
    for (ta, a), (tb, b) in zip(pts, pts[1:]):
        if ta <= t <= tb:
            f = (t - ta) / (tb - ta) if tb > ta else 0
            return tuple(a[k] + f * (b[k] - a[k]) for k in ("lat", "lon", "p"))
    return None


async def gust_max(m, run):
    """Maximum gust (km/h) over W0..W1 and the steps used."""
    every = 1 if m.id == "gfs" else 3
    steps = [int((t - run) / H) for t in (W0 + i * every * H for i in range(int((W1 - W0) / H) // every + 1))]
    if m.id == "ecmwf":
        steps = [s for s in steps if s > int((W0 - run) / H)]     # 3-h maxima ending at the step
    acc, used = None, []
    for s in steps:
        try:
            f = (await m.fetch(run, s, [("gust", 0)], DOM))[("gust", 0)]
        except (FieldUnavailable, KeyError):
            continue
        v = f.values * 3.6
        acc = v if acc is None else np.fmax(acc, v)
        used.append(s)
        lats, lons = f.lats, f.lons
    return (lats, lons, acc, used) if acc is not None else None


async def main():
    ev = archive.get_event("depresso-kristin")
    _use(ev)
    era = (await all_tracks(ev, ["era5"]))["Kristin"]["era5"]
    start = next(p for p in era if p["time"].startswith("2026-01-28T00"))
    ref_lf = interp(era, LANDFALL)
    cmap = ListedColormap(GUST_COLORS[:len(GUST_LEVELS) - 1])
    cmap.set_over("#ffe0ff")
    norm = BoundaryNorm(GUST_LEVELS, cmap.N)
    fig = Figure(figsize=(16, 9.6), dpi=110, facecolor="white")
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(2, 4, left=0.03, right=0.99, top=0.86, bottom=0.11, wspace=0.04, hspace=0.16)
    proj = ccrs.Mercator()
    rows = []
    for i, mid in enumerate(("ecmwf", "gfs")):
        m = MODELS[mid]
        for j, run in enumerate(RUNS):
            ax = fig.add_subplot(gs[i, j], projection=proj)
            ax.set_extent(EXT, ccrs.PlateCarree())
            g = await gust_max(m, run)
            tr = await track_forecast(m, run, ev, start)
            lead = int((LANDFALL - run) / H)
            if g:
                lats, lons, v, used = g
                ax.contourf(lons, lats, v, levels=GUST_LEVELS, cmap=cmap, norm=norm, extend="max",
                            transform=ccrs.PlateCarree(), zorder=1)
                LA, LO = np.meshgrid(lats, lons, indexing="ij")
                onland = _land_mask(LA, LO) & np.isfinite(v)
                inb = (LA >= BOX[0]) & (LA <= BOX[1]) & (LO >= BOX[2]) & (LO <= BOX[3]) & onland
                bmax = float(np.nanmax(np.where(inb, v, np.nan)))
                pt = (LA >= 37) & (LA <= 42.2) & onland
                k = np.nanargmax(np.where(pt, v, np.nan))
                at = (float(LA.flat[k]), float(LO.flat[k]), float(v.flat[k]))
            else:
                bmax, at, used = float("nan"), None, []
            ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="none", edgecolor="#333", lw=0.6, zorder=3)
            ax.add_feature(cfeature.BORDERS.with_scale("10m"), edgecolor="#555", lw=0.5, zorder=3)
            ax.plot([p["lon"] for p in era], [p["lat"] for p in era], "--", color="black", lw=1.4,
                    transform=ccrs.PlateCarree(), zorder=5)
            fc_lf = interp(tr, LANDFALL) if tr else None
            if tr:
                ax.plot([p["lon"] for p in tr], [p["lat"] for p in tr], "-o", color="white", lw=3.2, ms=5,
                        transform=ccrs.PlateCarree(), zorder=5)
                ax.plot([p["lon"] for p in tr], [p["lat"] for p in tr], "-o", color=COL[mid], lw=1.8, ms=3.5,
                        transform=ccrs.PlateCarree(), zorder=6)
            if fc_lf:
                ax.plot(fc_lf[1], fc_lf[0], marker="x", ms=10, mew=2.6, color=COL[mid], transform=ccrs.PlateCarree(), zorder=7)
            if ref_lf:
                ax.plot(ref_lf[1], ref_lf[0], marker="x", ms=10, mew=2.6, color="black", transform=ccrs.PlateCarree(), zorder=7)
            for name, la, lo, val, dx, dy, ha in OBS:
                ax.plot(lo, la, "o", ms=4, color="black", transform=ccrs.PlateCarree(), zorder=8)
                ax.text(lo + dx, la + dy, f"{val}", fontsize=7.2, fontweight="bold", color="black", ha=ha,
                        transform=ccrs.PlateCarree(), zorder=9,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
            ax.set_title(f"{NAME[mid]} · run {run:%d/%m %H} UTC (+{lead} h)", fontsize=9.5, fontweight="bold", loc="left")
            txt = f"em terra: máx. {at[2]:.0f} km/h a {at[0]:.2f}°N".replace(".", ",") if at else ""
            txt += f"\nfaixa Leiria–C. Branco: {bmax:.0f} km/h"
            if fc_lf:
                d = km(fc_lf[:2], ref_lf[:2]) if ref_lf else float("nan")
                txt += f"\ncentro às 05 UTC: {fc_lf[2]:.0f} hPa, {d:.0f} km da ERA5"
            ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=7.6, va="bottom", zorder=10,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999", alpha=0.9))
            pmin = min((p["p"] for p in tr), default=float("nan"))
            rows.append((mid, run, lead, bmax, at, fc_lf, pmin, used))
    cax = fig.add_axes([0.25, 0.055, 0.5, 0.02])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                      extend="max", ticks=GUST_LEVELS)
    cb.set_label("Rajada máxima prevista entre as 00 e as 09 UTC de 28/01 (km/h)", fontsize=9)
    fig.text(0.03, 0.965, "Kristin: rajada prevista para a madrugada de 28/01/2026, 1 a 2 dias antes", fontsize=15,
             fontweight="bold", color="#0d1b2a", va="top")
    fig.text(0.03, 0.93, "Sombreado: máximo da rajada a 10 m entre 00 e 09 UTC (ECMWF: máximos de 3 h; GFS: valores horários). "
             "Linha colorida: trajetória prevista do centro (de 6 em 6 h a partir de 28/00); tracejado preto: ERA5; "
             "✕: posição às 05 UTC. Números: rajadas observadas (km/h, IPMA; *Monte Real fora da rede IPMA).",
             fontsize=8.6, color="#1b263b", va="top", wrap=True)
    fig.text(0.99, 0.012, "Dados: ECMWF Open Data e NOAA GFS (arquivos na AWS) · ERA5 · IPMA · Forecast Workstation",
             fontsize=7.5, color="#666", ha="right")
    fig.savefig(OUT / "k13_rajada_prevista_ecmwf_gfs.png", dpi=110, facecolor="white")
    print("ok k13_rajada_prevista_ecmwf_gfs.png")
    print("ERA5 às 05 UTC:", ref_lf)
    for mid, run, lead, bmax, at, fc_lf, pmin, used in rows:
        print(f"{mid:6s} {run:%d/%H} +{lead:3d}h  rajada máx faixa {bmax:5.0f} em {at}  centro 05Z {fc_lf and tuple(round(x, 1) for x in fc_lf)}"
              f"  pmin {pmin:.1f}  passos {used}")
    # AIFS and all models: forecast centre at 05 UTC and minimum pressure, every 00/12Z run
    for mid in ("ecmwf", "aifs", "gfs"):
        m = MODELS[mid]
        for run in [T(2026, 1, 24, 12) + 12 * H * k for k in range(7)]:
            tr = await track_forecast(m, run, ev, start)
            lf = interp(tr, LANDFALL) if tr else None
            d = km(lf[:2], ref_lf[:2]) if lf and ref_lf else float("nan")
            print(f"  {mid:6s} {run:%d/%H} +{int((LANDFALL - run) / H):3d}h  pontos {len(tr):2d}  05Z {lf and tuple(round(x, 1) for x in lf)}"
                  f"  erro {d:5.0f} km  pmin {min((p['p'] for p in tr), default=float('nan')):.1f}")


asyncio.run(main())
