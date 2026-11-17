"""Figuras de satélite (EUMETSAT EUMETView, arquivo WMS) e de radar (mosaico RainViewer
arquivado pela Forecast Workstation) para o estudo das depressões Bruna e Charles.

Corre a partir de backend/:  ../.venv/bin/python ../docs/casos/bruna-charles/make_sat_radar.py
"""
import asyncio
import datetime as dt
import io
import math
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "backend"))
import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
from matplotlib import patheffects as pe  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

from app import archive  # noqa: E402
from app.cases import all_tracks, analysis_field  # noqa: E402
from app.models import MODELS  # noqa: E402

OUT = Path(__file__).resolve().parent / "figuras"
WMS = "https://view.eumetsat.int/geoserver/ows"
MERC = ccrs.Mercator.GOOGLE
PC = ccrs.PlateCarree()
R = 6378137.0
RADAR = archive.RADAR_DIR


def merc(lon, lat):
    return lon * math.pi / 180 * R, math.log(math.tan((90 + lat) * math.pi / 360)) * R


def wms(layer, style, t, lon0, lon1, lat0, lat1, width=1600):
    x0, y0 = merc(lon0, lat0)
    x1, y1 = merc(lon1, lat1)
    h = int(width * (y1 - y0) / (x1 - x0))
    q = urllib.parse.urlencode({"service": "WMS", "version": "1.3.0", "request": "GetMap", "layers": layer,
                                "styles": style, "crs": "EPSG:3857", "bbox": f"{x0},{y0},{x1},{y1}",
                                "width": width, "height": h, "format": "image/png",
                                "time": t.strftime("%Y-%m-%dT%H:%M:%SZ")})
    with urllib.request.urlopen(f"{WMS}?{q}", timeout=120) as r:
        img = Image.open(io.BytesIO(r.read())).convert("RGB")
    return np.asarray(img), (x0, x1, y0, y1)


def _fmt(t):
    week = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"][t.weekday()]
    return f"{week} {t.day:02d} Nov {t:%H:%M} UTC"


def _centre(track, t):
    """Linear interpolation of an analysed track at time t (None outside it)."""
    ts = [archive.parse_t(p["time"]) for p in track]
    if not ts or t < ts[0] or t > ts[-1]:
        return None
    for (a, pa), (b, pb) in zip(zip(ts, track), zip(ts[1:], track[1:])):
        if a <= t <= b:
            f = (t - a).total_seconds() / max(1, (b - a).total_seconds())
            return pa["lat"] + f * (pb["lat"] - pa["lat"]), pa["lon"] + f * (pb["lon"] - pa["lon"]), \
                pa["p"] + f * (pb["p"] - pa["p"])
    return None


def _label(ax, x, y, s, size=11, color="white"):
    ax.text(x, y, s, transform=PC, fontsize=size, fontweight="bold", color=color, ha="center", va="center",
            zorder=8, path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])


async def satellite(name, layer, style, t, box, title, sub, tracks, legend):
    img, (x0, x1, y0, y1) = await asyncio.to_thread(wms, layer, style, t, *box)
    f = await analysis_field(MODELS["ecmwf"], t.replace(minute=0), ("msl", 0))
    fig = Figure(figsize=(14.5, 10.6), dpi=110, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0.03, 0.07, 0.94, 0.83], projection=MERC)
    ax.set_extent([x0, x1, y0, y1], crs=MERC)
    ax.imshow(img, origin="upper", extent=[x0, x1, y0, y1], transform=MERC, zorder=0)
    ax.coastlines("50m", color="#facc15", lw=0.7, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#facc15", lw=0.4, zorder=3)
    gl = ax.gridlines(draw_labels=True, color="white", alpha=0.25, lw=0.5, x_inline=False, y_inline=False)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 8}
    if f is not None:
        sm = gaussian_filter(f.values / 100.0, 1.0)
        cs = ax.contour(f.lons, f.lats, sm, levels=np.arange(940, 1048, 4), colors="white", linewidths=0.8,
                        transform=PC, zorder=4)
        ax.clabel(cs, fmt="%d", fontsize=7, inline_spacing=1, colors="white")
    for sysname, tr in tracks.items():
        c = _centre(tr, t)
        if c and box[0] < c[1] < box[1] and box[2] < c[0] < box[3]:
            ax.plot(c[1], c[0], marker="x", ms=13, mew=3, color="black", transform=PC, zorder=7)
            ax.plot(c[1], c[0], marker="x", ms=11, mew=1.6, color="#facc15", transform=PC, zorder=7)
            _label(ax, c[1], c[0] + 1.3, f"{sysname} {c[2]:.0f} hPa")
    fig.text(0.03, 0.965, title, fontsize=14, fontweight="bold", color="#0d1b2a", va="top")
    fig.text(0.03, 0.93, sub, fontsize=9.5, color="#1b263b", va="top")
    fig.text(0.03, 0.04, legend, fontsize=8.5, color="#333", va="top")
    fig.text(0.97, 0.015, "Imagem © EUMETSAT (EUMETView) · análise ECMWF IFS (arquivo Forecast Workstation)",
             fontsize=7.5, color="#666", ha="right")
    fig.savefig(OUT / name, dpi=110, facecolor="white")
    print("ok", name)


def _tile_bounds(z, x, y):
    n = 2 ** z
    lon0, lon1 = x / n * 360 - 180, (x + 1) / n * 360 - 180
    lat = lambda yy: math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yy / n))))  # noqa: E731
    return lon0, lon1, lat(y + 1), lat(y)


def radar_mosaic(t, z=6):
    """Mosaic of the archived z6 RainViewer tiles of frame t → (RGBA array, merc extent)."""
    d = RADAR / str(int(t.replace(tzinfo=dt.timezone.utc).timestamp()))
    tiles = sorted(d.glob(f"{z}_*_*.png"))
    xs = sorted({int(p.stem.split("_")[1]) for p in tiles})
    ys = sorted({int(p.stem.split("_")[2]) for p in tiles})
    W, H = 512 * len(xs), 512 * len(ys)
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for p in tiles:
        _, x, y = map(int, p.stem.split("_"))
        canvas.paste(Image.open(p).convert("RGBA"), ((x - xs[0]) * 512, (y - ys[0]) * 512))
    lon0, _, _, lat1 = _tile_bounds(z, xs[0], ys[0])
    _, lon1, lat0, _ = _tile_bounds(z, xs[-1], ys[-1])
    (mx0, my0), (mx1, my1) = merc(lon0, lat0), merc(lon1, lat1)
    return np.asarray(canvas), (mx0, mx1, my0, my1)


def radar_panels(name, times, box, tracks):
    (bx0, by0), (bx1, by1) = merc(box[0], box[2]), merc(box[1], box[3])
    n = len(times)
    pw = 14.5 * 0.94 / n - 0.12                        # panel width (in)
    ph = pw * (by1 - by0) / (bx1 - bx0)                # keep the map's aspect
    H = ph + 1.35
    fig = Figure(figsize=(14.5, H), dpi=110, facecolor="white")
    FigureCanvasAgg(fig)
    for k, t in enumerate(times):
        ax = fig.add_axes([0.03 + k * (pw + 0.12) / 14.5, 0.35 / H, pw / 14.5, ph / H], projection=MERC)
        ax.set_extent([bx0, bx1, by0, by1], crs=MERC)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#ece7dc", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#d6e4f0", edgecolor="none", zorder=0)
        img, ext = radar_mosaic(t)
        ax.imshow(img, origin="upper", extent=list(ext), transform=MERC, zorder=2, interpolation="nearest")
        ax.coastlines("10m", color="#333", lw=0.6, zorder=3)
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), edgecolor="#555", lw=0.5, zorder=3)
        for sysname, tr in tracks.items():
            pts = [(p["lon"], p["lat"]) for p in tr]
            ax.plot(*zip(*pts), color="#111", lw=1.0, ls=(0, (3, 2)), transform=PC, zorder=4)
            c = _centre(tr, t)
            if c and box[0] < c[1] < box[1] and box[2] < c[0] < box[3]:
                ax.plot(c[1], c[0], marker="x", ms=11, mew=2.5, color="#b91c1c", transform=PC, zorder=6)
                ax.text(c[1] + 0.25, c[0] + 0.25, f"{sysname}\n{c[2]:.0f} hPa", transform=PC, fontsize=8.5,
                        fontweight="bold", color="#b91c1c", zorder=6,
                        path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        ax.text(0.03, 0.975, _fmt(t)[4:], transform=ax.transAxes, fontsize=10, fontweight="bold", va="top",
                color="#0d1b2a", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999", alpha=0.9), zorder=9)
    fig.text(0.03, 1 - 0.18 / H, "Charles: radar de precipitação durante a passagem por Portugal — 11 Nov 2026",
             fontsize=14, fontweight="bold", color="#0d1b2a", va="top")
    fig.text(0.03, 1 - 0.5 / H, "Mosaico RainViewer arquivado pela app (azul = chuva fraca; amarelo/laranja = moderada "
             "a forte; vermelho/magenta = muito forte; as riscas retas no Algarve são interferências) · ✕ = centro na "
             "análise ECMWF, tracejado = trajetória", fontsize=8.8, color="#1b263b", va="top")
    fig.text(0.98, 0.01, "Radar © RainViewer · trajetória: análises ECMWF IFS (arquivo Forecast Workstation)",
             fontsize=7.5, color="#666", ha="right")
    fig.savefig(OUT / name, dpi=110, facecolor="white")
    print("ok", name)


async def main():
    ev = archive.get_event("bruna")
    tr = await all_tracks(ev, ["ecmwf"])
    tracks = {k: v.get("ecmwf") or [] for k, v in tr.items()}
    T = dt.datetime
    await satellite("f02_satelite_massa_ar_0811.png", "msg_fes:rgb_airmass", "", T(2026, 11, 8, 12),
                    (-40, 6, 25, 55), "Bruna no máximo de intensidade — 08 Nov 2026, 12:00 UTC",
                    "Compósito RGB «massa de ar» (MSG/SEVIRI) com a pressão ao nível do mar da análise ECMWF "
                    "(isóbaras de 4 em 4 hPa) e o centro seguido pela app (✕)", tracks,
                    "Massa de ar: vermelho/laranja = ar seco de origem estratosférica (intrusão seca, PV elevada); "
                    "verde = massa de ar quente e húmida; azul/violeta = ar frio; branco/ocre = nuvens altas e espessas.")
    await satellite("f08_satelite_ir_1111.png", "mtg_fd:ir105_hrfi", "mtg_fd:mtg_fd_ir105_hrfi_style_01",
                    T(2026, 11, 11, 12), (-30, 6, 28, 52),
                    "Charles a chegar a Portugal e a Bruna a noroeste — 11 Nov 2026, 12:00 UTC",
                    "Infravermelho 10,5 µm realçado (MTG-I1/FCI) com a pressão ao nível do mar da análise ECMWF "
                    "e os centros seguidos pela app (✕)", tracks,
                    "Infravermelho realçado: azul → violeta → rosa = topos de nuvem cada vez mais frios e altos "
                    "(convecção profunda, bandas frontais ativas); cinzento = nuvens baixas e médias ou superfície.")
    radar_panels("f10_radar_charles_1111.png",
                 [T(2026, 11, 11, 10), T(2026, 11, 11, 12), T(2026, 11, 11, 14), T(2026, 11, 11, 16)],
                 (-11.5, -5.0, 36.0, 42.6), {"Charles": tracks.get("Charles", [])})


asyncio.run(main())
