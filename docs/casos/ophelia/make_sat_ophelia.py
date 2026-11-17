"""Figuras de satélite do furacão Ophelia (15–16/10/2017) a partir da NASA GIBS: MODIS em cor real
(com o fumo dos incêndios de Portugal e da Galiza), com a pressão ao nível do mar da reanálise ERA5
e a trajetória observada do NHC.

Corre a partir de backend/:  ../.venv/bin/python ../docs/casos/ophelia/make_sat_ophelia.py
"""
import asyncio
import datetime as dt
import io
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

from app import atcf  # noqa: E402
from app.models import MODELS  # noqa: E402
from app.cases import analysis_field  # noqa: E402

OUT = Path(__file__).resolve().parent / "figuras"
OUT.mkdir(exist_ok=True)
GIBS = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
PC = ccrs.PlateCarree()
STORM = "AL172017"


def gibs(layer, t, lon0, lon1, lat0, lat1, width=1600, fmt="image/png", time_fmt="%Y-%m-%dT%H:%M:%SZ"):
    h = int(width * (lat1 - lat0) / (lon1 - lon0))
    q = urllib.parse.urlencode({"SERVICE": "WMS", "REQUEST": "GetMap", "VERSION": "1.3.0", "LAYERS": layer,
                                "STYLES": "", "FORMAT": fmt, "TRANSPARENT": "true", "CRS": "EPSG:4326",
                                "BBOX": f"{lat0},{lon0},{lat1},{lon1}", "WIDTH": width, "HEIGHT": h,
                                "TIME": t.strftime(time_fmt)})
    with urllib.request.urlopen(f"{GIBS}?{q}", timeout=180) as r:
        return np.asarray(Image.open(io.BytesIO(r.read())).convert("RGBA"))


def best_at(t):
    bt = atcf.best_track(STORM)
    for a, b in zip(bt, bt[1:]):
        if a["time"] <= t <= b["time"]:
            f = (t - a["time"]) / (b["time"] - a["time"])
            return a["lat"] + f * (b["lat"] - a["lat"]), a["lon"] + f * (b["lon"] - a["lon"])
    return None


def _track(ax, t0, t1, until=None):
    bt = [p for p in atcf.best_track(STORM) if t0 <= p["time"] <= t1]
    ax.plot([p["lon"] for p in bt], [p["lat"] for p in bt], color="black", lw=2.2, transform=PC, zorder=6)
    ax.plot([p["lon"] for p in bt], [p["lat"] for p in bt], color="#facc15", lw=1.0, transform=PC, zorder=7)
    for p in bt:
        if p["time"].hour in (0, 12):
            ax.plot(p["lon"], p["lat"], "o", ms=4, color="#facc15", mec="black", transform=PC, zorder=8)


async def msl(t):
    return await analysis_field(MODELS["era5"], t, ("msl", 0))


async def modis(name, layer, day, t_msl, box, title, sub):
    img = gibs(layer, day, *box, fmt="image/jpeg", time_fmt="%Y-%m-%d")
    f = await msl(t_msl)
    lon0, lon1, lat0, lat1 = box
    fig = Figure(figsize=(14.5, 14.5 * (lat1 - lat0) / (lon1 - lon0) * 0.95 + 1.3), dpi=110, facecolor="white")
    FigureCanvasAgg(fig)
    H = fig.get_figheight()
    ax = fig.add_axes([0.04, 0.45 / H, 0.92, 1 - 1.25 / H], projection=PC)
    ax.set_extent(box, PC)
    ax.imshow(img, origin="upper", extent=box, transform=PC, zorder=0)
    ax.coastlines("50m", color="#facc15", lw=0.6, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#facc15", lw=0.4, zorder=3)
    if f is not None:
        cs = ax.contour(f.lons, f.lats, gaussian_filter(f.values / 100, 1), levels=np.arange(940, 1048, 4),
                        colors="white", linewidths=0.8, transform=PC, zorder=4)
        ax.clabel(cs, fmt="%d", fontsize=7, colors="white")
    _track(ax, day - dt.timedelta(days=2), day + dt.timedelta(days=2))
    c = best_at(t_msl)
    if c:
        ax.plot(c[1], c[0], "x", ms=14, mew=3, color="black", transform=PC, zorder=9)
        ax.plot(c[1], c[0], "x", ms=12, mew=1.6, color="#facc15", transform=PC, zorder=9)
    gl = ax.gridlines(draw_labels=True, color="white", alpha=0.3, lw=0.5)
    gl.top_labels = gl.right_labels = False
    fig.text(0.04, 1 - 0.2 / H, title, fontsize=14, fontweight="bold", color="#0d1b2a", va="top")
    fig.text(0.04, 1 - 0.55 / H, sub, fontsize=9.5, color="#1b263b", va="top")
    fig.text(0.96, 0.12 / H, "Imagem: NASA GIBS/Worldview (MODIS) · pressão: ERA5 (ECMWF/Copernicus) · trajetória: NHC",
             fontsize=7.5, color="#666", ha="right")
    fig.savefig(OUT / name, dpi=110, facecolor="white")
    print("ok", name)


async def imerg(name, times, box):
    lon0, lon1, lat0, lat1 = box
    n = len(times)
    pw = 14.5 * 0.95 / n - 0.1
    ph = pw * (lat1 - lat0) / (lon1 - lon0) / np.cos(np.radians((lat0 + lat1) / 2))
    H = ph + 1.9
    fig = Figure(figsize=(14.5, H), dpi=110, facecolor="white")
    FigureCanvasAgg(fig)
    proj = ccrs.Mercator()
    for k, t in enumerate(times):
        img = gibs("IMERG_Precipitation_Rate_30min", t, *box, width=700)
        ax = fig.add_axes([0.025 + k * (pw + 0.1) / 14.5, 0.75 / H, pw / 14.5, ph / H], projection=proj)
        ax.set_extent(box, PC)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#ece7dc", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#d6e4f0", edgecolor="none", zorder=0)
        ax.imshow(img, origin="upper", extent=box, transform=PC, zorder=2, interpolation="nearest")
        ax.coastlines("10m", color="#333", lw=0.6, zorder=3)
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), edgecolor="#555", lw=0.5, zorder=3)
        f = await msl(t.replace(minute=0))
        if f is not None:
            cs = ax.contour(f.lons, f.lats, gaussian_filter(f.values / 100, 1), levels=np.arange(940, 1048, 4),
                            colors="#111", linewidths=0.6, transform=PC, zorder=4)
            ax.clabel(cs, fmt="%d", fontsize=6)
        _track(ax, t - dt.timedelta(days=1), t + dt.timedelta(hours=12))
        c = best_at(t)
        if c:
            ax.plot(c[1], c[0], "x", ms=11, mew=2.5, color="#b91c1c", transform=PC, zorder=9)
        ax.text(0.03, 0.975, f"{t:%d/%m %H:%M} UTC", transform=ax.transAxes, fontsize=10, fontweight="bold", va="top",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999", alpha=0.9), zorder=10)
    leg = Image.open(io.BytesIO(urllib.request.urlopen(
        "https://gibs.earthdata.nasa.gov/legends/GPM_Precipitation_Rate_H.png", timeout=60).read()))
    lax = fig.add_axes([0.25, 0.05 / H, 0.5, 0.6 / H])
    lax.imshow(np.asarray(leg))
    lax.axis("off")
    fig.text(0.025, 1 - 0.18 / H, "Ophelia: chuva estimada por satélite (GPM IMERG, 30 min)",
             fontsize=13.5, fontweight="bold", color="#0d1b2a", va="top")
    fig.text(0.025, 1 - 0.48 / H, "Intensidade da precipitação (mm/h) · isóbaras da ERA5 · ✕ = centro na trajetória do NHC "
             "(interpolado) · linha = trajetória observada", fontsize=9, color="#1b263b", va="top")
    fig.text(0.975, 0.05 / H, "Dados: NASA GPM IMERG via GIBS · ERA5 · NHC", fontsize=7.5, color="#666", ha="right")
    fig.savefig(OUT / name, dpi=110, facecolor="white")
    print("ok", name)


async def main():
    from app import archive
    from app.cases import _use
    _use(archive.get_event("furaco-ophelia"))      # the event's region (Ireland is outside "nea")
    T = dt.datetime
    await modis("s1_modis_aqua_1510.png", "MODIS_Aqua_CorrectedReflectance_TrueColor", T(2017, 10, 15),
                T(2017, 10, 15, 14), (-28, 2, 33, 50),
                "Ophelia a oeste da Península e fumo dos incêndios — 15/10/2017, passagem do Aqua (~13–15 UTC)",
                "Cor real MODIS (Aqua) · isóbaras da ERA5 às 14 UTC (4 em 4 hPa) · ✕ = centro às 14 UTC na trajetória do "
                "NHC · linha amarela = trajetória observada")
    await modis("s2_modis_terra_1610.png", "MODIS_Terra_CorrectedReflectance_TrueColor", T(2017, 10, 16),
                T(2017, 10, 16, 11), (-22, 6, 38, 60),
                "Ophelia a chegar à Irlanda e fumo da Península Ibérica — 16/10/2017, passagem do Terra (~10–12 UTC)",
                "Cor real MODIS (Terra) · isóbaras da ERA5 às 11 UTC (4 em 4 hPa) · ✕ = centro às 11 UTC na trajetória do "
                "NHC · linha amarela = trajetória observada")


asyncio.run(main())
