"""Cartopy/Matplotlib chart renderer.

``render_job`` is a pure function of plain data (numpy arrays + small dicts) so
it can run in a worker process; it returns PNG bytes and the position of the
map axes inside the image (used by the client to turn clicks into lat/lon).
"""
from __future__ import annotations

import io
import warnings
from functools import lru_cache

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import numpy as np
import shapely.geometry as sgeom

matplotlib.use("Agg")
from matplotlib import patheffects as pe  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter, zoom  # noqa: E402

from ..regions import REGIONS  # noqa: E402
from .styles import Contour, Fill, style_for  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

PC = ccrs.PlateCarree()
DPI = 130
FIG_W = 12.6                      # inches -> ~1640 px
FONT = "DejaVu Sans"
matplotlib.rcParams.update({"font.family": FONT, "font.size": 9,
                            "axes.linewidth": 0.8})

LAND = "#efebe3"
OCEAN = "#d6e4ef"
COAST = "#262626"


def _upsample(lats, lons, arr, f):
    if f <= 1:
        return lats, lons, arr
    nan = np.isnan(arr)
    fill = np.where(nan, np.nanmean(arr) if np.isfinite(arr).any() else 0, arr)
    a = zoom(fill, f, order=3)
    a = gaussian_filter(a, sigma=0.45 * f)
    if nan.any():
        a[zoom(nan.astype(float), f, order=0) > 0.5] = np.nan
    la = np.linspace(lats[0], lats[-1], a.shape[0])
    lo = np.linspace(lons[0], lons[-1], a.shape[1])
    return la, lo, a


def _contour_levels(arr, c: Contour):
    if c.levels:
        return [x for x in c.levels if np.nanmin(arr) <= x <= np.nanmax(arr)]
    lo = np.nanmin(arr) if c.vmin is None else max(c.vmin, np.nanmin(arr))
    hi = np.nanmax(arr) if c.vmax is None else min(c.vmax, np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return []
    a = np.floor(lo / c.interval) * c.interval
    b = np.ceil(hi / c.interval) * c.interval
    lv = np.arange(a, b + c.interval / 2, c.interval)
    if c.vmin is not None:
        lv = lv[lv >= c.vmin]
    if c.vmax is not None:
        lv = lv[lv <= c.vmax]
    if len(lv) > 120:
        lv = lv[:: int(np.ceil(len(lv) / 120))]
    return lv


def _extrema(ax, lats, lons, arr, crs, extent, size_frac=0.09, fmt="%d", color_hi="#0b3c8c",
             color_lo="#b3001b"):
    ny, nx = arr.shape
    size = max(5, int(max(ny, nx) * size_frac))
    a = np.where(np.isnan(arr), np.nanmean(arr), arr)
    mx = (maximum_filter(a, size=size, mode="nearest") == a)
    mn = (minimum_filter(a, size=size, mode="nearest") == a)
    edge = max(2, size // 3)
    for mask, letter, col in ((mx, "A", color_hi), (mn, "B", color_lo)):
        jj, ii = np.where(mask)
        placed = []
        for j, i in zip(jj, ii):
            if j < edge or i < edge or j >= ny - edge or i >= nx - edge:
                continue
            x, y = crs.transform_point(lons[i], lats[j], PC)
            if not (extent[0] < x < extent[1] and extent[2] < y < extent[3]):
                continue
            if any(abs(x - px) < (extent[1] - extent[0]) * 0.05 and
                   abs(y - py) < (extent[3] - extent[2]) * 0.05 for px, py in placed):
                continue
            placed.append((x, y))
            stroke = [pe.withStroke(linewidth=2.6, foreground="white")]
            ax.text(x, y, letter, color=col, fontsize=17, fontweight="bold", ha="center",
                    va="center", transform=crs, zorder=9, path_effects=stroke, clip_on=True)
            ax.text(x, y, "\n\n" + (fmt % a[j, i]), color=col, fontsize=7.5, fontweight="bold",
                    ha="center", va="center", transform=crs, zorder=9, path_effects=stroke,
                    clip_on=True)


def _draw_fill(ax, lats, lons, arr, fill: Fill, opacity=1.0):
    cmap, norm = fill.cmap_norm()
    data = arr
    extend = fill.extend
    if fill.transparent_under:
        data = np.ma.masked_less(arr, fill.levels[0])
        extend = "max" if fill.extend in ("max", "both") else "neither"
        if np.ma.count(data) == 0:  # nothing to draw, keep the colour bar
            return cmap, norm, extend
    ax.contourf(lons, lats, data, levels=fill.levels, cmap=cmap, norm=norm, extend=extend,
                transform=PC, zorder=1, alpha=opacity, antialiased=True)
    return cmap, norm, extend


def _draw_contour(ax, lats, lons, arr, c: Contour, region_crs, extent, override=None):
    override = override or {}
    c = Contour(**{**c.__dict__, **{k: v for k, v in override.items() if v is not None}})
    lv = _contour_levels(arr, c)
    if len(lv) == 0:
        return
    if c.style == "dashed":
        styles = ["--"] * len(lv)
    elif c.style == "auto" and c.ref is not None:
        styles = ["--" if v < c.ref else "-" for v in lv]
    else:
        styles = ["-"] * len(lv)
    widths = [c.width * (1.9 if c.bold_every and abs(v / c.bold_every - round(v / c.bold_every)) < 1e-6
                         else 1.0) for v in lv]
    if c.style == "auto" and c.ref is not None:
        widths = [w * (2.0 if abs(v - c.ref) < 1e-6 else 1.0) for w, v in zip(widths, lv)]
    cs = ax.contour(lons, lats, arr, levels=lv, colors=c.color, linewidths=widths,
                    linestyles=styles, transform=PC, zorder=4)
    if c.labels:
        try:
            labels = ax.clabel(cs, cs.levels, inline=True, fontsize=7.2, fmt=c.fmt,
                               inline_spacing=2)
            r, g, b = to_rgb(c.color)
            halo = "#111111" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.6 else "white"
            for t in labels:
                t.set_path_effects([pe.withStroke(linewidth=2, foreground=halo)])
                t.set_fontweight("bold")
        except Exception:  # noqa: BLE001
            pass
    if c.extrema:
        _extrema(ax, lats, lons, arr, region_crs, extent, fmt=c.fmt)


def _draw_wind(ax, lats, lons, u, v, layer, region):
    style = layer.get("style", "barbs")
    color = layer.get("color") or "#111111"
    density = float(layer.get("density", 1.0))
    if style == "streamlines":
        spd = np.hypot(u, v)
        ax.streamplot(lons, lats, u, v, transform=PC, density=1.6 * density, color=color,
                      linewidth=0.45 + 1.4 * spd / max(1e-6, np.nanmax(spd)),
                      arrowsize=0.8, zorder=5)
        return
    n = max(8, int(region.barbs * density))
    if style == "arrows":
        q = ax.quiver(lons, lats, u, v, transform=PC, regrid_shape=n, color=color, zorder=5,
                      scale_units="inches", scale=60, width=0.0016, headwidth=4, headlength=4.5)
        ax.quiverkey(q, 0.985, 1.012, 20, "20 kt", labelpos="W", coordinates="axes",
                     fontproperties={"size": 7})
        return
    ax.barbs(lons, lats, u, v, transform=PC, regrid_shape=n, length=5.4, linewidth=0.55,
             color=color, zorder=5, sizes={"emptybarb": 0.0, "spacing": 0.16, "height": 0.42})


@lru_cache(maxsize=16)
def _land_geoms(region_id: str):
    """Land polygons pre-clipped to the region (clipping the global 10m set is slow)."""
    region = REGIONS[region_id]
    d = region.download_domain
    box = sgeom.box(d.lon0, d.lat0, d.lon1, d.lat1)
    feat = cfeature.NaturalEarthFeature("physical", "land", region.detail)
    out = []
    for g in feat.intersecting_geometries((d.lon0, d.lon1, d.lat0, d.lat1)):
        c = g.intersection(box)
        if not c.is_empty:
            out.append(c.simplify(0.002 if region.detail == "10m" else 0.0))
    return out


def _draw_values(ax, lats, lons, arr, region, fmt):
    """Plot the field value at the region's reference cities."""
    from ..grib import Field
    from ..regions import CITIES
    f = Field(lats, lons, arr)
    stroke = [pe.withStroke(linewidth=2.8, foreground="white")]
    for name, la, lo in CITIES.get(region.id, []):
        if not (lats[0] <= la <= lats[-1] and lons[0] <= lo <= lons[-1]):
            continue
        v = f.at(la, lo)
        if not np.isfinite(v):
            continue
        ax.plot(lo, la, "o", ms=2.2, color="#111", transform=PC, zorder=8)
        ax.text(lo, la, fmt % v, transform=PC, fontsize=8.6, fontweight="bold", color="#111",
                ha="center", va="bottom", zorder=9, path_effects=stroke, clip_on=True)
        if region.admin1 or region.id in ("azores", "madeira"):
            ax.text(lo, la, name, transform=PC, fontsize=6.2, color="#333", ha="center", va="top",
                    zorder=9, path_effects=stroke, clip_on=True)


def _features(ax, region):
    ax.set_facecolor(OCEAN)
    ax.add_geometries(_land_geoms(region.id), crs=PC, facecolor=LAND, edgecolor="none", zorder=0)


def _overlay_features(ax, region):
    scale = region.detail
    ax.add_feature(cfeature.NaturalEarthFeature("physical", "coastline", scale),
                   facecolor="none", edgecolor=COAST, linewidth=0.75, zorder=3)
    ax.add_feature(cfeature.NaturalEarthFeature("cultural", "admin_0_boundary_lines_land", scale),
                   facecolor="none", edgecolor="#333333", linewidth=0.55, linestyle="-", zorder=3)
    ax.add_feature(cfeature.NaturalEarthFeature("physical", "lakes", scale),
                   facecolor="none", edgecolor=COAST, linewidth=0.4, zorder=3)
    if region.admin1:
        ax.add_feature(cfeature.NaturalEarthFeature("cultural", "admin_1_states_provinces_lines",
                                                    "10m"),
                       facecolor="none", edgecolor="#555555", linewidth=0.35, zorder=3)
    g = region.grid
    gl = ax.gridlines(crs=PC, draw_labels=True, linewidth=0.35, color="#666666", alpha=0.55,
                      linestyle=(0, (3, 3)), x_inline=False, y_inline=False, zorder=6,
                      xlocs=np.arange(-180, 180.01, g), ylocs=np.arange(-90, 90.01, g),
                      rotate_labels=False)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 7, "color": "#333"}


def render_job(job: dict) -> tuple[bytes, list[float]]:
    region = REGIONS[job["region"]]
    crs = region.crs
    extent = region.proj_extent
    lats, lons = job["lats"], job["lons"]
    fills = [ly for ly in job["layers"] if ly["type"] == "fill"]

    margin_l, margin_r = 0.42, 0.18
    head_h = 1.02
    foot_h = (0.88 if fills else 0.0) + 0.24
    map_w = FIG_W - margin_l - margin_r
    map_h = map_w * region.aspect
    fig_h = head_h + map_h + foot_h
    fig = Figure(figsize=(FIG_W, fig_h), dpi=DPI, facecolor="white")
    FigureCanvasAgg(fig)
    rect = [margin_l / FIG_W, foot_h / fig_h, map_w / FIG_W, map_h / fig_h]
    ax = fig.add_axes(rect, projection=crs)
    ax.set_extent(extent, crs=crs)
    _features(ax, region)

    # Fine grids on large domains: keep ~2 grid points per output pixel column at most.
    res = abs(float(lats[1] - lats[0])) if len(lats) > 1 else 0.25
    stride = max(1, int(np.ceil(max(len(lons) / 750, len(lats) / 560))))
    if stride > 1:
        lats, lons = lats[::stride], lons[::stride]
        for ly in job["layers"]:
            for k in ("data", "u", "v"):
                if k in ly and ly[k] is not None:
                    ly[k] = ly[k][::stride, ::stride]
        res *= stride
    up = max(1, int(round(region.upsample * min(1.0, res / 0.25))))
    cbar_specs = []
    for ly in job["layers"]:
        t = ly["type"]
        if t == "fill":
            st = ly.get("fill_style") or style_for(ly["var"]).get_fill(ly.get("level", 0))
            if st is None:
                continue
            la, lo, a = _upsample(lats, lons, ly["data"], up)
            cmap, norm, extend = _draw_fill(ax, la, lo, a, st, ly.get("opacity", 1.0))
            cbar_specs.append((cmap, norm, st, extend, ly["label"]))
        elif t == "contour":
            st = style_for(ly["var"]).get_contour(ly.get("level", 0)) or Contour(5)
            if region.contour_scale != 1 and ly["var"] in ("msl", "gh", "thick") and not ly.get("interval"):
                st = Contour(**{**st.__dict__, "interval": st.interval * region.contour_scale,
                                "bold_every": None, "extrema": False})
            la, lo, a = _upsample(lats, lons, ly["data"], min(up, 2))
            _draw_contour(ax, la, lo, a, st, crs, extent,
                          {k: ly.get(k) for k in ("interval", "color", "width", "style",
                                                  "extrema", "levels", "labels", "fmt")})
        elif t == "wind":
            _draw_wind(ax, lats, lons, ly["u"], ly["v"], ly, region)
        elif t == "values":
            _draw_values(ax, lats, lons, ly["data"], region, ly.get("fmt", "%.0f"))

    _overlay_features(ax, region)
    for sp in ax.spines.values():
        sp.set_linewidth(1.0)

    # ---- header
    h = job["header"]
    top = 1 - 0.12 / fig_h
    fig.text(margin_l / FIG_W, top, h["left1"], ha="left", va="top", fontsize=12.5,
             fontweight="bold", color="#0d1b2a")
    fig.text(margin_l / FIG_W, top - 0.30 / fig_h, h["left2"], ha="left", va="top", fontsize=9.6,
             color="#1b263b")
    fig.text(margin_l / FIG_W, top - 0.56 / fig_h, h["layers"], ha="left", va="top", fontsize=8.8,
             color="#3a4a5c", wrap=False)
    fig.text(1 - margin_r / FIG_W, top, h["right1"], ha="right", va="top", fontsize=12.5,
             fontweight="bold", color="#9d0208")
    fig.text(1 - margin_r / FIG_W, top - 0.30 / fig_h, h["right2"], ha="right", va="top",
             fontsize=9.6, color="#1b263b")

    # ---- colour bars
    if cbar_specs:
        n = len(cbar_specs)
        gap = 0.35
        cb_w = (map_w - gap * (n - 1)) / n
        for i, (cmap, norm, st, extend, label) in enumerate(cbar_specs):
            x = margin_l + i * (cb_w + gap)
            cax = fig.add_axes([x / FIG_W, 0.50 / fig_h, cb_w / FIG_W, 0.15 / fig_h])
            from matplotlib.cm import ScalarMappable
            sm = ScalarMappable(norm=norm, cmap=cmap)
            max_ticks = max(4, int(cb_w * (2.6 if max(abs(x) for x in st.levels) >= 100 else 3.4)))
            every = max(st.tick_every, int(np.ceil(len(st.levels) / max_ticks)))
            ticks = st.levels[::every]
            cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=ticks,
                              extend=extend if extend != "neither" else "neither",
                              spacing="uniform", drawedges=True)
            cb.outline.set_linewidth(0.6)
            cb.dividers.set_linewidth(0.3)
            cb.ax.tick_params(labelsize=7, length=2, pad=1.5)
            cb.ax.set_xticklabels([f"{t:g}" for t in ticks])
            cb.set_label(label, fontsize=8, labelpad=2)
            cb.ax.xaxis.set_label_position("top")

    fig.text(margin_l / FIG_W, 0.07 / fig_h, job.get("footer", ""), ha="left", va="bottom",
             fontsize=6.8, color="#555")
    fig.text(1 - margin_r / FIG_W, 0.07 / fig_h, "Forecast Workstation", ha="right",
             va="bottom", fontsize=6.8, color="#555", style="italic")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor="white")
    # axes rect relative to the image, measured from the top-left corner
    l, b, w, hh = rect
    return buf.getvalue(), [l, 1 - (b + hh), w, hh]
