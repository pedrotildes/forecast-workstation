"""Colour tables and default contour styles for every variable.

Palettes are hand-built discrete tables in the spirit of operational charts
(NWS/WPC, Meteociel, ECMWF) rather than generic matplotlib maps, so that each
colour step carries a meaning and values are readable from the colour bar.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap


def _interp(anchors: list[str], n: int, name="c") -> list:
    cm = LinearSegmentedColormap.from_list(name, anchors)
    return [cm(i) for i in np.linspace(0, 1, n)]


@dataclass
class Fill:
    levels: list[float]
    colors: list                    # len(levels) - 1 (+ under/over optional)
    extend: str = "both"
    under: str | None = None
    over: str | None = None
    transparent_under: bool = False  # mask values below levels[0]
    tick_every: int = 1

    def cmap_norm(self):
        cmap = ListedColormap(self.colors)
        if self.under:
            cmap.set_under(self.under)
        if self.over:
            cmap.set_over(self.over)
        return cmap, BoundaryNorm(self.levels, cmap.N, extend="neither")


@dataclass
class Contour:
    interval: float
    color: str = "#1a1a1a"
    width: float = 1.0
    style: str = "solid"            # solid | dashed | auto (dashed below `ref`)
    ref: float | None = None
    bold_every: float | None = None  # thicker line at multiples of this value
    labels: bool = True
    fmt: str = "%g"
    extrema: bool = False           # H / L markers
    vmin: float | None = None
    vmax: float | None = None
    levels: list | None = None      # explicit isoline values (overrides interval)


@dataclass
class Style:
    fill: Fill | None = None
    contour: Contour | None = None
    level_fill: dict[int, Fill] = field(default_factory=dict)
    level_contour: dict[int, Contour] = field(default_factory=dict)

    def get_fill(self, level=0):
        return self.level_fill.get(level, self.fill)

    def get_contour(self, level=0):
        return self.level_contour.get(level, self.contour)


def rng(a, b, s):
    return [round(x, 6) for x in np.arange(a, b + s / 2, s)]


# --------------------------------------------------------------- palettes
TEMP_ANCHORS = ["#f0d9ff", "#c77dff", "#7b2cbf", "#3a0ca3", "#2f5dd3", "#4ea8de",
                "#90e0ef", "#d8f3dc", "#95d5b2", "#52b788", "#f9e076", "#f4a261",
                "#e76f51", "#d62828", "#9d0208", "#6a040f", "#3d0010", "#ff9ecd"]


def temp_fill(lo, hi, step):
    lv = rng(lo, hi, step)
    return Fill(lv, _interp(TEMP_ANCHORS, len(lv) - 1, "temp"), under="#ffffff",
                over="#ffd6ec", tick_every=max(1, int(round(4 / step)) if step < 4 else 1))


PRECIP_LEVELS = [0.1, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200]
PRECIP_COLORS = ["#c6e9fb", "#9fd3f7", "#6cb6ef", "#3c8ee0", "#1f5fc7", "#14a33a",
                 "#5bc236", "#a5dc2b", "#f2ec2c", "#f9c623", "#f99d1c", "#f2661d",
                 "#e2231a", "#b5121b", "#8b0a50", "#b02fa3", "#d77fe0", "#f0c4f2"]

WIND_LEVELS = [10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200]
WIND_COLORS = _interp(["#e8f4fa", "#a9d6e5", "#61a5c2", "#2a9d8f", "#57cc99", "#c7f9cc",
                       "#ffe66d", "#ffb703", "#fb8500", "#e63946", "#a4133c", "#7b2cbf",
                       "#c77dff", "#f2c6ff"], len(WIND_LEVELS) - 1, "wind")

SFC_WIND_LEVELS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80]
SFC_WIND_COLORS = _interp(["#e8f4fa", "#9ad1d4", "#57cc99", "#c7f9cc", "#ffe66d", "#ffb703",
                           "#fb8500", "#e63946", "#a4133c", "#7b2cbf", "#c77dff", "#f2c6ff"],
                          len(SFC_WIND_LEVELS) - 1, "sfcwind")

GUST_LEVELS = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 140, 160, 180, 200]
GUST_COLORS = _interp(["#d9ed92", "#99d98c", "#52b69a", "#ffe66d", "#ffb703", "#fb8500",
                       "#e63946", "#a4133c", "#7b2cbf", "#c77dff", "#f2c6ff"],
                      len(GUST_LEVELS) - 1, "gust")

CAPE_LEVELS = [50, 100, 250, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 3500, 4000, 5000]
CAPE_COLORS = _interp(["#e0f2e9", "#a3d9b1", "#5cb85c", "#d9e021", "#fcd116", "#f8961e",
                       "#f3722c", "#d62828", "#9d0208", "#7b2cbf", "#c77dff", "#f2c6ff"],
                      len(CAPE_LEVELS) - 1, "cape")

RH_LEVELS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
RH_COLORS = ["#7f4f24", "#a47148", "#c9a27e", "#e6ccb2", "#f5ebe0", "#ffffff",
             "#d8f3dc", "#95d5b2", "#52b788", "#2d6a4f", "#1b4332"]

CLOUD_LEVELS = [5, 15, 30, 45, 60, 75, 90, 100.01]
CLOUD_COLORS = ["#f2f2f2", "#d9d9d9", "#bfbfbf", "#a6a6a6", "#8c8c8c", "#737373", "#595959"]

REFL_LEVELS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
REFL_COLORS = ["#04e9e7", "#019ff4", "#0300f4", "#02fd02", "#01c501", "#008e00", "#fdf802",
               "#e5bc00", "#fd9500", "#fd0000", "#d40000", "#bc0000", "#f800fd", "#9854c6"]

PWAT_LEVELS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70]
PWAT_COLORS = _interp(["#f6efe2", "#d8c3a5", "#b5d99c", "#56c596", "#2a9d8f", "#1d70a2",
                       "#3a0ca3", "#7209b7", "#b5179e", "#f72585"], len(PWAT_LEVELS) - 1, "pw")


def diverging(lim, step, neg="#08306b", pos="#67000d", zero_gap=True):
    lv = rng(-lim, lim, step)
    n = len(lv) - 1
    cols = _interp([neg, "#2171b5", "#9ecae1", "#ffffff", "#ffffff", "#fc9272", "#cb181d", pos], n)
    if zero_gap:
        mid = n // 2
        cols[mid - 1] = cols[mid] = (1, 1, 1, 0)
    return Fill(lv, cols, under=neg, over=pos)


Z500_LEVELS = rng(492, 600, 4)
Z500_COLORS = _interp(["#6a00a8", "#3a0ca3", "#1e3fbf", "#3d7ff0", "#5ec6f0", "#7ff0c9",
                       "#c4f07f", "#f5e663", "#f8b54a", "#f07a3a", "#d93a3a", "#a00d2d",
                       "#6b001b"], len(Z500_LEVELS) - 1, "z500")

THETAE_LEVELS = rng(280, 352, 2)
THETAE_COLORS = _interp(TEMP_ANCHORS[2:-1], len(THETAE_LEVELS) - 1, "the")


STYLES: dict[str, Style] = {
    "msl": Style(contour=Contour(4, "#111111", 1.1, bold_every=20, extrema=True, fmt="%d")),
    "t2m": Style(fill=temp_fill(-30, 46, 2), contour=Contour(4, "#7a0000", 0.7, "auto", ref=0)),
    "d2m": Style(fill=temp_fill(-30, 30, 2), contour=Contour(4, "#00553a", 0.7, "auto", ref=0)),
    "rh2m": Style(fill=Fill(RH_LEVELS, RH_COLORS, extend="neither"),
                  contour=Contour(20, "#2d6a4f", 0.7)),
    "wind10": Style(fill=Fill(SFC_WIND_LEVELS, SFC_WIND_COLORS, extend="max", over="#ffe0ff",
                              transparent_under=True), contour=Contour(10, "#333", 0.7)),
    "gust": Style(fill=Fill(GUST_LEVELS, GUST_COLORS, extend="max", over="#ffe0ff",
                            transparent_under=True), contour=Contour(20, "#552", 0.8)),
    "precip": Style(fill=Fill(PRECIP_LEVELS, PRECIP_COLORS, extend="max", over="#ffe6ff",
                              transparent_under=True), contour=Contour(5, "#1f5fc7", 0.7)),
    "tcc": Style(fill=Fill(CLOUD_LEVELS, CLOUD_COLORS, extend="neither", transparent_under=True),
                 contour=Contour(25, "#555", 0.6)),
    "lcc": Style(fill=Fill(CLOUD_LEVELS, CLOUD_COLORS, extend="neither", transparent_under=True)),
    "mcc": Style(fill=Fill(CLOUD_LEVELS, CLOUD_COLORS, extend="neither", transparent_under=True)),
    "hcc": Style(fill=Fill(CLOUD_LEVELS, CLOUD_COLORS, extend="neither", transparent_under=True)),
    "pwat": Style(fill=Fill(PWAT_LEVELS, PWAT_COLORS, extend="max", over="#ffd1f0",
                            transparent_under=True), contour=Contour(10, "#1d3557", 0.7)),
    "refc": Style(fill=Fill(REFL_LEVELS, REFL_COLORS, extend="max", over="#ffffff",
                            transparent_under=True)),
    "cape": Style(fill=Fill(CAPE_LEVELS, CAPE_COLORS, extend="max", over="#ffe0ff",
                            transparent_under=True), contour=Contour(500, "#6a040f", 0.7, vmin=500)),
    "mlcape": Style(fill=Fill(CAPE_LEVELS, CAPE_COLORS, extend="max", over="#ffe0ff",
                              transparent_under=True), contour=Contour(500, "#6a040f", 0.7, vmin=500)),
    "mucape": Style(fill=Fill(CAPE_LEVELS, CAPE_COLORS, extend="max", over="#ffe0ff",
                              transparent_under=True), contour=Contour(500, "#6a040f", 0.7, vmin=500)),
    "cin": Style(fill=Fill([-500, -300, -200, -150, -100, -75, -50, -25, -10],
                           _interp(["#08306b", "#2171b5", "#6baed6", "#c6dbef", "#f7fbff"], 8),
                           extend="min", under="#041a3d")),
    "shear": Style(fill=Fill(rng(20, 80, 5), _interp(["#e9f5db", "#b5e48c", "#52b69a", "#168aad",
                                                       "#1e6091", "#7b2cbf", "#c77dff", "#ff70a6"], 12),
                             extend="max", over="#ffd1e8", transparent_under=True),
                   contour=Contour(10, "#1e6091", 0.7, vmin=20)),
    "lapse": Style(fill=Fill(rng(4, 9.5, 0.5), _interp(["#f1faee", "#a8dadc", "#ffe66d", "#fb8500",
                                                          "#d62828", "#7b2cbf"], 11),
                             extend="both", under="#e5e5e5", over="#3c096c"),
                   contour=Contour(0.5, "#333", 0.6, vmin=6.5)),
    "tt": Style(fill=Fill(rng(40, 62, 2), _interp(["#e9f5db", "#ffe66d", "#fb8500", "#d62828",
                                                     "#7b2cbf", "#f2c6ff"], 11),
                          extend="max", over="#ffe0ff", transparent_under=True),
                contour=Contour(4, "#552", 0.6, vmin=44)),
    "kindex": Style(fill=Fill(rng(15, 45, 3), _interp(["#e9f5db", "#ffe66d", "#fb8500", "#d62828",
                                                         "#7b2cbf", "#f2c6ff"], 10),
                              extend="max", over="#ffe0ff", transparent_under=True),
                    contour=Contour(5, "#552", 0.6, vmin=20)),
    "frzl": Style(fill=Fill(rng(0, 5000, 250), _interp(["#3a0ca3", "#4ea8de", "#90e0ef", "#d8f3dc",
                                                          "#f9e076", "#e76f51", "#9d0208"], 20),
                            extend="max", over="#6a040f"),
                  contour=Contour(500, "#222", 0.7)),
    "gh": Style(fill=Fill(Z500_LEVELS, Z500_COLORS, under="#40005f", over="#3d000f"),
                contour=Contour(6, "#111111", 1.2, bold_every=12, extrema=True, fmt="%d"),
                level_fill={
                    850: Fill(rng(120, 168, 2), _interp(Z500_COLORS[::3], 24), under="#40005f", over="#3d000f"),
                    700: Fill(rng(270, 318, 2), _interp(Z500_COLORS[::3], 24), under="#40005f", over="#3d000f"),
                    300: Fill(rng(840, 972, 6), _interp(Z500_COLORS[::3], 22), under="#40005f", over="#3d000f"),
                    250: Fill(rng(936, 1080, 6), _interp(Z500_COLORS[::3], 24), under="#40005f", over="#3d000f"),
                    200: Fill(rng(1080, 1236, 6), _interp(Z500_COLORS[::3], 26), under="#40005f", over="#3d000f"),
                },
                level_contour={
                    1000: Contour(4, "#111", 1.1, extrema=True, fmt="%d"),
                    925: Contour(3, "#111", 1.1, extrema=True, fmt="%d"),
                    850: Contour(3, "#111", 1.1, bold_every=15, extrema=True, fmt="%d"),
                    700: Contour(3, "#111", 1.1, extrema=True, fmt="%d"),
                    300: Contour(12, "#111", 1.1, extrema=True, fmt="%d"),
                    250: Contour(12, "#111", 1.1, extrema=True, fmt="%d"),
                    200: Contour(12, "#111", 1.1, extrema=True, fmt="%d"),
                }),
    "t": Style(fill=temp_fill(-40, 30, 2), contour=Contour(4, "#b00000", 0.8, "auto", ref=0,
                                                        bold_every=None),
               level_fill={500: temp_fill(-48, 0, 2), 300: temp_fill(-64, -24, 2),
                           250: temp_fill(-68, -32, 2), 200: temp_fill(-72, -40, 2),
                           700: temp_fill(-32, 20, 2), 850: temp_fill(-28, 32, 2),
                           925: temp_fill(-24, 36, 2), 1000: temp_fill(-20, 40, 2)},
               level_contour={500: Contour(4, "#b00000", 0.8, "auto", ref=-20)}),
    "rh": Style(fill=Fill(RH_LEVELS, RH_COLORS, extend="neither"), contour=Contour(20, "#2d6a4f", 0.7)),
    "td": Style(fill=temp_fill(-40, 24, 2), contour=Contour(4, "#00553a", 0.7, "auto", ref=0)),
    "tdd": Style(fill=Fill([0, 1, 2, 3, 4, 6, 8, 10, 15, 20, 30],
                           ["#1b4332", "#2d6a4f", "#52b788", "#95d5b2", "#d8f3dc", "#ffffff",
                            "#f5ebe0", "#e6ccb2", "#c9a27e", "#a47148"], extend="max", over="#7f4f24"),
                 contour=Contour(2, "#2d6a4f", 0.6, vmax=6)),
    "thetae": Style(fill=Fill(THETAE_LEVELS, THETAE_COLORS, under="#3a0ca3", over="#ff9ecd"),
                    contour=Contour(4, "#3c096c", 0.7)),
    "wspd": Style(fill=Fill(WIND_LEVELS, WIND_COLORS, extend="max", over="#ffe6ff",
                            transparent_under=True), contour=Contour(20, "#333", 0.6, vmin=40)),
    "w": Style(fill=diverging(2.0, 0.2, neg="#003d1f", pos="#5e2c00"), contour=Contour(0.5, "#444", 0.5)),
    "vort": Style(fill=diverging(30, 3), contour=Contour(5, "#444", 0.6)),
    "div": Style(fill=diverging(10, 1), contour=Contour(2, "#444", 0.6)),
    "tadv": Style(fill=diverging(6, 0.5), contour=Contour(1, "#444", 0.5)),
    "q": Style(fill=Fill(rng(0, 16, 1), _interp(["#f6efe2", "#b5d99c", "#2a9d8f", "#1d70a2",
                                                   "#3a0ca3", "#b5179e"], 16), extend="max",
                         over="#f72585", transparent_under=False), contour=Contour(2, "#1d3557", 0.6)),
    "thick": Style(fill=Fill(rng(500, 600, 4), _interp(Z500_COLORS, 25), under="#40005f", over="#3d000f"),
                   contour=Contour(6, "#c1121f", 0.9, "dashed", bold_every=None, fmt="%d")),
}

def _anom(lim=12, step=1):
    lv = rng(-lim, lim, step)
    cols = _interp(["#08306b", "#2171b5", "#6baed6", "#c6dbef", "#f7f7f7", "#f7f7f7",
                    "#fcbba1", "#fb6a4a", "#cb181d", "#67000d"], len(lv) - 1, "anom")
    return Fill(lv, cols, under="#041a3d", over="#3d0007", tick_every=2)


STYLES.update({
    "tmax_day": Style(fill=temp_fill(-12, 46, 2), contour=Contour(4, "#7a0000", 0.7)),
    "tmin_day": Style(fill=temp_fill(-26, 32, 2), contour=Contour(4, "#00305a", 0.7)),
    "tmax_anom": Style(fill=_anom(), contour=Contour(2, "#333", 0.6, levels=[-5, 5])),
    "tmin_anom": Style(fill=_anom(), contour=Contour(2, "#333", 0.6, levels=[-5, 5])),
})

# Default fill palette for the "Omega" variable: ascent (negative) in green tones
STYLES["w"].fill = Fill(rng(-2.0, 2.0, 0.2),
                        _interp(["#004b23", "#2d6a4f", "#74c69d", "#d8f3dc", "#ffffff", "#ffffff",
                                 "#fde2e4", "#f4978e", "#c9184a", "#590d22"], 20),
                        under="#002611", over="#3d0614")
for _i in (9, 10):
    STYLES["w"].fill.colors[_i] = (1, 1, 1, 0)


def style_for(var_id: str) -> Style:
    return STYLES.get(var_id, Style(contour=Contour(5)))
