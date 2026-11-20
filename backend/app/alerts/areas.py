"""Warning areas: the 18 districts of mainland Portugal, Madeira (4 areas) and the Azores
(3 island groups), with the codes IPMA uses in its warnings (``idAreaAviso``).

Each area has *sample points* where model fields are read:

* districts — every node of the 0.25° lattice (the ECMWF/GFS grid) inside the district,
  so a district of 2 500 km² has about 4 points and Beja about 17;
* islands — the islands are only a few grid cells wide, so the points are towns and
  inland sites, read by bilinear interpolation (the same way the app reads a point).

The value of a field in an area is the maximum (minimum for cold) over its points: a
warning is issued for a district when the threshold is expected in part of it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class Area:
    code: str                      # IPMA idAreaAviso
    name: str
    group: str                     # continente | madeira | acores
    region: str                    # map region id (regions.REGIONS)
    zone: str = ""                 # sub-group used by the criteria (litoral/interior…)
    towns: tuple = ()              # (name, lat, lon) sample points for islands
    label: tuple = (0.0, 0.0)      # lat, lon of the label on maps
    points: tuple = field(default=(), compare=False)   # ((lat, lon), …)


# Mainland districts: IPMA code -> Natural Earth name. The zone splits coastal and inland
# districts (used only for the text of the messages; thresholds come from criteria.py).
DISTRICTS = {
    "VCT": ("Viana do Castelo", "Viana do Castelo", "norte"),
    "BRG": ("Braga", "Braga", "norte"),
    "PTO": ("Porto", "Porto", "norte"),
    "VRL": ("Vila Real", "Vila Real", "norte"),
    "BGC": ("Bragança", "Bragança", "norte"),
    "AVR": ("Aveiro", "Aveiro", "centro"),
    "VIS": ("Viseu", "Viseu", "centro"),
    "GDA": ("Guarda", "Guarda", "centro"),
    "CBR": ("Coimbra", "Coimbra", "centro"),
    "CBO": ("Castelo Branco", "Castelo Branco", "centro"),
    "LRA": ("Leiria", "Leiria", "centro"),
    "STM": ("Santarém", "Santarém", "sul"),
    "PTG": ("Portalegre", "Portalegre", "sul"),
    "LSB": ("Lisboa", "Lisboa", "sul"),
    "STB": ("Setúbal", "Setúbal", "sul"),
    "EVR": ("Évora", "Évora", "sul"),
    "BJA": ("Beja", "Beja", "sul"),
    "FAR": ("Faro", "Faro", "sul"),
}

MADEIRA = {
    "MCN": ("Madeira — Costa Norte", "costa norte",
            (("Porto Moniz", 32.866, -17.168), ("São Vicente", 32.800, -17.043),
             ("Santana", 32.807, -16.883), ("Faial", 32.786, -16.848))),
    "MCS": ("Madeira — Costa Sul", "costa sul",
            (("Funchal", 32.650, -16.908), ("Câmara de Lobos", 32.650, -16.977),
             ("Ribeira Brava", 32.672, -17.064), ("Calheta", 32.720, -17.177),
             ("Machico", 32.718, -16.767), ("Santa Cruz", 32.688, -16.792))),
    "MRM": ("Madeira — Regiões Montanhosas", "montanha",
            (("Pico do Areeiro", 32.735, -16.928), ("Encumeada", 32.754, -17.019),
             ("Paul da Serra", 32.735, -17.059), ("Poiso", 32.713, -16.887))),
    "MPS": ("Porto Santo", "porto santo",
            (("Vila Baleira", 33.061, -16.340), ("Pico do Facho", 33.085, -16.320))),
}

AZORES = {
    "AOC": ("Açores — Grupo Ocidental", "ocidental",
            (("Santa Cruz das Flores", 39.453, -31.127), ("Lajes das Flores", 39.377, -31.174),
             ("Vila do Corvo", 39.672, -31.111))),
    "ACE": ("Açores — Grupo Central", "central",
            (("Horta", 38.534, -28.626), ("Madalena", 38.536, -28.528),
             ("Lajes do Pico", 38.396, -28.252), ("São Roque do Pico", 38.524, -28.316),
             ("Velas", 38.682, -28.208), ("Calheta (S. Jorge)", 38.601, -28.013),
             ("Santa Cruz da Graciosa", 39.086, -28.006), ("Angra do Heroísmo", 38.655, -27.220),
             ("Praia da Vitória", 38.733, -27.066))),
    "AOR": ("Açores — Grupo Oriental", "oriental",
            (("Ponta Delgada", 37.741, -25.668), ("Ribeira Grande", 37.822, -25.519),
             ("Povoação", 37.749, -25.247), ("Nordeste", 37.829, -25.148),
             ("Vila Franca do Campo", 37.716, -25.432), ("Vila do Porto", 36.944, -25.145))),
}

GROUP_NAME = {"continente": "Portugal continental", "madeira": "Madeira", "acores": "Açores"}
# Box (lon0, lon1, lat0, lat1) that holds every sample point: model fields are cropped to it.
BOX = (-32.0, -2.0, 31.5, 44.5)


@lru_cache(maxsize=1)
def _district_shapes() -> dict:
    import cartopy.io.shapereader as shp
    path = shp.natural_earth(resolution="10m", category="cultural", name="admin_1_states_provinces")
    out = {}
    for rec in shp.Reader(path).records():
        a = rec.attributes
        if a.get("adm0_a3") == "PRT":
            out[a["name"]] = rec.geometry
    return out


def _lattice_points(geom, step: float = 0.25) -> list[tuple[float, float]]:
    import shapely
    lon0, lat0, lon1, lat1 = geom.bounds
    lons = np.arange(np.floor(lon0 / step) * step, lon1 + step, step)
    lats = np.arange(np.floor(lat0 / step) * step, lat1 + step, step)
    LO, LA = np.meshgrid(lons, lats)
    inside = shapely.contains_xy(geom, LO.ravel(), LA.ravel())
    pts = [(float(la), float(lo)) for la, lo in zip(LA.ravel()[inside], LO.ravel()[inside])]
    if len(pts) < 3:                       # very small district: add its interior point
        p = geom.representative_point()
        pts.append((round(p.y, 3), round(p.x, 3)))
    return pts


@lru_cache(maxsize=1)
def areas() -> tuple[Area, ...]:
    shapes = _district_shapes()
    out = []
    for code, (name, ne_name, zone) in DISTRICTS.items():
        g = shapes[ne_name]
        lp = g.representative_point()
        out.append(Area(code, name, "continente", "portugal", zone, (), (lp.y, lp.x),
                        tuple(_lattice_points(g))))
    for code, (name, zone, towns) in MADEIRA.items():
        la = float(np.mean([t[1] for t in towns]))
        lo = float(np.mean([t[2] for t in towns]))
        out.append(Area(code, name, "madeira", "madeira", zone, towns, (la, lo),
                        tuple((t[1], t[2]) for t in towns)))
    for code, (name, zone, towns) in AZORES.items():
        la = float(np.mean([t[1] for t in towns]))
        lo = float(np.mean([t[2] for t in towns]))
        out.append(Area(code, name, "acores", "azores", zone, towns, (la, lo),
                        tuple((t[1], t[2]) for t in towns)))
    return tuple(out)


def names() -> dict[str, str]:
    """Area code -> name, without loading any geometry."""
    out = {c: v[0] for c, v in DISTRICTS.items()}
    out.update({c: v[0] for c, v in MADEIRA.items()})
    out.update({c: v[0] for c, v in AZORES.items()})
    return out


def by_code() -> dict[str, Area]:
    return {a.code: a for a in areas()}


def district_geometry(code: str):
    """Polygon of a mainland district (None for the islands)."""
    if code not in DISTRICTS:
        return None
    return _district_shapes()[DISTRICTS[code][1]]


# ------------------------------------------------------------------ sampling
class Sampler:
    """Bilinear weights of every area's points on one regular grid (lats/lons ascending),
    so an array (..., ny, nx) is reduced to (..., n_points) with one matrix product."""

    def __init__(self, lats: np.ndarray, lons: np.ndarray):
        self.lats, self.lons = np.asarray(lats, float), np.asarray(lons, float)
        self.slices: dict[str, slice] = {}
        idx, wts = [], []
        n = 0
        for a in areas():
            k0 = n
            for la, lo in a.points:
                if not (self.lats[0] <= la <= self.lats[-1] and self.lons[0] <= lo <= self.lons[-1]):
                    # outside this grid (e.g. ICON-EU and the Azores): keep the slot, as NaN,
                    # so every grid gives the points in the same order
                    idx.append([(0, 0)] * 4)
                    wts.append([np.nan] * 4)
                    n += 1
                    continue
                j = int(np.clip(np.searchsorted(self.lats, la) - 1, 0, len(self.lats) - 2))
                i = int(np.clip(np.searchsorted(self.lons, lo) - 1, 0, len(self.lons) - 2))
                ty = float(np.clip((la - self.lats[j]) / (self.lats[j + 1] - self.lats[j]), 0, 1))
                tx = float(np.clip((lo - self.lons[i]) / (self.lons[i + 1] - self.lons[i]), 0, 1))
                idx.append([(j, i), (j, i + 1), (j + 1, i), (j + 1, i + 1)])
                wts.append([(1 - ty) * (1 - tx), (1 - ty) * tx, ty * (1 - tx), ty * tx])
                n += 1
            if n > k0:
                self.slices[a.code] = slice(k0, n)
        nx = len(self.lons)
        self.flat = np.array([[j * nx + i for j, i in p] for p in idx], dtype=np.int64)
        self.w = np.array(wts, dtype=np.float32)

    def points(self, arr: np.ndarray) -> np.ndarray:
        """(..., ny, nx) -> (..., n_points). NaN where a corner is NaN."""
        a = np.asarray(arr, np.float32)
        flat = a.reshape(a.shape[:-2] + (-1,))
        return (flat[..., self.flat] * self.w).sum(-1)

    def reduce(self, arr: np.ndarray, how: str = "max") -> dict[str, np.ndarray]:
        """(..., ny, nx) -> {area code: (...)} with the max (or min) over the area's points."""
        p = self.points(arr)
        f = np.nanmax if how == "max" else np.nanmin
        out = {}
        for code, sl in self.slices.items():
            sub = p[..., sl]
            with np.errstate(all="ignore"):
                out[code] = f(sub, axis=-1) if sub.shape[-1] else np.full(sub.shape[:-1], np.nan)
        return out
