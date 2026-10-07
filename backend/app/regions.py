"""Map domains: projection, extent and the lon/lat box to download."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import cartopy.crs as ccrs
import numpy as np

from .models.base import Domain


@dataclass(frozen=True)
class Region:
    id: str
    name: str
    extent: tuple[float, float, float, float]   # lon0, lon1, lat0, lat1
    proj: str = "lcc"                            # lcc | merc | pc
    central_lon: float = 0.0
    central_lat: float = 45.0
    detail: str = "50m"                          # natural earth scale
    admin1: bool = False                         # draw districts/provinces
    barbs: int = 26                              # wind symbols across the width
    upsample: int = 1                            # smoothing factor for small regions
    grid: float = 10.0                           # graticule spacing (degrees)
    contour_scale: float = 1.0                   # <1 -> denser isobars/isohypses

    @cached_property
    def crs(self) -> ccrs.Projection:
        if self.proj == "lcc":
            return ccrs.LambertConformal(
                central_longitude=self.central_lon, central_latitude=self.central_lat,
                standard_parallels=(self.central_lat - 10, self.central_lat + 10))
        if self.proj == "merc":
            return ccrs.Mercator(central_longitude=self.central_lon)
        if self.proj == "stere":
            return ccrs.NorthPolarStereo(central_longitude=self.central_lon)
        return ccrs.PlateCarree()

    @cached_property
    def proj_extent(self) -> tuple[float, float, float, float]:
        """Projected bounding box (x0, x1, y0, y1) of the lon/lat extent."""
        lon0, lon1, lat0, lat1 = self.extent
        n = 200
        lons = np.concatenate([np.linspace(lon0, lon1, n), np.full(n, lon1),
                               np.linspace(lon1, lon0, n), np.full(n, lon0)])
        lats = np.concatenate([np.full(n, lat0), np.linspace(lat0, lat1, n),
                               np.full(n, lat1), np.linspace(lat1, lat0, n)])
        xyz = self.crs.transform_points(ccrs.PlateCarree(), lons, lats)
        if self.proj == "lcc":
            # Use the inner rectangle on the north side so the map has no blank corners
            x0, x1 = xyz[:, 0].min(), xyz[:, 0].max()
            y0, y1 = xyz[:, 1].min(), xyz[:, 1].max()
            return float(x0), float(x1), float(y0), float(y1)
        return (float(xyz[:, 0].min()), float(xyz[:, 0].max()),
                float(xyz[:, 1].min()), float(xyz[:, 1].max()))

    @cached_property
    def download_domain(self) -> Domain:
        """Lon/lat box that fully covers the projected rectangle (+margin)."""
        x0, x1, y0, y1 = self.proj_extent
        n = 120
        xs = np.concatenate([np.linspace(x0, x1, n), np.full(n, x1),
                             np.linspace(x1, x0, n), np.full(n, x0)])
        ys = np.concatenate([np.full(n, y0), np.linspace(y0, y1, n),
                             np.full(n, y1), np.linspace(y1, y0, n)])
        ll = ccrs.PlateCarree().transform_points(self.crs, xs, ys)
        m = 2.0
        return Domain(max(-180.0, float(ll[:, 0].min()) - m), min(179.75, float(ll[:, 0].max()) + m),
                      max(-90.0, float(ll[:, 1].min()) - m), min(90.0, float(ll[:, 1].max()) + m)).rounded()

    @property
    def aspect(self) -> float:
        x0, x1, y0, y1 = self.proj_extent
        return (y1 - y0) / (x1 - x0)

    def to_lonlat(self, fx: float, fy: float) -> tuple[float, float]:
        """Map-axes fraction (0..1 from left, 0..1 from top) -> lon, lat."""
        x0, x1, y0, y1 = self.proj_extent
        x = x0 + fx * (x1 - x0)
        y = y1 - fy * (y1 - y0)
        lon, lat = ccrs.PlateCarree().transform_point(x, y, self.crs)
        return float(lon), float(lat)

    def info(self) -> dict:
        return {"id": self.id, "name": self.name, "extent": self.extent}


REGIONS: dict[str, Region] = {r.id: r for r in [
    Region("europe", "Europa", (-25, 40, 33, 68), "lcc", 8, 50, "50m", barbs=28, grid=5),
    Region("natl", "Atlântico Norte", (-65, 25, 25, 68), "lcc", -20, 48, "110m", barbs=30),
    Region("iberia", "Península Ibérica", (-13, 5, 35, 45), "lcc", -4, 40, "50m", barbs=26,
           upsample=2, grid=2),
    Region("portugal", "Portugal Continental", (-12.0, -4.6, 36.6, 42.4), "merc", -8, 40,
           "10m", admin1=True, barbs=22, upsample=4, grid=1, contour_scale=0.5),
    Region("azores", "Açores", (-32, -24.5, 36.5, 40.3), "merc", -28, 38, "10m", barbs=22,
           upsample=4, grid=1, contour_scale=0.5),
    Region("madeira", "Madeira e Canárias", (-19.5, -12.5, 27.0, 33.8), "merc", -16, 30,
           "10m", barbs=20, upsample=3, grid=1, contour_scale=0.5),
    Region("med", "Mediterrâneo", (-8, 38, 29, 47), "lcc", 15, 38, "50m", barbs=28, grid=5),
]}


# Reference cities for value labels on maps
_PT = [("Viana do Castelo", 41.69, -8.83), ("Braga", 41.55, -8.42), ("Vila Real", 41.30, -7.74),
       ("Bragança", 41.81, -6.76), ("Porto", 41.15, -8.61), ("Aveiro", 40.64, -8.65),
       ("Viseu", 40.66, -7.91), ("Guarda", 40.54, -7.27), ("Coimbra", 40.21, -8.43),
       ("Castelo Branco", 39.82, -7.49), ("Leiria", 39.74, -8.81), ("Santarém", 39.24, -8.69),
       ("Portalegre", 39.29, -7.43), ("Lisboa", 38.72, -9.14), ("Setúbal", 38.52, -8.89),
       ("Évora", 38.57, -7.91), ("Beja", 38.02, -7.86), ("Sines", 37.96, -8.87),
       ("Faro", 37.02, -7.93), ("Lagos", 37.10, -8.67), ("Mértola", 37.64, -7.66)]
_ES = [("Madrid", 40.42, -3.70), ("Barcelona", 41.39, 2.17), ("Sevilha", 37.39, -5.98),
       ("Valência", 39.47, -0.38), ("Bilbau", 43.26, -2.93), ("Saragoça", 41.65, -0.88),
       ("Corunha", 43.36, -8.41), ("Valladolid", 41.65, -4.72), ("Badajoz", 38.88, -6.97),
       ("Córdova", 37.88, -4.78), ("Málaga", 36.72, -4.42), ("Múrcia", 37.99, -1.13),
       ("Salamanca", 40.97, -5.66), ("Oviedo", 43.36, -5.85), ("Granada", 37.18, -3.60),
       ("Palma", 39.57, 2.65)]
_EU = [("Lisboa", 38.72, -9.14), ("Madrid", 40.42, -3.70), ("Paris", 48.86, 2.35),
       ("Londres", 51.51, -0.13), ("Dublin", 53.35, -6.26), ("Berlim", 52.52, 13.40),
       ("Roma", 41.90, 12.50), ("Viena", 48.21, 16.37), ("Varsóvia", 52.23, 21.01),
       ("Estocolmo", 59.33, 18.07), ("Oslo", 59.91, 10.75), ("Helsínquia", 60.17, 24.94),
       ("Atenas", 37.98, 23.73), ("Bucareste", 44.43, 26.10), ("Kiev", 50.45, 30.52),
       ("Istambul", 41.01, 28.98), ("Argel", 36.75, 3.06), ("Rabat", 34.02, -6.84),
       ("Barcelona", 41.39, 2.17), ("Milão", 45.46, 9.19), ("Munique", 48.14, 11.58),
       ("Praga", 50.08, 14.44), ("Budapeste", 47.50, 19.04), ("Copenhaga", 55.68, 12.57),
       ("Amesterdão", 52.37, 4.90), ("Bordéus", 44.84, -0.58), ("Sevilha", 37.39, -5.98),
       ("Porto", 41.15, -8.61), ("Reiquiavique", 64.15, -21.94), ("Belgrado", 44.79, 20.45),
       ("Túnis", 36.81, 10.18), ("Moscovo", 55.76, 37.62), ("Riga", 56.95, 24.11)]
CITIES = {
    "portugal": _PT,
    "iberia": _PT[::3] + [("Lisboa", 38.72, -9.14), ("Porto", 41.15, -8.61), ("Faro", 37.02, -7.93)] + _ES,
    "europe": _EU, "natl": _EU[:20] + [("Ponta Delgada", 37.74, -25.67), ("Funchal", 32.65, -16.91)],
    "med": _EU + [("Nápoles", 40.85, 14.27), ("Palermo", 38.12, 13.36), ("Trípoli", 32.89, 13.19)],
    "azores": [("Ponta Delgada", 37.74, -25.67), ("Angra", 38.65, -27.22), ("Horta", 38.53, -28.63),
               ("Santa Cruz (Flores)", 39.45, -31.13), ("Vila do Porto", 36.94, -25.15),
               ("Velas", 38.68, -28.21), ("Madalena", 38.53, -28.53), ("S. Cruz Graciosa", 39.09, -28.01)],
    "madeira": [("Funchal", 32.65, -16.91), ("Porto Santo", 33.07, -16.35), ("S. Vicente", 32.80, -17.04),
                ("Las Palmas", 28.12, -15.43), ("S. C. Tenerife", 28.46, -16.25), ("Arrecife", 28.96, -13.55),
                ("Puerto del Rosario", 28.50, -13.86), ("La Palma", 28.68, -17.76)],
}
