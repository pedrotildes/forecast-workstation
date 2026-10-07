"""Minimal GRIB2 decoding with ecCodes.

We decode messages ourselves instead of going through cfgrib so that we keep
full control over accumulation windows (``stepRange``), layer levels and grid
orientation. Every decoded message becomes a :class:`Field` on a regular
lat/lon grid with latitudes ascending and longitudes ascending in [-180, 180).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import eccodes
import numpy as np


@dataclass
class Field:
    lats: np.ndarray            # 1-D, ascending
    lons: np.ndarray            # 1-D, ascending, -180..180
    values: np.ndarray          # 2-D (lat, lon)
    units: str = ""
    meta: dict = field(default_factory=dict)

    def crop(self, lon0: float, lon1: float, lat0: float, lat1: float) -> "Field":
        jm = (self.lats >= lat0) & (self.lats <= lat1)
        im = (self.lons >= lon0) & (self.lons <= lon1)
        return Field(self.lats[jm], self.lons[im], self.values[np.ix_(jm, im)],
                     self.units, dict(self.meta))

    def at(self, lat: float, lon: float) -> float:
        """Bilinear interpolation at a point."""
        la, lo, v = self.lats, self.lons, self.values
        j = int(np.clip(np.searchsorted(la, lat) - 1, 0, len(la) - 2))
        i = int(np.clip(np.searchsorted(lo, lon) - 1, 0, len(lo) - 2))
        ty = (lat - la[j]) / (la[j + 1] - la[j])
        tx = (lon - lo[i]) / (lo[i + 1] - lo[i])
        ty, tx = float(np.clip(ty, 0, 1)), float(np.clip(tx, 0, 1))
        return float((1 - ty) * ((1 - tx) * v[j, i] + tx * v[j, i + 1])
                     + ty * ((1 - tx) * v[j + 1, i] + tx * v[j + 1, i + 1]))


def _grid_axes(h) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    g = lambda k: eccodes.codes_get(h, k)  # noqa: E731
    ni, nj = g("Ni"), g("Nj")
    lat1 = g("latitudeOfFirstGridPointInDegrees")
    lon1 = g("longitudeOfFirstGridPointInDegrees")
    dlon = g("iDirectionIncrementInDegrees")
    dlat = g("jDirectionIncrementInDegrees")
    i_neg = bool(g("iScansNegatively"))
    j_pos = bool(g("jScansPositively"))
    lons = lon1 + np.arange(ni) * dlon * (-1 if i_neg else 1)
    lats = lat1 + np.arange(nj) * dlat * (1 if j_pos else -1)
    return lats, lons, i_neg, j_pos


def decode_message(buf: bytes, bbox: tuple[float, float, float, float] | None = None
                   ) -> tuple[dict, Field]:
    """Decode one message; ``bbox`` = (lon0, lon1, lat0, lat1) crops before reordering,
    which is much faster for global fields."""
    h = eccodes.codes_new_from_message(buf)
    try:
        if eccodes.codes_get(h, "gridType") != "regular_ll":
            raise ValueError("only regular_ll grids are supported")
        # string keys go through concept tables and are slow (~4 ms each): keep it minimal
        meta = {"units": eccodes.codes_get(h, "units")}
        lats, lons, _, _ = _grid_axes(h)
        vals = eccodes.codes_get_values(h).reshape(len(lats), len(lons))
        missing = eccodes.codes_get(h, "missingValue")
        has_bitmap = eccodes.codes_get(h, "bitmapPresent")
    finally:
        eccodes.codes_release(h)

    # Normalise orientation: lons -> [-180, 180) ascending, lats ascending.
    lons = ((lons + 180.0) % 360.0) - 180.0
    if bbox is not None:
        lon0, lon1, lat0, lat1 = bbox
        ic = np.nonzero((lons >= lon0) & (lons <= lon1))[0]
        jr = np.nonzero((lats >= lat0) & (lats <= lat1))[0]
        lons, lats = lons[ic], lats[jr]
        vals = vals[np.ix_(jr, ic)]
    order = np.argsort(lons, kind="stable")
    lons, vals = lons[order], vals[:, order]
    if len(lats) > 1 and lats[0] > lats[-1]:
        lats, vals = lats[::-1], vals[::-1, :]
    vals = np.ascontiguousarray(vals, dtype=np.float32)
    if has_bitmap:
        vals[vals == np.float32(missing)] = np.nan
    return meta, Field(lats.astype(float), lons.astype(float), vals, meta["units"], meta)


def split_messages(data: bytes) -> Iterator[tuple[dict, bytes]]:
    """Split a multi-message GRIB blob into (header, raw message) pairs."""
    pos = 0
    n = len(data)
    while pos < n:
        start = data.find(b"GRIB", pos)
        if start < 0:
            break
        edition = data[start + 7]
        if edition == 2:
            length = int.from_bytes(data[start + 8:start + 16], "big")
        else:
            length = int.from_bytes(data[start + 4:start + 7], "big")
        msg = data[start:start + length]
        h = eccodes.codes_new_from_message(msg)
        try:
            hdr = {k: eccodes.codes_get(h, k) for k in
                   ("shortName", "typeOfLevel", "level", "stepRange")}
            try:
                hdr["topLevel"] = eccodes.codes_get(h, "topLevel")
                hdr["bottomLevel"] = eccodes.codes_get(h, "bottomLevel")
            except eccodes.KeyValueNotFoundError:
                pass
        finally:
            eccodes.codes_release(h)
        yield hdr, msg
        pos = start + length
