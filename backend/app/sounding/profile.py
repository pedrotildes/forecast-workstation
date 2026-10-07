"""Extract a model vertical profile (forecast sounding) at a point."""
from __future__ import annotations

import datetime as dt

import numpy as np

from ..models import Domain, FieldUnavailable, Model
from ..products import dewpoint_from_rh, rh_from_q

KT = 1.943844


async def get_profile(model: Model, run: dt.datetime, step: int, lat: float, lon: float) -> dict:
    lv = model.levels
    hum = "r" if model.has("r") else "q"
    upper = [(n, p) for p in lv for n in ("gh", "t", "u", "v", hum) + (("w",) if model.has("w") else ())]
    single = [(n, 0) for n in ("sp", "t2m", "d2m", "u10", "v10", "orog", "msl",
                               "cape", "mlcape", "mucape", "pwat") if model.has(n)]
    dom = Domain.tile(lat, lon)
    f = await model.fetch(run, step, upper + single, dom)
    if not f:
        raise FieldUnavailable(f"{model.name}: sem dados para a sondagem")

    def val(k):
        return f[k].at(lat, lon) if k in f else np.nan

    P, Z, T, TD, U, V, W = [], [], [], [], [], [], []
    for p in sorted(lv, reverse=True):
        t = val(("t", p))
        z = val(("gh", p))
        if np.isnan(t) or np.isnan(z):
            continue
        if hum == "r":
            rh = val(("r", p))
        else:
            rh = float(rh_from_q(t, val(("q", p)), p))
        td = float(dewpoint_from_rh(t, max(rh, 1.0)))
        P.append(p), Z.append(z), T.append(t - 273.15), TD.append(min(td, t - 273.15))
        U.append(val(("u", p)) * KT), V.append(val(("v", p)) * KT), W.append(val(("w", p)))

    P, Z, T, TD, U, V, W = map(np.array, (P, Z, T, TD, U, V, W))

    psfc = val(("sp", 0)) / 100.0
    if np.isnan(psfc):
        psfc = val(("msl", 0)) / 100.0
    zsfc = val(("orog", 0))
    if np.isnan(zsfc):  # interpolate geopotential height to the surface pressure in ln p
        lnp = np.log(P)
        zsfc = float(np.interp(np.log(psfc), lnp[::-1], Z[::-1]))
        if psfc > P[0]:  # below the lowest level: hypsometric extrapolation
            tm = T[0] + 273.15
            zsfc = Z[0] - 287.05 * tm / 9.80665 * np.log(psfc / P[0])
    zsfc = max(float(zsfc), 0.0) if psfc > 1005 else float(zsfc)

    above = P < psfc - 2.0
    sfc = {"p": float(psfc), "z": float(zsfc), "t": val(("t2m", 0)) - 273.15,
           "td": val(("d2m", 0)) - 273.15, "u": val(("u10", 0)) * KT, "v": val(("v10", 0)) * KT}
    if np.isnan(sfc["t"]):  # fall back to the lowest level above ground
        i = int(np.argmax(above))
        sfc.update(t=T[i], td=TD[i], u=U[i], v=V[i])
    sfc["td"] = min(sfc["td"], sfc["t"])

    prof = {
        "p": np.concatenate([[sfc["p"]], P[above]]),
        "z": np.concatenate([[sfc["z"]], Z[above]]),
        "t": np.concatenate([[sfc["t"]], T[above]]),
        "td": np.concatenate([[sfc["td"]], TD[above]]),
        "u": np.concatenate([[sfc["u"]], U[above]]),
        "v": np.concatenate([[sfc["v"]], V[above]]),
        "w": np.concatenate([[np.nan], W[above]]),
    }
    model_diag = {k: val((k, 0)) for k in ("cape", "mlcape", "mucape", "pwat") if model.has(k)}
    return {"lat": lat, "lon": lon, "profile": prof, "surface": sfc, "model_diag": model_diag}
