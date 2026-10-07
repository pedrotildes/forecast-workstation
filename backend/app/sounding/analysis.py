"""Thermodynamic and kinematic sounding diagnostics (MetPy)."""
from __future__ import annotations

import warnings

import metpy.calc as mpcalc
import numpy as np
from metpy.units import units

warnings.filterwarnings("ignore")


def _m(x, u):
    try:
        v = x.to(u).magnitude
        v = float(np.asarray(v).ravel()[0]) if np.ndim(v) else float(v)
        return None if np.isnan(v) else v
    except Exception:  # noqa: BLE001
        return None


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return default


def _height_at_p(prof, p_target):
    if p_target is None:
        return None
    p, z = prof["p"], prof["z"]
    return float(np.interp(np.log(p_target), np.log(p[::-1]), z[::-1]) - z[0])


def _level_where(z_agl, x, value):
    """First height (AGL, m) where profile x crosses `value` going up."""
    for i in range(1, len(x)):
        if (x[i - 1] - value) * (x[i] - value) <= 0 and x[i - 1] != x[i]:
            f = (value - x[i - 1]) / (x[i] - x[i - 1])
            return float(z_agl[i - 1] + f * (z_agl[i] - z_agl[i - 1]))
    return None


def analyse(prof: dict) -> dict:
    p = prof["p"] * units.hPa
    T = prof["t"] * units.degC
    Td = prof["td"] * units.degC
    u = prof["u"] * units.knot
    v = prof["v"] * units.knot
    z = prof["z"] * units.m
    zagl = (prof["z"] - prof["z"][0]) * units.m
    out: dict = {"parcels": {}}

    # ---- parcels
    parcels = {}
    sb = _safe(lambda: mpcalc.parcel_profile(p, T[0], Td[0]).to("degC"))
    if sb is not None:
        parcels["SB"] = {"p0": float(p[0].m), "trace": sb}
    ml = _safe(lambda: mpcalc.mixed_parcel(p, T, Td, depth=100 * units.hPa))
    if ml is not None:
        mp, mt, mtd = ml
        trace = _safe(lambda: mpcalc.parcel_profile(p, mt, mtd).to("degC"))
        if trace is not None:
            parcels["ML"] = {"p0": float(p[0].m), "trace": trace, "t0": mt, "td0": mtd}
    mu = _safe(lambda: mpcalc.most_unstable_parcel(p, T, Td, depth=300 * units.hPa))
    if mu is not None:
        mup, mut, mutd, idx = mu
        sub = slice(int(idx), None)
        trace = _safe(lambda: mpcalc.parcel_profile(p[sub], mut, mutd).to("degC"))
        if trace is not None:
            parcels["MU"] = {"p0": float(mup.m), "trace": trace, "idx": int(idx)}

    for name, pc in parcels.items():
        sub = slice(pc.get("idx", 0), None)
        pp, tt, ttd, tr = p[sub], T[sub], Td[sub], pc["trace"]
        cape, cin = _safe(lambda: mpcalc.cape_cin(pp, tt, ttd, tr), (None, None))
        lcl_p, lcl_t = _safe(lambda: mpcalc.lcl(pp[0], tt[0] if name != "ML" else pc["t0"],
                                                 ttd[0] if name != "ML" else pc["td0"]), (None, None))
        lfc_p, _ = _safe(lambda: mpcalc.lfc(pp, tt, ttd, tr), (None, None))
        el_p, _ = _safe(lambda: mpcalc.el(pp, tt, ttd, tr), (None, None))
        li = _safe(lambda: mpcalc.lifted_index(pp, tt, tr))
        d = {
            "cape": _m(cape, "J/kg") if cape is not None else None,
            "cin": _m(cin, "J/kg") if cin is not None else None,
            "lcl_p": _m(lcl_p, "hPa") if lcl_p is not None else None,
            "lfc_p": _m(lfc_p, "hPa") if lfc_p is not None else None,
            "el_p": _m(el_p, "hPa") if el_p is not None else None,
            "li": _m(li, "delta_degC") if li is not None else None,
            "p0": pc["p0"],
        }
        for k in ("lcl", "lfc", "el"):
            d[f"{k}_z"] = _height_at_p(prof, d[f"{k}_p"])
        out["parcels"][name] = d

    # ---- classic indices
    idx = {}
    lp = np.log(prof["p"][::-1])

    def at(arr, pl):
        if pl > prof["p"][0] or pl < prof["p"][-1]:
            return None
        return float(np.interp(np.log(pl), lp, arr[::-1]))

    t850, t700, t500 = at(prof["t"], 850), at(prof["t"], 700), at(prof["t"], 500)
    d850, d700 = at(prof["td"], 850), at(prof["td"], 700)
    idx["k"] = (t850 - t500 + d850 - (t700 - d700)) if None not in (t850, t700, t500, d850, d700) else None
    idx["tt"] = (t850 + d850 - 2 * t500) if None not in (t850, d850, t500) else None
    idx["showalter"] = _m(_safe(lambda: mpcalc.showalter_index(p, T, Td)), "delta_degC")
    idx["pw"] = _m(_safe(lambda: mpcalc.precipitable_water(p, Td)), "mm")
    if "SB" in parcels:
        idx["sweat"] = _safe(lambda: float(np.ravel(mpcalc.sweat_index(p, T, Td, mpcalc.wind_speed(u, v),
                                                                       mpcalc.wind_direction(u, v)).m)[0]))
    dc = _safe(lambda: mpcalc.downdraft_cape(p, T, Td))
    idx["dcape"] = _m(dc[0], "J/kg") if dc else None

    def t_at(pl):
        if pl > p[0].m or pl < p[-1].m:
            return None, None
        lp = np.log(prof["p"][::-1])
        return (float(np.interp(np.log(pl), lp, prof["t"][::-1])),
                float(np.interp(np.log(pl), lp, prof["z"][::-1])))

    for a, b in ((850, 500), (700, 500)):
        ta, za = t_at(a)
        tb, zb = t_at(b)
        idx[f"lr_{a}_{b}"] = (ta - tb) / ((zb - za) / 1000) if ta is not None and tb is not None else None
    idx["lr_0_3"] = None
    z3 = prof["z"][0] + 3000
    if prof["z"][-1] > z3:
        t3 = float(np.interp(z3, prof["z"], prof["t"]))
        idx["lr_0_3"] = (prof["t"][0] - t3) / 3.0
    idx["frz"] = _level_where(zagl.m, prof["t"], 0.0)
    wb = _safe(lambda: mpcalc.wet_bulb_temperature(p, T, Td).to("degC").m)
    idx["wbz"] = _level_where(zagl.m, wb, 0.0) if wb is not None else None
    if idx["frz"] is not None:
        idx["frz_msl"] = idx["frz"] + prof["z"][0]
    if idx.get("wbz") is not None:
        idx["wbz_msl"] = idx["wbz"] + prof["z"][0]
    out["wetbulb"] = wb.tolist() if wb is not None else None

    # ---- kinematics
    kin = {}
    for d in (1, 3, 6):
        sh = _safe(lambda: mpcalc.bulk_shear(p, u, v, height=zagl, depth=d * 1000 * units.m))
        kin[f"shear_0_{d}"] = _m(mpcalc.wind_speed(*sh), "knot") if sh else None
    bk = _safe(lambda: mpcalc.bunkers_storm_motion(p, u, v, zagl))
    if bk:
        rm, lm, mean = bk
        kin["rm"] = [_m(rm[0], "knot"), _m(rm[1], "knot")]
        kin["lm"] = [_m(lm[0], "knot"), _m(lm[1], "knot")]
        kin["mean"] = [_m(mean[0], "knot"), _m(mean[1], "knot")]
        for d in (1, 3):
            srh = _safe(lambda: mpcalc.storm_relative_helicity(
                zagl, u, v, depth=d * 1000 * units.m, storm_u=rm[0], storm_v=rm[1]))
            kin[f"srh_0_{d}"] = _m(srh[2], "m^2/s^2") if srh else None
    out["kinematics"] = kin

    # ---- composites
    sbp, mlp, mup = (out["parcels"].get(k, {}) for k in ("SB", "ML", "MU"))
    try:
        mucape = mup.get("cape") or 0
        srh3 = kin.get("srh_0_3") or 0
        sh6 = (kin.get("shear_0_6") or 0) * 0.514444
        idx["scp"] = (mucape / 1000) * (srh3 / 50) * (min(sh6, 20) / 20 if sh6 >= 10 else 0)
        lcl = mlp.get("lcl_z") or sbp.get("lcl_z") or 9999
        lcl_t = 1.0 if lcl < 1000 else (0 if lcl > 2000 else (2000 - lcl) / 1000)
        sh_t = 0 if sh6 < 12.5 else min(sh6, 30) / 20
        cape_ml = mlp.get("cape") or 0
        cin_ml = mlp.get("cin") or 0
        cin_t = 1.0 if cin_ml > -50 else (0 if cin_ml < -200 else (200 + cin_ml) / 150)
        idx["stp"] = (cape_ml / 1500) * lcl_t * ((kin.get("srh_0_1") or 0) / 150) * sh_t * cin_t
    except Exception:  # noqa: BLE001
        pass
    out["indices"] = idx
    out["_parcel_traces"] = {k: v["trace"].m for k, v in parcels.items()}
    out["_mu_idx"] = parcels.get("MU", {}).get("idx", 0)
    return out
