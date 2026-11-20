"""Figures of an alert event (sent to Telegram and shown in the app's «Avisos» tab).

1. probability map — ENS probability of reaching the orange criterion (peak of the
   event days), confirmed areas outlined, the probability of each area written on it;
2. «valor alto provável» — 90th percentile of the 50 members (10th for cold);
3. members per day — box plots of the area value per day, thresholds and the
   deterministic models;
4. plume (wind and rain) — the 50 members every 3 h in the area with the highest chance;
5. trend — the probability in the successive ENS runs (how certainty built up);
6. deterministic models side by side — IFS, AIFS, GFS, ICON-EU on the peak day;
7. synoptic context — IFS pressure, precipitation and wind (or 850 hPa temperature).
"""
from __future__ import annotations

import datetime as dt
import io
import json
from functools import lru_cache

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..config import ALERTS_DIR
from ..maps.render import render_job
from ..maps.styles import Fill, _interp
from ..models import MODELS, FieldUnavailable
from ..regions import CITIES, REGIONS
from . import criteria as C
from .areas import areas, by_code, district_geometry
from .detect import DOMAIN, RUNS_DIR, by_area, levels, sampler_for
from .messages import MODEL_SHORT, fmt_day, fmt_days, fmt_run, pct, unit

FIG_DIR = ALERTS_DIR / "figs"
H = dt.timedelta(hours=1)
REGION_OF = {"continente": "portugal", "madeira": "madeira", "acores": "azores"}
LEVEL_COLOR = {"vermelho": "#d00000", "laranja": "#f77f00", "vigilancia": "#555555", None: "#555555"}
MODEL_COLOR = {"ecmwf": "#1d3557", "aifs": "#6a4c93", "gfs": "#2a9d8f", "icon_eu": "#e76f51"}
PROB_FILL = Fill([10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                 _interp(["#fff3b0", "#ffd166", "#f8961e", "#f3722c", "#e63946", "#9d0208",
                          "#6a040f", "#3c096c"], 9), extend="neither", transparent_under=True)
VAR_OF = {"vento": "gust", "chuva": "precip", "calor": "tmax_day", "frio": "tmin_day"}
HOW = {"vento": "max", "chuva": "max", "calor": "max", "frio": "min"}
FOOT = "Dados: © ECMWF Open Data (CC BY 4.0), NOAA GFS, DWD ICON-EU · Critérios de aviso: IPMA"


# ---------------------------------------------------------------- data
def load_npz(run_iso: str, phen: str) -> dict | None:
    p = RUNS_DIR / dt.datetime.fromisoformat(run_iso).strftime("%Y%m%d%H") / f"{phen}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return {k: z[k] for k in z.files}


def load_det(run_iso: str, phen: str) -> dict:
    p = RUNS_DIR / dt.datetime.fromisoformat(run_iso).strftime("%Y%m%d%H") / f"{phen}_det.json"
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def area_slices() -> dict[str, slice]:
    out, k = {}, 0
    for a in areas():
        out[a.code] = slice(k, k + len(a.points))
        k += len(a.points)
    return out


@lru_cache(maxsize=8)
def _area_grid(sig) -> np.ndarray:
    """Area code of every grid node (districts by polygon; islands: nearest town ≤ 0.35°)."""
    import shapely
    lat0, nlat, dlat, lon0, nlon, dlon = sig
    lats, lons = lat0 + np.arange(nlat) * dlat, lon0 + np.arange(nlon) * dlon
    LO, LA = np.meshgrid(lons, lats)
    out = np.full(LA.shape, "", dtype=object)
    for a in areas():
        g = district_geometry(a.code)
        if g is not None:
            out[shapely.contains_xy(g, LO, LA)] = a.code
    towns = [(a.code, la, lo) for a in areas() if a.group != "continente" for _, la, lo in a.towns]
    tl = np.array([[la, lo] for _, la, lo in towns])
    d = np.hypot(LA[..., None] - tl[:, 0], (LO[..., None] - tl[:, 1]) * np.cos(np.radians(LA[..., None])))
    near = d.argmin(-1)
    ok = d.min(-1) <= 0.35
    codes = np.array([c for c, _, _ in towns], dtype=object)
    out[ok & (out == "")] = codes[near[ok & (out == "")]]
    return out


def area_grid(lats, lons) -> np.ndarray:
    sig = (round(float(lats[0]), 4), len(lats), round(float(lats[1] - lats[0]), 5),
           round(float(lons[0]), 4), len(lons), round(float(lons[1] - lons[0]), 5))
    return _area_grid(sig)


def threshold_grid(phen: str, lats, lons, region_id: str) -> np.ndarray:
    """(2, ny, nx) orange/red thresholds for the probability map (NaN where not defined)."""
    ny, nx = len(lats), len(lons)
    if phen in ("vento", "chuva"):
        t = (C.GUST_AZORES if region_id == "azores" else C.GUST) if phen == "vento" else C.RAIN6
        return np.stack([np.full((ny, nx), t["laranja"]), np.full((ny, nx), t["vermelho"])]).astype(np.float32)
    ag = area_grid(lats, lons)
    out = np.full((2, ny, nx), np.nan, np.float32)
    ac = by_code()
    for code in set(ag.ravel()) - {""}:
        t = C.thresholds(phen, code, ac[code].group)
        m = ag == code
        out[0][m], out[1][m] = t["laranja"], t["vermelho"]
    return out


def _png_size(png: bytes) -> tuple[int, int]:
    return int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")


def _fig_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=fig.dpi, facecolor="white")
    return buf.getvalue()


# ---------------------------------------------------------------- figures
def _labels(ev: dict, region_id: str) -> tuple[list, list]:
    ac = by_code()
    pts, geoms = [], []
    for code, a in ev["areas"].items():
        ar = ac[code]
        if REGION_OF[ar.group] != region_id:
            continue
        col = LEVEL_COLOR[a["status"]]
        name = ar.name.split(" — ")[-1]
        pts.append((ar.label[1], ar.label[0], f"{name}\n{pct(a['p_o'])}", col))
        if a["status"] in ("laranja", "vermelho") and (g := district_geometry(code)) is not None:
            geoms.append((g, col))
    return pts, geoms


def prob_map(ev: dict, z: dict, region_id: str) -> bytes:
    phen = ev["phen"]
    lats, lons = z["lats"], z["lons"]
    thr = threshold_grid(phen, lats, lons, region_id)
    days = [d for d in ev["days"] if f"grid_{d}" in z]
    probs = []
    for d in days:
        g = z[f"grid_{d}"].astype(np.float32)                       # (50, ny, nx)
        lev = (g >= thr[0]) if phen != "frio" else (g <= thr[0])
        p = lev.mean(0) * 100
        p[np.isnan(thr[0])] = np.nan
        probs.append(p)
    p = np.minimum(np.nanmax(np.stack(probs), 0), 99.9).astype(np.float32)   # 100 % in the top class
    pts, geoms = _labels(ev, region_id)
    r = REGIONS[region_id]
    what = {"vento": f"rajada ≥ {C.fmt_threshold('vento', (C.GUST_AZORES if region_id == 'azores' else C.GUST)['laranja'])}",
            "chuva": "≥ 41 mm em 6 h", "calor": "Tmáx ≥ limiar laranja do distrito (≥ 48 h no texto)",
            "frio": "Tmín ≤ limiar laranja do distrito"}[phen]
    job = {"region": region_id, "lats": lats, "lons": lons, "layers": [
        {"type": "fill", "var": "t2m", "data": p, "fill_style": PROB_FILL,
         "label": f"Probabilidade (%) de {what} — máximo nos dias do evento"},
        {"type": "contour", "var": "t2m", "data": np.nan_to_num(p), "levels": [50, 80],
         "color": "#222222", "width": 0.9, "labels": False, "extrema": False},
        {"type": "areas", "geoms": geoms},
        {"type": "labels", "points": pts, "size": 8.0}],
        "header": {"left1": f"ECMWF ENS (50 membros)   ·   Run {fmt_run(ev['last_run'])}",
                   "left2": f"Probabilidade de chegar ao aviso laranja · {fmt_days(ev['days'], True)}",
                   "layers": "Sombreado: em cada ponto · linhas finas: 50 % e 80 % · número: na área "
                             "(algum ponto chega ao laranja) · contorno: distrito confirmado",
                   "right1": C.PHENOMENA[phen]["name"], "right2": r.name},
        "footer": FOOT}
    png, _ = render_job(job)
    return png


def high_map(ev: dict, z: dict, region_id: str) -> bytes:
    phen = ev["phen"]
    peak = _peak_day(ev)
    if f"grid_{peak}" not in z:
        peak = [d for d in ev["days"] if f"grid_{d}" in z][0]
    g = z[f"grid_{peak}"].astype(np.float32)
    q = 10 if phen == "frio" else 90
    arr = np.nanpercentile(g, q, axis=0).astype(np.float32)
    var = VAR_OF[phen]
    pts, geoms = _labels(ev, region_id)
    what = {"vento": "rajada máxima do dia (km/h)", "chuva": "maior precipitação em 6 h no dia (mm)",
            "calor": "temperatura máxima (°C)", "frio": "temperatura mínima (°C)"}[phen]
    job = {"region": region_id, "lats": z["lats"], "lons": z["lons"], "layers": [
        {"type": "fill", "var": var, "data": arr, "label": f"Percentil {q} dos 50 membros: {what}"},
        {"type": "values", "var": var, "data": arr, "fmt": "%.0f"},
        {"type": "areas", "geoms": geoms}],
        "header": {"left1": f"ECMWF ENS (50 membros)   ·   Run {fmt_run(ev['last_run'])}",
                   "left2": f"Valor alto provável: {'só 1 em cada 10 membros abaixo' if phen == 'frio' else 'só 1 em cada 10 membros acima'} · "
                            f"{fmt_day(dt.date.fromisoformat(peak), True)}",
                   "layers": f"Percentil {q} da {what} em cada ponto · valores nas cidades",
                   "right1": C.PHENOMENA[phen]["name"], "right2": REGIONS[region_id].name},
        "footer": FOOT}
    png, _ = render_job(job)
    return png


def _peak_day(ev: dict) -> str:
    conf = [a for a in ev["areas"].values() if a["status"] in ("laranja", "vermelho")] or list(ev["areas"].values())
    best = max(conf, key=lambda a: (a["p_o"], a["p_r"]))
    return best.get("peak_day") or ev["days"][0]


def _top_areas(ev: dict, n: int = 4) -> list[str]:
    return list(ev["areas"])[:n]


def boxes(ev: dict, z: dict, det: dict) -> bytes:
    phen = ev["phen"]
    sl = area_slices()
    days = [str(d) for d in z["days"]]
    codes = _top_areas(ev, 4)
    n = len(codes)
    fig = Figure(figsize=(11.5, 2.9 * n + 0.9), dpi=100, facecolor="white")
    FigureCanvasAgg(fig)
    u = unit(phen)
    ac = by_code()
    for i, code in enumerate(codes):
        ax = fig.add_subplot(n, 1, i + 1)
        data = []
        for d in days:
            v = z[f"pts_{d}"].astype(np.float32)[:, sl[code]]
            with np.errstate(all="ignore"):
                data.append(np.nanmax(v, 1) if HOW[phen] == "max" else np.nanmin(v, 1))
        x = np.arange(len(days))
        bp = ax.boxplot(data, positions=x, widths=0.5, whis=(10, 90), showfliers=True, patch_artist=True,
                        medianprops={"color": "#111", "lw": 1.6},
                        flierprops={"marker": ".", "ms": 3, "mfc": "#888", "mec": "none"})
        for b in bp["boxes"]:
            b.set(facecolor="#cfe1f2", edgecolor="#2b4c7e")
        a = ev["areas"][code]
        ax.axhline(a["thr_o"], color="#f77f00", lw=1.6, ls="--")
        ax.axhline(a["thr_r"], color="#d00000", lw=1.6, ls="--")
        for j, (mid, md) in enumerate(sorted(det.items())):
            for k, d in enumerate(days):
                vals = (md.get("days") or {}).get(d)
                if vals is None:
                    continue
                vv = np.array([np.nan if x_ is None else x_ for x_ in vals[sl[code]]], float)
                if not np.isfinite(vv).any():
                    continue
                y = np.nanmax(vv) if HOW[phen] == "max" else np.nanmin(vv)
                ax.plot(k + 0.32 + 0.07 * j, y, marker="D", ms=6, color=MODEL_COLOR.get(mid, "#333"),
                        mec="white", mew=0.6, ls="none", label=MODEL_SHORT.get(mid, mid) if k == 0 or i == 0 else None)
        ev_days = set(ev["days"])
        for k, d in enumerate(days):
            if d in ev_days:
                ax.axvspan(k - 0.45, k + 0.45, color="#fde2c8", alpha=0.45, zorder=0)
        ax.set_xticks(x, [fmt_day(dt.date.fromisoformat(d)) for d in days], fontsize=9)
        ax.set_ylabel(u)
        ax.grid(alpha=0.3, axis="y")
        ax.set_title(f"{ac[code].name} — {pct(a['p_o'])} laranja"
                     + (f", {pct(a['p_r'])} vermelho" if a["p_r"] >= 0.05 else ""), loc="left",
                     fontsize=10.5, fontweight="bold")
        h, lab = ax.get_legend_handles_labels()
        uniq = dict(zip(lab, h))
        if uniq and i == 0:
            ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="upper right", ncol=4, framealpha=0.9)
    fig.suptitle(f"{C.PHENOMENA[phen]['name']}: os 50 membros por dia ({unit(phen)}) · run {fmt_run(ev['last_run'])}",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold", color="#0d1b2a")
    fig.text(0.01, 0.005, "Caixas: 25–75 % dos membros, bigodes 10–90 %, pontos: membros extremos · "
             "tracejado laranja/vermelho: limiares IPMA · losangos: modelos determinísticos · "
             "fundo: dias do evento", fontsize=7.8, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    return _fig_png(fig)


def plume(ev: dict, z: dict) -> bytes | None:
    phen = ev["phen"]
    if phen not in ("vento", "chuva"):
        return None
    sl = area_slices()
    code = _top_areas(ev, 1)[0]
    run = dt.datetime.fromisoformat(ev["last_run"])
    times, rows = [], []
    for d in [str(x) for x in z["days"]]:
        if f"series_{d}" not in z:
            continue
        s = z[f"series_{d}"].astype(np.float32)[:, :, sl[code]]      # (steps, 50, n)
        tt = [run + int(st) * H for st in z[f"steps_{d}"]]
        if times and tt and tt[0] - times[-1] > 6 * H:                  # day not analysed: break the lines
            rows.append(np.full((1, s.shape[1]), np.nan, np.float32))
            times.append(times[-1] + 3 * H)
        with np.errstate(all="ignore"):
            rows.append(np.nanmax(s, -1))
        times += tt
    if not rows:
        return None
    v = np.concatenate(rows, 0)                                        # (T, 50)
    fig = Figure(figsize=(11.5, 4.6), dpi=100, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    for m in range(v.shape[1]):
        ax.plot(times, v[:, m], color="#7a8ca5", lw=0.6, alpha=0.55)
    q10, q50, q90 = np.nanpercentile(v, [10, 50, 90], axis=1)
    ax.fill_between(times, q10, q90, color="#457b9d", alpha=0.18, label="10–90 % dos membros")
    ax.plot(times, q50, color="#1d3557", lw=2.4, label="mediana")
    a = ev["areas"][code]
    ax.axhline(a["thr_o"], color="#f77f00", lw=1.6, ls="--", label="limiar laranja")
    ax.axhline(a["thr_r"], color="#d00000", lw=1.6, ls="--", label="limiar vermelho")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %Hh"))
    ax.grid(alpha=0.3)
    ax.set_ylabel(f"{unit(phen)}" + (" em 6 h" if phen == "chuva" else ""))
    ax.legend(fontsize=8.5, loc="upper left")
    what = "rajada máxima (3 h)" if phen == "vento" else "precipitação nas 6 h anteriores"
    ax.set_title(f"{by_code()[code].name}: {what}, os 50 membros · run {fmt_run(ev['last_run'])}",
                 loc="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    return _fig_png(fig)


def trend(ev: dict) -> bytes | None:
    """Probability in the successive ENS runs (from the saved signals)."""
    runs = sorted(p.name for p in RUNS_DIR.iterdir() if (p / "signals.json").exists()) if RUNS_DIR.exists() else []
    last = dt.datetime.fromisoformat(ev["last_run"]).strftime("%Y%m%d%H")
    runs = [r for r in runs if r <= last][-8:]
    if len(runs) < 2:
        return None
    codes = [c for c, a in ev["areas"].items() if a["status"] in ("laranja", "vermelho")][:5] or _top_areas(ev, 3)
    series = {c: ([], []) for c in codes}
    xs = []
    for r in runs:
        s = json.loads((RUNS_DIR / r / "signals.json").read_text())
        by_day = s.get("signals", {}).get(ev["phen"], {})
        xs.append(dt.datetime.strptime(r, "%Y%m%d%H"))
        for c in codes:
            po = max((by_day.get(d, {}).get(c, {}).get("p_o", 0.0) for d in ev["days"]), default=0.0)
            pr = max((by_day.get(d, {}).get(c, {}).get("p_r", 0.0) for d in ev["days"]), default=0.0)
            series[c][0].append(100 * po)
            series[c][1].append(100 * pr)
    fig = Figure(figsize=(11.5, 4.2), dpi=100, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    cols = ["#d62828", "#f77f00", "#1d3557", "#2a9d8f", "#6a4c93"]
    for i, c in enumerate(codes):
        ax.plot(xs, series[c][0], "-o", color=cols[i % 5], lw=2, ms=5, label=f"{by_code()[c].name} (laranja)")
        if max(series[c][1]) >= 5:
            ax.plot(xs, series[c][1], "--", color=cols[i % 5], lw=1.3, label=f"{by_code()[c].name} (vermelho)")
    ax.axhline(60, color="#888", lw=1, ls=":")
    ax.text(xs[0], 61, " limiar de confirmação (60 %)", fontsize=8, color="#666", va="bottom")
    ax.set_ylim(0, 102)
    ax.set_ylabel("probabilidade (%)")
    ax.set_xticks(xs, [f"{x.day} {x:%H}h" for x in xs], fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.set_title(f"Como a certeza evoluiu: runs do ensemble do ECMWF para {fmt_days(ev['days'])}",
                 loc="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    return _fig_png(fig)


async def _det_grid(mid: str, phen: str, day: str):
    """Daily grid of a deterministic model on the event day (cached GRIBs from the detector)."""
    from .detect import _det_daily  # noqa: F401 (same steps/logic)
    m = MODELS[mid]
    run = await m.best_run()
    dday = dt.date.fromisoformat(day)
    if phen in ("calor", "frio"):
        from ..daily import daily_fields
        kind = "tmax" if phen == "calor" else "tmin"
        d = await daily_fields(m, run, DOMAIN, kinds=(kind,))
        if dday not in d[kind]:
            return None
        return d["grid"].lats, d["grid"].lons, d[kind][dday], run
    key = ("gust", 0) if phen == "vento" else ("tp", 0)
    if not m.has(key[0]):
        return None
    steps = await m.list_steps(run)
    t0 = dt.datetime(dday.year, dday.month, dday.day)
    lo, hi = int((t0 - run) / H), int((t0 + 24 * H - run) / H)
    if lo < 0 or hi > steps[-1]:
        return None
    use = [s for s in steps if (lo - 6 if phen == "chuva" else lo) <= s <= hi and s % 3 == 0]
    got = {}
    for s in use:
        try:
            got[s] = (await m.fetch(run, s, [key], DOMAIN))[key]
        except FieldUnavailable:
            pass
    if not got:
        return None
    f0 = next(iter(got.values()))
    if phen == "vento":
        arr = np.nanmax(np.stack([got[s].values for s in got if s > lo]), 0) * 3.6
    else:
        acc = [np.maximum(got[s].values - got[s - 6].values, 0) for s in got if s > lo and s - 6 in got]
        if not acc:
            return None
        arr = np.nanmax(np.stack(acc), 0)
    return f0.lats, f0.lons, arr.astype(np.float32), run


async def models_panel(ev: dict, region_id: str) -> bytes | None:
    import asyncio

    from matplotlib.image import imread
    phen = ev["phen"]
    day = _peak_day(ev)
    var = VAR_OF[phen]
    pts, geoms = _labels(ev, region_id)
    pngs = []
    r = REGIONS[region_id]
    for mid in ("ecmwf", "aifs", "gfs", "icon_eu"):
        m = MODELS[mid]
        if not m.covers_box(*r.extent) or (phen == "vento" and not m.has("gust")):
            continue
        try:
            g = await _det_grid(mid, phen, day)
        except Exception:  # noqa: BLE001
            g = None
        if g is None:
            continue
        lats, lons, arr, run = g
        job = {"region": region_id, "lats": lats, "lons": lons, "layers": [
            {"type": "fill", "var": var, "data": arr, "label": f"{unit(phen)}"},
            {"type": "values", "var": var, "data": arr, "fmt": "%.0f"},
            {"type": "areas", "geoms": geoms}],
            "header": {"left1": f"{m.name} · run {fmt_run(run.isoformat())}",
                       "left2": fmt_day(dt.date.fromisoformat(day), True),
                       "layers": "", "right1": "", "right2": ""}, "footer": ""}
        png, _ = await asyncio.to_thread(render_job, job)
        pngs.append(png)
    if len(pngs) < 2:
        return None
    ncol = 2
    nrow = (len(pngs) + 1) // 2
    w, h = _png_size(pngs[0])
    fig = Figure(figsize=(ncol * w / 100 * 0.62, nrow * h / 100 * 0.62 + 0.5), dpi=100, facecolor="white")
    FigureCanvasAgg(fig)
    for i, png in enumerate(pngs):
        ax = fig.add_subplot(nrow, ncol, i + 1)
        ax.imshow(imread(io.BytesIO(png), format="png"))
        ax.axis("off")
    what = {"vento": "rajada máxima do dia", "chuva": "maior precipitação em 6 h",
            "calor": "temperatura máxima", "frio": "temperatura mínima"}[phen]
    fig.suptitle(f"Modelos determinísticos: {what} ({unit(phen)})", x=0.01, ha="left", fontsize=13,
                 fontweight="bold", color="#0d1b2a")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.95, bottom=0.005, wspace=0.01, hspace=0.01)
    return _fig_png(fig)


async def synoptic(ev: dict, z: dict) -> bytes | None:
    from ..maps.compose import build_job, preset_layers
    phen = ev["phen"]
    m = MODELS["ecmwf"]
    run = await m.best_run()
    steps = await m.list_steps(run)
    day = dt.date.fromisoformat(_peak_day(ev))
    if phen in ("vento", "chuva"):
        # time of the highest median in the most likely area
        sl = area_slices()
        code = _top_areas(ev, 1)[0]
        t_peak = dt.datetime(day.year, day.month, day.day, 12)
        k = f"series_{day.isoformat()}"
        if k in z:
            s = z[k].astype(np.float32)[:, :, sl[code]]
            med = np.nanmedian(np.nanmax(s, -1), 1)
            enr = dt.datetime.fromisoformat(ev["last_run"])
            t_peak = enr + int(z[f"steps_{day.isoformat()}"][int(np.nanargmax(med))]) * H
        preset, region = "sfc_precip", "nea"
    else:
        t_peak = dt.datetime(day.year, day.month, day.day, 12)
        preset, region = "t850", "iberia"
    step = int(round((t_peak - run) / H / 3) * 3)
    if step not in steps:
        cands = [s for s in steps if s > 0]
        if not cands:
            return None
        step = min(cands, key=lambda s: abs(s - step))
    try:
        job = await build_job(m, run, step, steps, region, preset_layers(preset, m))
    except FieldUnavailable:
        return None
    png, _ = render_job(job)
    return png


async def render_all(ev: dict, log=print) -> list[tuple[str, bytes, str]]:
    """All figures of an event for its last run: [(name, png, caption)], also saved on disk."""
    import asyncio
    z = load_npz(ev["last_run"], ev["phen"])
    if z is None:
        return []
    det = load_det(ev["last_run"], ev["phen"])
    groups = []
    for code, a in ev["areas"].items():
        g = by_code()[code].group
        if REGION_OF[g] not in groups:
            groups.append(REGION_OF[g])
    out: list[tuple[str, bytes, str]] = []

    async def add(name, coro_or_fn, caption):
        try:
            png = await coro_or_fn() if asyncio.iscoroutinefunction(coro_or_fn) else \
                await asyncio.to_thread(coro_or_fn)
        except Exception as e:  # noqa: BLE001 — a figure that fails must not stop the alert
            log(f"  figura {name}: {e}")
            return
        if png:
            out.append((name, png, caption))

    for reg in groups[:2]:
        await add(f"prob_{reg}", lambda reg=reg: prob_map(ev, z, reg),
                  f"Probabilidade de chegar ao aviso laranja ({REGIONS[reg].name})")
    await add("valor_alto", lambda: high_map(ev, z, groups[0]), "Valor alto provável (percentil 90 dos membros)")
    await add("membros_dia", lambda: boxes(ev, z, det), "Os 50 membros, dia a dia, e os modelos determinísticos")
    await add("pluma", lambda: plume(ev, z), "Evolução hora a hora na área mais exposta")
    await add("tendencia", lambda: trend(ev), "Como a certeza evoluiu de run para run")

    async def _models():
        return await models_panel(ev, groups[0])

    async def _syn():
        return await synoptic(ev, z)

    await add("modelos", _models, "Modelos determinísticos no dia de maior risco")
    await add("sinoptico", _syn, "Situação sinóptica (ECMWF IFS)")
    d = FIG_DIR / ev["id"] / dt.datetime.fromisoformat(ev["last_run"]).strftime("%Y%m%d%H")
    d.mkdir(parents=True, exist_ok=True)
    for name, png, cap in out:
        (d / f"{name}.png").write_bytes(png)
    (d / "captions.json").write_text(json.dumps({n: c for n, _, c in out}, ensure_ascii=False))
    return out
