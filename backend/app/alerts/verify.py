"""Verification of the alerts after the events.

For every area and day, three things side by side:

* the service — confirmed orange or red (an alert was sent), in watch, or nothing; with the
  ENS probability of orange one and two days before (the calibration of the probabilities);
* IPMA — the highest official warning of the same type for that area and day. The warnings
  feed holds only the warnings in force, so it is archived at every cycle (all levels, with
  the time each warning was first seen); before the archive existed only the orange and red
  warnings relayed to Telegram are known, without that time;
* observed — what the stations measured, where the open data allow it: the IPMA stations
  (hourly rain, mean wind and temperature; the open data hold only the last 24 h, so they are
  archived every hour) and the METAR of the airports (gusts and visibility, from the Iowa
  Environmental Mesonet archive). There are no open observations of gusts away from the
  airports, of waves, snow, hail or visibility in the valleys.

The reference for hits and misses is the IPMA warning: the service exists to anticipate the
orange and red warnings. An area-day is a hit (alert and IPMA orange or red), a false alarm
(alert, IPMA yellow or nothing) or a miss (IPMA orange or red without an alert); for the hits,
the level (the same, lower, higher) and the lead over IPMA (the alert reached the phone before
the IPMA warning first appeared in its feed). The observations are shown beside it.

A day is verified the next morning (≥ VERIFY_HOUR UTC). When a confirmed event has ended (or
was cancelled) a verification message is sent; IPMA orange or red warnings that no alert
announced get one too. The scores accumulate in ``summary()`` (POD, FAR, lead, reliability of
the probabilities, Brier score) for tuning the thresholds once there are enough cases.
"""
from __future__ import annotations

import csv
import datetime as dt
import gzip
import io
import json
from functools import lru_cache

import httpx
import numpy as np

from ..config import ALERTS_DIR, USER_AGENT
from . import criteria as C
from .areas import areas, by_code, district_geometry

VDIR = ALERTS_DIR / "verif"
WARN_FILE = VDIR / "ipma_warnings.json"
OBS_DIR = VDIR / "obs"
DAYS_DIR = VDIR / "days"
STATIONS_FILE = VDIR / "stations.json"
STATE_KEY = "verif"
OBS_URL = "https://api.ipma.pt/open-data/observation/meteorology/stations/observations.json"
STATIONS_URL = "https://api.ipma.pt/open-data/observation/meteorology/stations/stations.json"
IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
AIRPORTS = ["LPPT", "LPPR", "LPFR", "LPBJ", "LPMA", "LPPS", "LPPD", "LPAZ", "LPLA", "LPHR", "LPFL",
            "LPMR", "LPMT", "LPOV", "LPST", "LPCS", "LPAR", "LPOT"]
START = dt.date(2026, 11, 20)            # the service's first day
VERIFY_HOUR = 7                          # UTC: a day is verified the next morning
KEEP_DAYS = 400
H = dt.timedelta(hours=1)

TYPES = {"Precipitação": ("chuva", "chuva1h"), "Vento": ("vento",), "Tempo Quente": ("calor",),
         "Tempo Frio": ("frio",), "Agitação Marítima": ("mar",), "Neve": ("neve",),
         "Trovoada": ("trovoada",), "Nevoeiro": ("nevoeiro",)}
TYPE_ICON = {"Precipitação": "🌧️", "Vento": "💨", "Tempo Quente": "🌡️", "Tempo Frio": "🥶",
             "Agitação Marítima": "🌊", "Neve": "❄️", "Trovoada": "⛈️", "Nevoeiro": "🌫️"}
_RANK = {None: 0, "yellow": 1, "orange": 2, "red": 3}
_FC_RANK = {None: 0, "vigilancia": 1, "cancelado": 1, "laranja": 2, "vermelho": 3}
_LVL_PT = {"yellow": "amarelo", "orange": "laranja", "red": "vermelho", None: "sem aviso"}


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def _load(p, default):
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def _save(p, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False))
    tmp.replace(p)


# ---------------------------------------------------------------- IPMA warnings archive
def _wkey(w: dict) -> str:
    return f"{w['area']}|{w['type']}|{w['level']}|{w['start'][:16]}"


def archive_warnings(ws: list[dict], now: dt.datetime | None = None) -> None:
    """Merge the warnings now in the IPMA feed into the archive (every level; first and last
    time seen; a later end time extends the warning)."""
    now = now or _now()
    t = now.isoformat(timespec="minutes")
    arch = _load(WARN_FILE, {})
    for w in ws:
        k = _wkey(w)
        a = arch.get(k)
        if a is None:
            arch[k] = {"area": w["area"], "type": w["type"], "level": w["level"], "start": w["start"],
                       "end": w["end"], "text": w.get("text", ""), "first_seen": t, "last_seen": t}
        else:
            a["last_seen"] = t
            if w["end"] and (not a.get("end") or w["end"] > a["end"]):
                a["end"] = w["end"]
    cut = (now - dt.timedelta(days=KEEP_DAYS)).isoformat()
    arch = {k: v for k, v in arch.items() if (v.get("end") or v["start"]) >= cut}
    _save(WARN_FILE, arch)


def backfill_from_state(st: dict) -> int:
    """Orange/red warnings relayed before the archive existed (state "seen": area, type, level,
    start hour — no end, no first-seen time). Returns how many were added."""
    arch = _load(WARN_FILE, {})
    n = 0
    for s in st.get("ipma", {}).get("seen", []):
        try:
            area, typ, lvl, start = s.split("|")
        except ValueError:
            continue
        start = f"{start}:00:00" if len(start) == 13 else start
        if any(v["area"] == area and v["type"] == typ and v["level"] == lvl and v["start"][:13] == start[:13]
               for v in arch.values()):
            continue
        arch[f"{area}|{typ}|{lvl}|{start[:16]}"] = {"area": area, "type": typ, "level": lvl, "start": start,
                                                     "end": None, "text": "", "first_seen": None,
                                                     "last_seen": None, "backfill": True}
        n += 1
    if n:
        _save(WARN_FILE, arch)
    return n


def _overlaps(w: dict, day: dt.date) -> bool:
    t0 = dt.datetime(day.year, day.month, day.day)
    s = dt.datetime.fromisoformat(w["start"][:19])
    e = dt.datetime.fromisoformat(w["end"][:19]) if w.get("end") else s + 6 * H
    return s < t0 + 24 * H and e > t0


def ipma_day(day: dt.date) -> dict:
    """{(type, area): {"level", "first_seen" (of the first orange/red), "start"}} for one day."""
    out: dict = {}
    for w in _load(WARN_FILE, {}).values():
        if not _overlaps(w, day):
            continue
        k = (w["type"], w["area"])
        r = out.setdefault(k, {"level": None, "first_seen": None, "start": None})
        if _RANK[w["level"]] > _RANK[r["level"]]:
            r["level"] = w["level"]
        if w["level"] in ("orange", "red"):
            fs = w.get("first_seen")
            if fs and (r["first_seen"] is None or fs < r["first_seen"]):
                r["first_seen"] = fs
            if r["start"] is None or w["start"] < r["start"]:
                r["start"] = w["start"]
    return out


# ---------------------------------------------------------------- observations
def area_of(lat: float, lon: float) -> str | None:
    """Warning area of a point: the district that contains it, or the island area of the
    nearest town within 0.35°."""
    return _area_of(round(lat, 4), round(lon, 4))


@lru_cache(maxsize=2048)
def _area_of(lat, lon):
    import shapely
    for a in areas():
        g = district_geometry(a.code)
        if g is not None and shapely.contains_xy(g, lon, lat):
            return a.code
    best, bd = None, 0.35
    for a in areas():
        if a.group == "continente":
            continue
        for _, la, lo in a.towns:
            d = float(np.hypot(la - lat, (lo - lon) * np.cos(np.radians(la))))
            if d <= bd:
                best, bd = a.code, d
    return best


async def _get_json(url: str):
    async with httpx.AsyncClient(timeout=60, headers={"User-Agent": USER_AGENT}) as c:
        r = await c.get(url)
        r.raise_for_status()
        return r.json()


async def stations() -> dict:
    """{station id: {"name", "lat", "lon", "area"}} (refreshed monthly)."""
    st = _load(STATIONS_FILE, {})
    if st and _now().timestamp() - STATIONS_FILE.stat().st_mtime < 30 * 86400:
        return st
    data = await _get_json(STATIONS_URL)
    out = {}
    for f in data:
        lon, lat = f["geometry"]["coordinates"]
        sid = str(f["properties"]["idEstacao"])
        out[sid] = {"name": f["properties"]["localEstacao"], "lat": lat, "lon": lon, "area": area_of(lat, lon)}
    _save(STATIONS_FILE, out)
    return out


def _obs_path(day: dt.date):
    return OBS_DIR / f"{day.isoformat()}.json.gz"


def _obs_load(day: dt.date) -> dict:
    p = _obs_path(day)
    try:
        return json.loads(gzip.decompress(p.read_bytes()))
    except (OSError, ValueError):
        return {}


async def archive_obs(st: dict, now: dt.datetime | None = None) -> None:
    """Hourly IPMA station observations of the last 24 h into daily files (at most once an
    hour): {station: {"HH": [temperature °C, rain mm in the hour, mean wind km/h]}}."""
    now = now or _now()
    vs = st.setdefault(STATE_KEY, {})
    last = vs.get("obs_fetched")
    if last and now - dt.datetime.fromisoformat(last) < H:
        return
    await stations()
    data = await _get_json(OBS_URL)
    by_day: dict[dt.date, dict] = {}
    for t, rows in data.items():
        tt = dt.datetime.fromisoformat(t[:16])
        d = by_day.setdefault(tt.date(), _obs_load(tt.date()))
        for sid, v in (rows or {}).items():
            if not v:
                continue
            vals = [v.get("temperatura"), v.get("precAcumulada"), v.get("intensidadeVentoKM")]
            vals = [None if x is None or x <= -99 else round(float(x), 1) for x in vals]
            d.setdefault(sid, {})[f"{tt.hour:02d}"] = vals
    OBS_DIR.mkdir(parents=True, exist_ok=True)
    for day, d in by_day.items():
        _obs_path(day).write_bytes(gzip.compress(json.dumps(d).encode()))
    cut = (now - dt.timedelta(days=KEEP_DAYS)).date()
    for p in OBS_DIR.glob("*.json.gz"):
        try:
            if dt.date.fromisoformat(p.name[:10]) < cut:
                p.unlink()
        except ValueError:
            pass
    vs["obs_fetched"] = now.isoformat(timespec="seconds")


def obs_day(day: dt.date) -> dict:
    """Per area, the largest of what the IPMA stations measured on the day: rain in 1 h and
    in 6 h (windows may start the evening before), mean wind, hourly Tmax/Tmin — each with
    its station. Empty if the day was not archived."""
    today, prev = _obs_load(day), _obs_load(day - dt.timedelta(days=1))
    if not today:
        return {}
    sts = _load(STATIONS_FILE, {})
    out: dict = {}

    def best(code, key, v, sid, high=True):
        r = out.setdefault(code, {})
        if v is None:
            return
        if key not in r or (v > r[key][0] if high else v < r[key][0]):
            r[key] = (round(v, 1), sts.get(sid, {}).get("name", sid))

    for sid, hours in today.items():
        code = sts.get(sid, {}).get("area")
        if not code:
            continue
        ser = [(prev.get(sid, {}).get(f"{h:02d}") or [None] * 3) for h in range(19, 24)] + \
              [(hours.get(f"{h:02d}") or [None] * 3) for h in range(24)]
        rain = [x[1] for x in ser]
        day_part = ser[5:]
        r1 = [x for x in rain[5:] if x is not None]
        best(code, "rain1h", max(r1) if r1 else None, sid)
        sums = [sum(rain[i:i + 6]) for i in range(len(rain) - 5)       # windows ending on the day
                if all(x is not None for x in rain[i:i + 6])]
        best(code, "rain6h", max(sums) if sums else None, sid)
        w = [x[2] for x in day_part if x[2] is not None]
        best(code, "wind", max(w) if w else None, sid)
        t = [x[0] for x in day_part if x[0] is not None]
        if len(t) >= 18:                                     # a day with most hours
            best(code, "tmax", max(t), sid)
            best(code, "tmin", min(t), sid, high=False)
    return out


async def metar_day(day: dt.date) -> dict:
    """Per area, from the airports' METAR: the largest gust (km/h) and the hours with
    visibility under 1 km. Empty on failure."""
    params = [("station", s) for s in AIRPORTS] + [
        ("data", "gust"), ("data", "vsby"), ("year1", day.year), ("month1", day.month), ("day1", day.day),
        ("year2", (day + dt.timedelta(days=1)).year), ("month2", (day + dt.timedelta(days=1)).month),
        ("day2", (day + dt.timedelta(days=1)).day), ("tz", "Etc/UTC"), ("format", "onlycomma"),
        ("latlon", "yes"), ("missing", "M"), ("trace", "T"), ("direct", "no"),
        ("report_type", "3"), ("report_type", "4")]
    try:
        async with httpx.AsyncClient(timeout=60, headers={"User-Agent": USER_AGENT}) as c:
            r = await c.get(IEM_URL, params=params)
            r.raise_for_status()
    except httpx.HTTPError:
        return {}
    out: dict = {}
    fog_hours: dict = {}
    for row in csv.DictReader(io.StringIO(r.text)):
        try:
            t = dt.datetime.fromisoformat(row["valid"])
            code = area_of(float(row["lat"]), float(row["lon"]))
        except (KeyError, ValueError):
            continue
        if t.date() != day or not code:
            continue
        o = out.setdefault(code, {})
        if row.get("gust") not in (None, "", "M"):
            g = float(row["gust"]) * 1.852
            if g > o.get("gust", (0,))[0]:
                o["gust"] = (round(g), row["station"])
        if row.get("vsby") not in (None, "", "M") and float(row["vsby"]) * 1.609 < 1.0:
            fog_hours.setdefault((code, row["station"]), set()).add(t.hour)
    for (code, stn), hrs in fog_hours.items():
        o = out.setdefault(code, {})
        if len(hrs) > o.get("fog_h", (0,))[0]:
            o["fog_h"] = (len(hrs), stn)
    return out


def obs_level(typ: str, code: str, o: dict) -> str | None:
    """IPMA level the observations reach (only where they can tell)."""
    lv = None

    def up(v, thr):
        nonlocal lv
        if v is None:
            return
        x = "red" if v >= thr["vermelho"] else "orange" if v >= thr["laranja"] else None
        if _RANK[x] > _RANK[lv]:
            lv = x
    if typ == "Precipitação":
        up(o.get("rain6h", (None,))[0], C.RAIN6)
        up(o.get("rain1h", (None,))[0], C.RAIN1)
    elif typ == "Vento":
        grp = by_code()[code].group
        up(o.get("gust", (None,))[0], C.gust(code, grp))
        if grp == "acores":
            up(o.get("wind", (None,))[0], C.MEAN_WIND_AZORES)
    elif typ == "Tempo Quente" and o.get("tmax"):
        t = C.heat(code)
        v = o["tmax"][0]
        lv = "red" if v >= t["vermelho"] else "orange" if v >= t["laranja"] else None
    elif typ == "Tempo Frio" and o.get("tmin"):
        t = C.cold(code)
        v = o["tmin"][0]
        lv = "red" if v <= t["vermelho"] else "orange" if v <= t["laranja"] else None
    return lv


def obs_text(typ: str, o: dict) -> str:
    if not o:
        return ""
    if typ in ("Precipitação", "Trovoada"):
        parts = []
        if o.get("rain6h"):
            parts.append(f"{o['rain6h'][0]:.0f} mm em 6 h ({o['rain6h'][1]})")
        if o.get("rain1h") and o["rain1h"][0] >= 5:
            parts.append(f"{o['rain1h'][0]:.0f} mm em 1 h ({o['rain1h'][1]})")
        return ", ".join(parts)
    if typ == "Vento":
        parts = []
        if o.get("gust"):
            parts.append(f"rajada {o['gust'][0]} km/h ({o['gust'][1]})")
        if o.get("wind"):
            parts.append(f"vento médio {o['wind'][0]:.0f} km/h ({o['wind'][1]})")
        return ", ".join(parts)
    if typ == "Tempo Quente" and o.get("tmax"):
        return f"máxima horária {o['tmax'][0]:.1f} °C ({o['tmax'][1]})".replace(".", ",")
    if typ == "Tempo Frio" and o.get("tmin"):
        return f"mínima horária {o['tmin'][0]:.1f} °C ({o['tmin'][1]})".replace(".", ",")
    if typ == "Nevoeiro" and o.get("fog_h"):
        return f"visibilidade < 1 km durante {o['fog_h'][0]} h ({o['fog_h'][1]})"
    return ""


# ---------------------------------------------------------------- what the service said
def _msg_after(ev: dict, run_iso: str) -> str | None:
    """Time the first alert/update reached Telegram after the analysis of ``run_iso``."""
    for m in ev.get("messages", []):
        if m["kind"] in ("alerta", "atualizacao") and m["t"] >= run_iso:
            return m["t"]
    return None


def forecast_day(st: dict, phens, day: str) -> dict:
    """{area: {"lvl", "t" (alert time), "p" (largest P(orange)), "ev"}} — the service's highest
    status for that day over every run of the events of these phenomena. An alert later
    cancelled counts as "cancelado" (what the user was last told)."""
    out: dict = {}
    for ev in st.get("events", []):
        if ev["phen"] not in phens or day not in ev.get("days", []):
            continue
        cancelled = ev["status"] == "cancelado"
        for h in ev.get("history", []):
            cdays = h.get("conf_days")
            for code in h.get("conf", []):
                if cdays is not None and day not in cdays.get(code, []):
                    continue
                lvl = "cancelado" if cancelled else (h.get("conf_lev", {}).get(code) or h.get("level") or "laranja")
                r = out.setdefault(code, {"lvl": None, "t": None, "p": 0.0, "ev": ev["id"]})
                if _FC_RANK[lvl] > _FC_RANK[r["lvl"]]:
                    r["lvl"], r["ev"] = lvl, ev["id"]
                t = _msg_after(ev, h["run"]) or (dt.datetime.fromisoformat(h["run"]) + 8 * H).isoformat(
                    timespec="minutes")
                if r["t"] is None or t < r["t"]:
                    r["t"] = t
            for code in h.get("watch", []):
                r = out.setdefault(code, {"lvl": None, "t": None, "p": 0.0, "ev": ev["id"]})
                if r["lvl"] is None:
                    r["lvl"] = "vigilancia"
            for code, pp in h.get("p", {}).items():
                if code in out:
                    out[code]["p"] = max(out[code]["p"], float(pp[0]))
    return out


def p_before(phens, day: dt.date, lead: int) -> dict | None:
    """{area: P(orange)} of the 00 UTC run ``lead`` days before (the largest of the phenomena);
    areas without a signal have 0. None if that run was not analysed."""
    from .detect import RUNS_DIR
    run = day - dt.timedelta(days=lead)
    sig = _load(RUNS_DIR / f"{run:%Y%m%d}00" / "signals.json", None)
    if sig is None:
        return None
    if any(p in sig.get("failed", []) for p in phens):
        return None
    out: dict = {}
    for phen in phens:
        for code, row in sig.get("signals", {}).get(phen, {}).get(day.isoformat(), {}).items():
            out[code] = max(out.get(code, 0.0), float(row.get("p_o", 0.0)))
    return out


# ---------------------------------------------------------------- one day
async def verify_day(st: dict, day: dt.date) -> dict:
    """The verification record of one day, saved in DAYS_DIR."""
    iso = day.isoformat()
    ip = ipma_day(day)
    obs = obs_day(day)
    met = await metar_day(day)
    for code, o in met.items():
        obs.setdefault(code, {}).update(o)
    rec = {"day": iso, "verified": _now().isoformat(timespec="minutes"), "obs": bool(obs),
           "types": {}}
    codes = [a.code for a in areas()]
    for typ, phens in TYPES.items():
        fc = forecast_day(st, phens, iso)
        p1, p2 = p_before(phens, day, 1), p_before(phens, day, 2)
        rows = {}
        for code in codes:
            f = fc.get(code, {})
            w = ip.get((typ, code), {})
            o = obs.get(code, {})
            ol = obs_level(typ, code, o)
            if not (f.get("lvl") or w.get("level") or ol):
                continue
            alerted = f.get("lvl") in ("laranja", "vermelho")
            warned = w.get("level") in ("orange", "red")
            outcome = "acerto" if alerted and warned else "alarme_falso" if alerted else \
                "falha" if warned else None
            lead = None
            if alerted and warned and f.get("t") and w.get("first_seen"):
                lead = round((dt.datetime.fromisoformat(w["first_seen"])
                              - dt.datetime.fromisoformat(f["t"])) / H, 1)
            rows[code] = {"fc": f.get("lvl"), "fc_p": round(f.get("p", 0.0), 2), "fc_t": f.get("t"),
                          "ev": f.get("ev"), "ipma": w.get("level"), "ipma_seen": w.get("first_seen"),
                          "obs": obs_text(typ, o), "obs_lvl": ol, "outcome": outcome, "lead_h": lead,
                          "p1": None if p1 is None else round(p1.get(code, 0.0), 2),
                          "p2": None if p2 is None else round(p2.get(code, 0.0), 2)}
        rec["types"][typ] = {"rows": rows, "n": len(codes),
                             "p1": None if p1 is None else {c: round(v, 2) for c, v in p1.items() if v > 0},
                             "p2": None if p2 is None else {c: round(v, 2) for c, v in p2.items() if v > 0}}
    _save(DAYS_DIR / f"{iso}.json", rec)
    return rec


def load_days() -> list[dict]:
    return [r for r in (_load(p, None) for p in sorted(DAYS_DIR.glob("*.json"))) if r]


# ---------------------------------------------------------------- scores
def summary(days: list[dict] | None = None) -> dict:
    """Cumulative scores per warning type: hits, false alarms, misses, POD, FAR, lead over
    IPMA, level agreement; reliability and Brier score of P(orange) one and two days before
    (reference: IPMA orange or red)."""
    days = load_days() if days is None else days
    out = {"days": len(days), "from": days[0]["day"] if days else None,
           "to": days[-1]["day"] if days else None, "types": {}}
    bins = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
    for typ in TYPES:
        hit = fa = miss = 0
        lv_same = lv_low = lv_high = 0
        leads = []
        rel = {k: [[0, 0] for _ in range(len(bins) - 1)] for k in ("p1", "p2")}
        brier = {k: [0.0, 0] for k in ("p1", "p2")}
        for d in days:
            t = d["types"].get(typ)
            if not t:
                continue
            for r in t["rows"].values():
                if r["outcome"] == "acerto":
                    hit += 1
                    fr, ir = _FC_RANK[r["fc"]], _RANK[r["ipma"]]
                    lv_same += fr == ir
                    lv_low += fr < ir
                    lv_high += fr > ir
                    if r["lead_h"] is not None:
                        leads.append(r["lead_h"])
                elif r["outcome"] == "alarme_falso":
                    fa += 1
                elif r["outcome"] == "falha":
                    miss += 1
            warned = {c for c, r in t["rows"].items() if r["ipma"] in ("orange", "red")}
            for k in ("p1", "p2"):
                if t.get(k) is None:
                    continue
                ps = t[k]
                for c in [a.code for a in areas()]:
                    p, o = ps.get(c, 0.0), 1.0 if c in warned else 0.0
                    i = int(np.searchsorted(bins, p, side="right") - 1)
                    rel[k][i][0] += 1
                    rel[k][i][1] += int(o)
                    brier[k][0] += (p - o) ** 2
                    brier[k][1] += 1
        n = hit + fa + miss
        out["types"][typ] = {
            "acertos": hit, "alarmes_falsos": fa, "falhas": miss,
            "pod": round(hit / (hit + miss), 2) if hit + miss else None,
            "far": round(fa / (hit + fa), 2) if hit + fa else None,
            "csi": round(hit / n, 2) if n else None,
            "nivel": {"igual": lv_same, "abaixo": lv_low, "acima": lv_high},
            "antecedencia_h": round(float(np.median(leads)), 1) if leads else None, "n_antecedencia": len(leads),
            "fiabilidade": {k: [{"p": f"{bins[i]:.0%}–{min(bins[i + 1], 1):.0%}", "n": v[0], "obs": v[1]}
                                for i, v in enumerate(rel[k]) if v[0]] for k in rel},
            "brier": {k: round(v[0] / v[1], 4) if v[1] else None for k, v in brier.items()}}
    return out


# ---------------------------------------------------------------- messages
def _day_pt(iso: str) -> str:
    from .messages import fmt_day
    return fmt_day(dt.date.fromisoformat(iso))


def _t_pt(iso: str | None) -> str:
    if not iso:
        return "?"
    t = dt.datetime.fromisoformat(iso)
    from .messages import MON, WK
    return f"{WK[t.weekday()]} {t.day} {MON[t.month - 1]} {t:%H:%M}"


_ICON = {"acerto": "✅", "alarme_falso": "⚠️", "falha": "❌", None: "▫️"}


def _row_line(code: str, day: str, r: dict, multi: bool) -> str:
    from .messages import esc, pct
    name = by_code()[code].name
    fc = {"vermelho": f"vermelho {pct(r['fc_p'])}", "laranja": f"laranja {pct(r['fc_p'])}",
          "vigilancia": f"vigilância {pct(r['fc_p'])}", "cancelado": "alerta cancelado",
          None: "sem sinal"}[r["fc"]]
    if r["fc"] is None and r.get("p1") is not None:
        fc += f" (P 1 dia antes {pct(r['p1'])})"
    s = f"{_ICON[r['outcome']]} <b>{esc(name)}</b>{f' ({_day_pt(day)})' if multi else ''}: {fc} → IPMA {_LVL_PT[r['ipma']]}"
    if r.get("lead_h") is not None:
        s += f" ({'+' if r['lead_h'] >= 0 else ''}{r['lead_h']:.0f} h)"
    if r.get("obs"):
        s += f" · {esc(r['obs'])}"
    return s


def event_report(ev: dict, recs: dict[str, dict]) -> str:
    """Verification text of an ended (or cancelled) event."""
    from .messages import esc, fmt_days, pct
    phen = ev["phen"]
    typ = C.IPMA_TYPE[phen]
    P = C.PHENOMENA[phen]
    days = sorted(ev["days"])
    multi = len(days) > 1
    lines = [f"🔎 <b>VERIFICAÇÃO · {esc(P['name'])}</b> · {fmt_days(days)}"]
    alert = next((m for m in ev.get("messages", []) if m["kind"] == "alerta"), None)
    if alert:
        lines.append(f"Alerta {ev.get('notified', {}).get('level') or 'laranja'} enviado {_t_pt(alert['t'])}"
                     + (" · depois cancelado" if any(m["kind"] == "cancelamento" for m in ev["messages"]) else ""))
    rows = []
    for d in days:
        t = recs.get(d, {}).get("types", {}).get(typ, {})
        for code, r in t.get("rows", {}).items():
            if r["fc"] or r["ipma"] in ("orange", "red"):
                rows.append((d, code, r))
    rank = {"falha": 0, "acerto": 1, "alarme_falso": 2, None: 3}
    rows.sort(key=lambda x: (rank[x[2]["outcome"]], x[0], -_RANK[x[2]["ipma"]], -x[2]["fc_p"]))
    lines += ["", "<b>Por área</b> (serviço → IPMA · observado):"]
    lines += [_row_line(c, d, r, multi) for d, c, r in rows[:14]]
    if len(rows) > 14:
        lines.append(f"… e mais {len(rows) - 14}")
    n = {k: sum(1 for _, _, r in rows if r["outcome"] == k) for k in ("acerto", "alarme_falso", "falha")}
    leads = [r["lead_h"] for _, _, r in rows if r.get("lead_h") is not None]
    s = f"{n['acerto']} acerto(s) · {n['alarme_falso']} alarme(s) falso(s) · {n['falha']} falha(s)"
    if leads:
        s += f" · antecedência face ao IPMA: mediana {np.median(leads):+.0f} h"
    lines += ["", s]
    if not any(recs.get(d, {}).get("obs") for d in days):
        lines.append("<i>Sem observações das estações do IPMA arquivadas para estes dias.</i>")
    lines.append("<i>Referência: o aviso do IPMA (laranja ou vermelho); o observado vem das estações do IPMA "
                 "e dos METAR dos aeroportos. Entre parênteses: horas de avanço do alerta sobre o aviso.</i>")
    return "\n".join(lines)


def miss_report(day: str, typ: str, rows: dict) -> str:
    """Text for IPMA orange/red warnings of one type and day that no alert announced."""
    from .messages import esc
    lines = [f"🔎 <b>VERIFICAÇÃO · {TYPE_ICON.get(typ, '')} {esc(typ)}</b> · {_day_pt(day)} — não previsto",
             "O IPMA teve aviso laranja ou vermelho onde o serviço não deu alerta:", ""]
    lines += [_row_line(c, day, r, False) for c, r in rows.items()]
    lines.append("")
    lines.append("<i>Fica nas pontuações como falha; serve para afinar os limiares.</i>")
    return "\n".join(lines)


# ---------------------------------------------------------------- service step
def _due(day: dt.date, now: dt.datetime) -> bool:
    return now >= dt.datetime(day.year, day.month, day.day) + dt.timedelta(days=1, hours=VERIFY_HOUR)


async def step(st: dict, now: dt.datetime | None = None, log=print) -> list[str]:
    """Called every cycle: archive the observations (hourly), verify the days that are due,
    and return the texts to send (event verifications and misses). The days before the
    verification was installed are verified silently (stored, shown in the app, not sent)."""
    now = now or _now()
    vs = st.setdefault(STATE_KEY, {})
    since = vs.setdefault("since", now.date().isoformat())
    try:
        await archive_obs(st, now)
    except Exception as e:  # noqa: BLE001 — observations are only context
        log(f"verificação: observações do IPMA falharam ({e})")
    if not vs.get("backfilled"):
        n = backfill_from_state(st)
        vs["backfilled"] = True
        if n:
            log(f"verificação: {n} aviso(s) laranja/vermelho do IPMA recuperados do estado")
    done = set(vs.get("days_done", []))
    d = max(START, (now - dt.timedelta(days=KEEP_DAYS)).date())
    todo = []
    while _due(d, now):
        if d.isoformat() not in done:
            todo.append(d)
        d += dt.timedelta(days=1)
    recs = {}
    for d in todo[-10:]:
        recs[d.isoformat()] = await verify_day(st, d)
        done.add(d.isoformat())
        log(f"verificação: dia {d} feito")
    vs["days_done"] = sorted(done)[-KEEP_DAYS:]
    out = []
    # events that ended: one verification message each
    for ev in st.get("events", []):
        if ev.get("verif") or ev["status"] not in ("terminado", "cancelado") or not ev.get("notified"):
            continue
        if not all(x in done for x in ev["days"]):
            continue
        rr = {x: _load(DAYS_DIR / f"{x}.json", {}) for x in ev["days"]}
        txt = event_report(ev, rr)
        ev["verif"] = {"t": now.isoformat(timespec="seconds"), "text": txt}
        if max(ev["days"]) >= since:
            out.append(txt)
    # IPMA orange/red no alert announced (one message per day and type)
    sent = set(vs.get("misses_sent", []))
    for iso, rec in recs.items():
        for typ, t in rec["types"].items():
            miss = {c: r for c, r in t["rows"].items() if r["outcome"] == "falha"}
            k = f"{iso}|{typ}"
            if miss and k not in sent:
                if iso >= since:
                    out.append(miss_report(iso, typ, miss))
                sent.add(k)
    vs["misses_sent"] = sorted(sent)[-500:]
    return out
