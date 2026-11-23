"""Alert events: classification of area-days, grouping into events, life cycle and the
decision of what to send.

Life cycle of an event (one phenomenon, consecutive days, any number of areas):

* ``vigilancia`` — some members reach the orange criterion (≥ ``p_watch``); shown in the
  app only, nothing is sent;
* ``confirmado`` — at least one area-day meets the certainty rule (below); an alert is
  sent with the figures, then updates when something important changes;
* ``cancelado`` — a confirmed event that the forecasts dropped (sent);
* ``terminado`` — its last day has passed.

Certainty rule for an area-day at level L (orange or red):
1. ENS probability ≥ ``p_confirm[L]`` in the current run;
2. ≥ ``p_previous[L]`` in the previous ENS run (12 h earlier): the signal is consistent;
3. at least ``det_support`` of the deterministic models available (IFS, AIFS, GFS,
   ICON-EU; for rain in 1 h only ICON-EU and GFS, the ones with hourly rain; for
   thunderstorms IFS, GFS and ICON-EU, the ones with CAPE; for fog IFS, GFS and ICON-EU, the
   ones with a 15 UTC step) come within a margin of the threshold;
4. the day is within ``lead_days`` (wind, rain, sea, snow and fog 4 days, heat and cold 6, rain
   in 1 h 2: the hourly ensembles reach +48 h; thunderstorms 2, the
   ensemble has little skill for storms beyond that).
For rain in 1 h the ensemble is the DWD ICON-EU-EPS (in the Azores the global ICON-EPS; 40
members) instead of the ECMWF ENS;
it is analysed with each ECMWF run, so "the previous run" is the analysis 12 h earlier.
Once confirmed, an area keeps its level while its probability stays ≥ ``p_hold`` (so a
small dip does not cancel and re-send the alert); new areas need the full rule.
"""
from __future__ import annotations

import datetime as dt
import json
import os

from ..config import ALERTS_DIR
from . import criteria as C

STATE_FILE = ALERTS_DIR / "state.json"
CONFIG_FILE = ALERTS_DIR / "config.json"

DEFAULTS = {
    "p_confirm": {"laranja": 0.6, "vermelho": 0.5},
    "p_previous": {"laranja": 0.4, "vermelho": 0.3},
    "p_hold": {"laranja": 0.25, "vermelho": 0.25},
    "p_watch": 0.2,
    "det_support": 0.5,
    "lead_days": {"vento": 4, "chuva": 4, "calor": 6, "frio": 6, "mar": 4, "neve": 4, "chuva1h": 2,
                  "trovoada": 2, "nevoeiro": 4},
    "quiet_hours": [23, 8],                 # Lisbon time: alerts arrive silently (red ones ring)
    "ipma_levels": ["orange", "red"],       # official IPMA warnings that are relayed
}
_RANK = {None: 0, "vigilancia": 1, "laranja": 2, "vermelho": 3}


def load_config() -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))
    try:
        user = json.loads(CONFIG_FILE.read_text())
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    except (OSError, json.JSONDecodeError):
        pass
    return cfg


def load_state() -> dict:
    try:
        st = json.loads(STATE_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        st = {}
    st.setdefault("events", [])
    st.setdefault("runs", [])
    st.setdefault("ipma", {"seen": [], "active": []})
    st.setdefault("telegram", {"offset": 0})
    return st


def save_state(st: dict) -> None:
    tmp = STATE_FILE.with_name(f"state.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1, default=str))
    tmp.replace(STATE_FILE)


def _d(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


# ---------------------------------------------------------------- classification
def support(row: dict, red: bool = False) -> tuple[float, int]:
    det = row.get("det", {})
    if not det:
        return 0.0, 0
    k = "near_r" if red else "near"
    return sum(1 for x in det.values() if x.get(k)) / len(det), len(det)


def classify(phen: str, run: dt.datetime, day: dt.date, row: dict, prev: dict | None,
             held: str | None, cfg: dict) -> str | None:
    """Status of one area-day: 'vermelho' | 'laranja' | 'vigilancia' | None.
    ``held`` is the level the area already had in a confirmed event."""
    lead_ok = (day - run.date()).days <= cfg["lead_days"].get(phen, 4)
    prev = prev or {"p_o": 0.0, "p_r": 0.0}
    for lvl, pk in (("vermelho", "p_r"), ("laranja", "p_o")):
        sup, n = support(row, red=lvl == "vermelho")
        need = cfg["p_confirm"][lvl] + (0.2 if n == 0 else 0.0)   # no model to check: be stricter
        full = (lead_ok and row[pk] >= need and prev[pk] >= cfg["p_previous"][lvl]
                and (n == 0 or sup >= cfg["det_support"]))
        hold = held is not None and _RANK[held] >= _RANK[lvl] and row[pk] >= cfg["p_hold"][lvl]
        if full or hold:
            return lvl
    if row["p_o"] >= cfg["p_watch"]:
        return "vigilancia"
    return None


def _clusters(days: list[dt.date]) -> list[list[dt.date]]:
    out: list[list[dt.date]] = []
    for d in sorted(set(days)):
        if out and (d - out[-1][-1]).days <= 2:      # one quiet day in between: same event
            out[-1].append(d)
        else:
            out.append([d])
    return out


# ---------------------------------------------------------------- update with a run
def _active(ev) -> bool:
    return ev["status"] in ("vigilancia", "confirmado")


def update(st: dict, analysis: dict, prev_analysis: dict | None, cfg: dict,
           now: dt.datetime | None = None) -> list[dict]:
    """Apply one analysed ENS run to the state; return the notifications to send
    [{"event": id, "kind": "alerta" | "atualizacao" | "cancelamento", "why": [...]}]."""
    now = now or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    run = dt.datetime.fromisoformat(analysis["run"])
    runs_iso = analysis["run"]
    notes: list[dict] = []
    failed = set(analysis.get("failed", []))
    events = st["events"]
    touched: set[str] = set()

    for phen, by_day in analysis["signals"].items():
        prev_sig = (prev_analysis or {}).get("signals", {}).get(phen, {})
        # levels already held by confirmed events of this phenomenon
        held: dict[tuple[str, str], str] = {}
        for ev in events:
            if ev["phen"] == phen and ev["status"] == "confirmado":
                for code, a in ev["areas"].items():
                    for d in a.get("conf_days", []):
                        held[(code, d)] = a["status"]
        status: dict[tuple[str, str], str] = {}
        for day, rows in by_day.items():
            for code, row in rows.items():
                s = classify(phen, run, _d(day), row, prev_sig.get(day, {}).get(code),
                             held.get((code, day)), cfg)
                if s:
                    status[(code, day)] = s
        for cl in _clusters([_d(d) for (_, d) in status]):
            cl_iso = {d.isoformat() for d in cl}
            ev = next((e for e in events if e["phen"] == phen and _active(e) and e["id"] not in touched
                       and _d(min(e["days"])) - dt.timedelta(days=2) <= cl[-1]
                       and cl[0] <= _d(max(e["days"])) + dt.timedelta(days=2)), None)
            if ev is None:
                ev = {"id": f"{phen}-{cl[0]:%Y%m%d}", "phen": phen, "status": "vigilancia",
                      "first_run": runs_iso, "confirmed_run": None, "history": [], "notified": None,
                      "messages": []}
                base, n = ev["id"], 2
                while any(e["id"] == ev["id"] for e in events):
                    ev["id"] = f"{base}-{n}"
                    n += 1
                events.append(ev)
            touched.add(ev["id"])
            areas: dict[str, dict] = {}
            for (code, day), s in status.items():
                if day not in cl_iso:
                    continue
                row = by_day[day][code]
                a = areas.setdefault(code, {"status": None, "days": [], "conf_days": [],
                                            "p_o": 0.0, "p_r": 0.0})
                a["days"].append(day)
                if s in ("laranja", "vermelho"):
                    a["conf_days"].append(day)
                if _RANK[s] > _RANK[a["status"]]:
                    a["status"] = s
                if row["p_o"] > a["p_o"] or (row["p_o"] == a["p_o"] and row["p_r"] > a["p_r"]):
                    a.update({k: row[k] for k in ("p_o", "p_r", "p10", "p50", "p90", "max",
                                                  "thr_o", "thr_r", "det")})
                    a.update({k: row[k] for k in ("z", "bands", "vm") if k in row})  # snow band, Azores mean wind
                    a["peak_day"] = day
                    a["prev_p_o"] = (prev_sig.get(day, {}).get(code) or {}).get("p_o")
            ev["areas"] = dict(sorted(areas.items(), key=lambda kv: (-_RANK[kv[1]["status"]],
                                                                       -kv[1]["p_o"])))
            ev["days"] = sorted(cl_iso)
            ev["last_run"] = runs_iso
            if analysis.get("src_runs", {}).get(phen):      # 1-h rain: the ICON-EU-EPS run
                ev["src_run"] = analysis["src_runs"][phen]
            conf = [c for c, a in areas.items() if a["status"] in ("laranja", "vermelho")]
            level = max((areas[c]["status"] for c in conf), key=lambda s: _RANK[s], default=None)
            if conf and ev["status"] != "confirmado":
                ev["status"], ev["confirmed_run"] = "confirmado", runs_iso
            ev["level"] = level
            ev["history"] = [h for h in ev["history"] if h["run"] != runs_iso] + [{
                "run": runs_iso, "level": level, "conf": conf,
                "watch": [c for c, a in areas.items() if a["status"] == "vigilancia"],
                "p": {c: [a["p_o"], a["p_r"]] for c, a in areas.items()}}]
            if ev["status"] == "confirmado":
                notes += _decide(ev, conf, level, now)

    # events of this run's phenomena that the forecasts no longer show
    for ev in events:
        if not _active(ev) or ev["id"] in touched or ev["phen"] in failed:
            continue
        if _d(max(ev["days"])) < first_day_of(run):
            continue                       # in the past: handled by tick()
        if ev["status"] == "confirmado":
            ev["status"], ev["level"] = "cancelado", None
            ev["last_run"] = runs_iso
            if analysis.get("src_runs", {}).get(ev["phen"]):
                ev["src_run"] = analysis["src_runs"][ev["phen"]]
            ev["history"].append({"run": runs_iso, "level": None, "conf": [], "watch": [], "p": {}})
            if ev.get("notified"):
                notes.append({"event": ev["id"], "kind": "cancelamento", "why": ["sinal desapareceu"]})
        else:
            ev["status"] = "descartado"
    if runs_iso not in st["runs"]:
        st["runs"].append(runs_iso)
    st["runs"] = sorted(st["runs"])[-20:]
    return notes


def first_day_of(run: dt.datetime) -> dt.date:
    return run.date() if run.hour == 0 else run.date() + dt.timedelta(days=1)


def _decide(ev: dict, conf: list[str], level: str | None, now: dt.datetime) -> list[dict]:
    """What a confirmed event needs to send after this run."""
    n = ev.get("notified")
    if not n:
        return [{"event": ev["id"], "kind": "alerta", "why": ["confirmado"]}]
    why = []
    if _RANK[level] > _RANK[n.get("level")]:
        why.append("nível subiu")
    elif _RANK[level] < _RANK[n.get("level")]:
        why.append("nível desceu")
    new = [c for c in conf if c not in n.get("conf", [])]
    if new:
        why.append("novas áreas: " + ", ".join(new))
    if set(ev["days"]) != set(n.get("days", [])):
        why.append("dias mudaram")
    pmax = max((ev["areas"][c]["p_o"] for c in conf), default=0.0)
    if n.get("p_o", 0) >= 0.6 and pmax < 0.4:
        why.append("certeza diminuiu")
    if not why:
        return []
    return [{"event": ev["id"], "kind": "atualizacao", "why": why}]


def mark_notified(ev: dict, kind: str, text: str, when: dt.datetime) -> None:
    conf = [c for c, a in ev["areas"].items() if a["status"] in ("laranja", "vermelho")]
    ev["notified"] = {"run": ev.get("last_run"), "level": ev.get("level"), "conf": conf,
                      "days": list(ev["days"]),
                      "p_o": max((ev["areas"][c]["p_o"] for c in conf), default=0.0)}
    ev["messages"].append({"t": when.isoformat(timespec="seconds"), "kind": kind, "text": text})


def tick(st: dict, now: dt.datetime | None = None) -> list[dict]:
    """Time-driven changes: events whose last day has passed end."""
    now = now or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    out = []
    for ev in st["events"]:
        if ev["status"] in ("vigilancia", "confirmado") and _d(max(ev["days"])) < now.date():
            was = ev["status"]
            ev["status"] = "terminado" if was == "confirmado" else "descartado"
            if was == "confirmado" and ev.get("notified"):
                out.append({"event": ev["id"], "kind": "fim", "why": []})
    # forget old discarded / finished events after 60 days
    cutoff = (now - dt.timedelta(days=60)).date()
    st["events"] = [e for e in st["events"] if _d(max(e["days"])) >= cutoff or e["status"] == "confirmado"]
    return out


def find(st: dict, eid: str) -> dict | None:
    return next((e for e in st["events"] if e["id"] == eid), None)


def phen_name(phen: str) -> str:
    return C.PHENOMENA[phen]["name"]
