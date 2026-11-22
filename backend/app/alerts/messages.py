"""Texts of the Telegram messages (HTML parse mode) and of the app's alert cards."""
from __future__ import annotations

import datetime as dt
import html

from . import criteria as C
from .areas import GROUP_NAME, by_code

WEEK = ["segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"]
WK = ["seg", "ter", "qua", "qui", "sex", "sáb", "dom"]
MON = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
LEVEL_ICON = {"vermelho": "🟥", "laranja": "🟧", "vigilancia": "▫️"}
MODEL_SHORT = {"ecmwf": "IFS", "aifs": "AIFS", "gfs": "GFS", "icon_eu": "ICON-EU",
               "ifs_wave": "IFS", "gfs_wave": "GFS-Wave"}
ZONE_NAME = {"norte": "Norte", "centro": "Centro", "sul": "Sul"}


def esc(s) -> str:
    return html.escape(str(s), quote=False)


def fmt_day(d: dt.date, long: bool = False) -> str:
    return f"{(WEEK if long else WK)[d.weekday()]} {d.day} {MON[d.month - 1]}"


def fmt_days(days: list[str], long: bool = False) -> str:
    ds = sorted(dt.date.fromisoformat(x) for x in days)
    if len(ds) == 1:
        return fmt_day(ds[0], long)
    if (ds[-1] - ds[0]).days == len(ds) - 1 and ds[0].month == ds[-1].month:
        return f"{WK[ds[0].weekday()]} {ds[0].day} a {WK[ds[-1].weekday()]} {ds[-1].day} {MON[ds[-1].month - 1]}"
    return ", ".join(fmt_day(d) for d in ds)


def fmt_run(iso: str) -> str:
    r = dt.datetime.fromisoformat(iso)
    return f"{r.day} {MON[r.month - 1]} {r:%H} UTC"


def num(v: float, phen: str) -> str:
    """Value as written in Portuguese: waves with one decimal (comma), the rest rounded."""
    return f"{v:.1f}".replace(".", ",") if phen == "mar" else f"{v:.0f}"


def pct(p: float) -> str:
    return f"{round(100 * p):d}%"


def unit(phen: str) -> str:
    return {"vento": "km/h", "chuva": "mm", "calor": "°C", "frio": "°C", "mar": "m", "neve": "cm"}[phen]


def value_what(phen: str) -> str:
    return {"vento": "rajada máxima", "chuva": "máximo de precipitação em 6 h",
            "calor": "temperatura máxima", "frio": "temperatura mínima",
            "mar": "altura significativa das ondas", "neve": "neve acumulada em 24 h"}[phen]


def criterion_text(phen: str, groups: set[str]) -> str:
    if phen == "vento":
        parts = []
        if groups & {"continente", "madeira"}:
            parts.append("rajadas 91–130 km/h (111–130 nas terras altas), vermelho > 130 km/h")
        if "acores" in groups:
            parts.append("Açores: rajadas 101–130 km/h, vermelho > 130 km/h")
        return "; ".join(parts)
    if phen == "chuva":
        return "41–60 mm em 6 h, vermelho > 60 mm (a chuva em 1 h e a trovoada não são avaliadas)"
    if phen == "calor":
        return "Tmáx acima do valor de cada distrito durante ≥ 48 h"
    if phen == "frio":
        return "Tmín abaixo do valor de cada distrito durante ≥ 48 h"
    if phen == "mar":
        parts = []
        if groups & {"continente", "madeira"}:
            parts.append("ondas (Hs) 5–7 m, vermelho > 7 m; costa sul do Algarve com ondulação de "
                         "sueste 3–5 m, vermelho > 5 m")
        if "acores" in groups:
            parts.append("Açores: 7–9 m, vermelho > 9 m")
        return "; ".join(parts)
    if phen == "neve":
        return ("neve acumulada abaixo de 1000 m 6–15 cm (vermelho > 15), de 1000 a 1600 m 11–30 cm "
                "(vermelho > 30), acima de 1600 m 26–50 cm (vermelho > 50); aqui, em 24 h")
    return ""


def where(ev: dict, statuses=("laranja", "vermelho")) -> str:
    """'Norte e Centro', 'Madeira', … from the zones of the areas."""
    ac = by_code()
    names = []
    for code, a in ev["areas"].items():
        if a["status"] not in statuses:
            continue
        ar = ac[code]
        n = ZONE_NAME.get(ar.zone, GROUP_NAME[ar.group]) if ar.group == "continente" else GROUP_NAME[ar.group]
        if n not in names:
            names.append(n)
    order = ["Norte", "Centro", "Sul", "Madeira", "Açores"]
    names.sort(key=lambda n: order.index(n) if n in order else 9)
    if not names:
        return ""
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " e " + names[-1]


def _area_line(phen: str, code: str, a: dict) -> str:
    ar = by_code()[code]
    s = f"{LEVEL_ICON.get(a['status'], '')} <b>{esc(ar.name)}</b> {pct(a['p_o'])}"
    if a.get("p_r", 0) >= 0.1:
        s += f" (vermelho {pct(a['p_r'])})"
    if phen in ("calor", "frio"):
        s += f" · limiar {C.fmt_threshold(phen, a['thr_o'])}"
    elif phen == "mar" and code == "FAR" and a.get("thr_o", 5) < 5:
        s += " · ondulação de sueste (limiar 3 m)"
    elif phen == "neve" and a.get("z"):
        s += f" · {alt_text(a['z'])} (limiar {C.fmt_threshold('neve', a['thr_o'])})"
    return s


def alt_text(z: float) -> str:
    """'a 1000 m', 'acima de 1600 m (1993 m)' — the altitude band of a snow value."""
    z = int(round(z))
    if z > 1600:
        return f"acima de 1600 m ({z} m)"
    return f"a {z} m"


def _values_line(phen: str, code: str, a: dict) -> str:
    ar = by_code()[code]
    u = unit(phen)
    tail = (f"1 em cada 10 membros abaixo de {num(a['p10'], phen)} {u}" if phen == "frio"
            else f"1 em cada 10 membros acima de {num(a['p90'], phen)} {u}")
    s = f"<b>{esc(ar.name)}</b>: mediana {num(a['p50'], phen)} {u} · {tail}"
    if phen == "neve" and a.get("z"):
        s = s.replace("</b>:", f"</b> ({alt_text(a['z'])}):", 1)
        others = [b for b in a.get("bands", []) if b["z"] != a["z"] and b["p90"] >= 1]
        if others:
            s += "\n   " + " · ".join(f"{alt_text(b['z'])}: mediana {num(b['p50'], phen)}, "
                                      f"1 em 10 acima de {num(b['p90'], phen)} {u}" for b in others)
    det = a.get("det") or {}
    if det:
        s += "\n   modelos: " + " · ".join(f"{MODEL_SHORT.get(m, m)} {num(x['v'], phen)}"
                                         for m, x in sorted(det.items(), key=lambda kv: -kv[1]["v"]
                                                             if phen != "frio" else kv[1]["v"]))
        s += f" {u}"
    return s


def _official_line(ev: dict, official: list[dict]) -> str:
    """IPMA warnings in force for the same phenomenon and days."""
    t = C.IPMA_TYPE.get(ev["phen"])
    days = set(ev["days"])
    hits = [w for w in official if w["type"] == t and w["start"][:10] <= max(days)
            and w["end"][:10] >= min(days)]
    if not hits:
        return "IPMA: ainda sem aviso laranja ou vermelho para este fenómeno."
    by_lvl: dict[str, list[str]] = {}
    for w in hits:
        by_lvl.setdefault(w["level"], []).append(w["area"])
    ac = by_code()
    parts = []
    for lvl in ("red", "orange"):
        if lvl in by_lvl:
            names = sorted({ac[c].name if c in ac else c for c in by_lvl[lvl]})
            parts.append(f"{'vermelho' if lvl == 'red' else 'laranja'} em {', '.join(names)}")
    return "IPMA: aviso " + "; ".join(parts) + "."


def event_text(ev: dict, kind: str, official: list[dict], why: list[str] | None = None) -> str:
    phen = ev["phen"]
    P = C.PHENOMENA[phen]
    level = ev.get("level") or "laranja"
    conf = [(c, a) for c, a in ev["areas"].items() if a["status"] in ("laranja", "vermelho")]
    watch = [(c, a) for c, a in ev["areas"].items() if a["status"] == "vigilancia"]
    groups = {by_code()[c].group for c, _ in conf + watch}
    head = {"alerta": "ALERTA", "atualizacao": "ATUALIZAÇÃO", "cancelamento": "CANCELADO",
            "fim": "TERMINOU"}[kind]
    if kind == "cancelamento":
        return (f"✅ <b>{head} · {esc(P['name'])}</b>\n{fmt_days(ev['days'], True)}\n\n"
                f"A run {fmt_run(ev['last_run'])} do ensemble do ECMWF já não mostra este evento "
                f"(nenhuma área com probabilidade ≥ 20 % de chegar ao laranja).")
    if kind == "fim":
        return (f"🏁 <b>{head} · {esc(P['name'])}</b> ({fmt_days(ev['days'])})\n"
                f"O período do alerta acabou.")
    lvl_name = "VERMELHO" if level == "vermelho" else "LARANJA"
    lines = [f"{P['icon']} <b>{head} {lvl_name} · {esc(P['name'])}</b>",
             f"<b>{fmt_days(ev['days'], True)}</b> · {where(ev)}"]
    if why and kind == "atualizacao":
        lines.append("<i>" + esc("; ".join(why)) + "</i>")
    lines += ["", f"<b>Probabilidade de chegar ao laranja</b> (ensemble ECMWF, 50 membros):"]
    lines += [_area_line(phen, c, a) for c, a in conf]
    if watch:
        lines.append("Em vigilância: " + ", ".join(f"{esc(by_code()[c].name)} {pct(a['p_o'])}"
                                                   for c, a in watch[:8]))
    lines += ["", f"<b>Valores previstos</b> ({value_what(phen)} na área):"]
    lines += [_values_line(phen, c, a) for c, a in conf[:3]]
    if phen == "neve":
        lines.append("<i>Neve estimada à altitude de cada faixa a partir da precipitação e da temperatura "
                     "(2 m e 850 hPa) dos modelos: a grelha de 0,25° não chega às serras.</i>")
    c0, a0 = conf[0] if conf else (None, None)
    lines += ["", "<b>Porquê esta certeza</b>"]
    if a0 is not None:
        prev = a0.get("prev_p_o")
        nd = len(a0.get("det", {}))
        sup = sum(1 for x in a0.get("det", {}).values() if x.get("near"))
        dtxt = (f"{sup} de {nd} modelos determinísticos perto ou acima do limiar." if nd
                else "Sem modelos determinísticos para comparar (exigida probabilidade maior).")
        lines.append(f"Run {fmt_run(ev['last_run'])}"
                     + (f"; na run anterior {pct(prev)} em {esc(by_code()[c0].name)}" if prev is not None else "")
                     + ". " + dtxt)
    lines += ["", _official_line(ev, official),
              f"Critério IPMA: {criterion_text(phen, groups)}."]
    return "\n".join(lines)


def official_text(group: list[dict]) -> str:
    w = group[0]
    lvl = "VERMELHO" if w["level"] == "red" else "LARANJA"
    icon = "🟥" if w["level"] == "red" else "🟧"
    ac = by_code()
    names = sorted({ac[x["area"]].name if x["area"] in ac else x["area"] for x in group})
    s0, s1 = dt.datetime.fromisoformat(w["start"]), dt.datetime.fromisoformat(w["end"])
    span = (f"{fmt_day(s0.date())} {s0:%H:%M} – {s1:%H:%M}" if s0.date() == s1.date()
            else f"{fmt_day(s0.date())} {s0:%H:%M} – {fmt_day(s1.date())} {s1:%H:%M}")
    txt = (f"{icon} <b>IPMA: aviso {lvl} · {esc(w['type'])}</b>\n{esc(', '.join(names))}\n{span}")
    if w.get("text"):
        txt += f"\n«{esc(w['text'])}»"
    return txt


def status_text(st: dict) -> str:
    """Answer to /estado."""
    act = [e for e in st["events"] if e["status"] in ("confirmado", "vigilancia")]
    lines = ["<b>Estado dos avisos</b>"]
    last = st["runs"][-1] if st["runs"] else None
    if last:
        lines.append(f"Última análise: run {fmt_run(last)} do ensemble do ECMWF.")
    if not act:
        lines.append("Sem eventos confirmados nem em vigilância.")
    for e in sorted(act, key=lambda e: (e["status"] != "confirmado", e["days"][0])):
        P = C.PHENOMENA[e["phen"]]
        tag = ("🟥 confirmado (vermelho)" if e.get("level") == "vermelho" else
               "🟧 confirmado" if e["status"] == "confirmado" else "▫️ em vigilância")
        top = ", ".join(f"{by_code()[c].name} {pct(a['p_o'])}" for c, a in list(e["areas"].items())[:4])
        lines.append(f"\n{P['icon']} <b>{esc(P['name'])}</b> · {fmt_days(e['days'])}\n{tag} — {esc(top)}")
    off = [w for w in st["ipma"].get("active", []) if w["level"] in ("orange", "red")]
    if off:
        lines.append("\n<b>IPMA em vigor (laranja/vermelho)</b>")
        seen = set()
        for w in off:
            k = (w["type"], w["level"], w["start"], w["end"])
            if k in seen:
                continue
            seen.add(k)
            grp = [x for x in off if (x["type"], x["level"], x["start"], x["end"]) == k]
            lines.append(official_text(grp).split("\n«")[0])
    return "\n".join(lines)
