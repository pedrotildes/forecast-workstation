"""Near-real-time observations: EUMETSAT satellite (EUMETView WMS) and radar (RainViewer).

The browser requests imagery directly from the providers; the backend only resolves
which layers exist and their latest available time (from the WMS capabilities, which
are too large and lack CORS headers for the browser to fetch itself).
"""
from __future__ import annotations

import re
import time

import httpx

from .config import USER_AGENT

WMS = "https://view.eumetsat.int/geoserver/ows"
RAINVIEWER = "https://api.rainviewer.com/public/weather-maps.json"

# (id, wms layer, style, name, group)
SAT_LAYERS = [
    ("ir105_c1", "mtg_fd:ir105_hrfi", "mtg_fd:mtg_fd_ir105_hrfi_style_01",
     "Infravermelho 10,5 µm — realçado (MTG, 1 km)", "Canais"),
    ("ir105_c2", "mtg_fd:ir105_hrfi", "mtg_fd:mtg_fd_ir105_hrfi_style_02",
     "Infravermelho 10,5 µm — topos frios (MTG)", "Canais"),
    ("ir105", "mtg_fd:ir105_hrfi", "mtg_fd:mtg_fd_ir105_hrfi_grayscale",
     "Infravermelho 10,5 µm — cinzentos (MTG)", "Canais"),
    ("vis06", "mtg_fd:vis06_hrfi", "", "Visível 0,6 µm (MTG, 500 m)", "Canais"),
    ("wv062", "msg_fes:wv062", "", "Vapor de água 6,2 µm (MSG)", "Canais"),
    ("geocolour", "mtg_fd:rgb_geocolour", "", "GeoColour — dia e noite (MTG)", "Compósitos RGB"),
    ("truecolour", "mtg_fd:rgb_truecolour", "", "Cor real (MTG)", "Compósitos RGB"),
    ("natural", "msg_fes:rgb_naturalenhncd", "", "Cor natural realçada (MSG)", "Compósitos RGB"),
    ("airmass", "msg_fes:rgb_airmass", "", "Massa de ar (MSG)", "Compósitos RGB"),
    ("convection", "msg_fes:rgb_convection", "", "Convecção severa (MSG)", "Compósitos RGB"),
    ("microphys", "msg_fes:rgb_microphysics", "", "Microfísica diurna (MSG)", "Compósitos RGB"),
    ("cloudtype", "mtg_fd:rgb_cloudtype", "", "Tipo de nuvem (MTG)", "Compósitos RGB"),
    ("cloudphase", "mtg_fd:rgb_cloudphase", "", "Fase da nuvem (MTG)", "Compósitos RGB"),
    ("fog", "mtg_fd:rgb_fog", "", "Nevoeiro / nuvens baixas (MTG)", "Compósitos RGB"),
    ("dust", "mtg_fd:rgb_dust", "", "Poeiras (MTG)", "Compósitos RGB"),
    ("cth", "msg_fes:cth", "", "Altura do topo das nuvens (MSG)", "Produtos"),
]
OVERLAYS = [
    ("lightning", "mtg_fd:li_afa", "", "Descargas elétricas — Lightning Imager (MTG)"),
    ("rdt", "msg_fes:rdt", "msg_fes:rdt_red", "Trovoadas em desenvolvimento rápido (RDT)"),
    ("satprecip", "mtg_fd:h40b", "", "Precipitação estimada por satélite (H SAF)"),
]
STATIC = {
    "coast": ("backgrounds:ne_10m_coastline,backgrounds:ne_boundary_lines_land", "whitelines,"),
    "admin1": ("osmgray:ne_10m_admin_1_states_provinces_lines", ""),
}

_cache: dict = {}


def _iso_period_min(p: str) -> int:
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?", p)
    if not m:
        return 15
    return int(m.group(1) or 0) * 60 + int(m.group(2) or 0)


async def _capabilities() -> dict:
    c = _cache.get("caps")
    if c and time.time() - c[0] < 240:
        return c[1]
    async with httpx.AsyncClient(timeout=60, headers={"User-Agent": USER_AGENT}) as cl:
        r = await cl.get(WMS, params={"service": "WMS", "request": "GetCapabilities",
                                       "version": "1.3.0"})
        r.raise_for_status()
        text = r.text
    out = {}
    wanted = {x[1] for x in SAT_LAYERS} | {x[1] for x in OVERLAYS}
    for name in wanted:
        i = text.find(f"<Name>{name}</Name>")
        if i < 0:
            continue
        j = text.find("</Layer>", i)
        m = re.search(r'<Dimension name="time"[^>]*>([^<]*)</Dimension>', text[i:j])
        if not m:
            continue
        last = m.group(1).split(",")[-1].strip()
        parts = last.split("/")
        latest = parts[1] if len(parts) == 3 else parts[0]
        period = _iso_period_min(parts[2]) if len(parts) == 3 else 15
        out[name] = {"latest": latest.replace(".000Z", "Z"), "period": period}
    _cache["caps"] = (time.time(), out)
    return out


async def _radar() -> dict | None:
    c = _cache.get("radar")
    if c and time.time() - c[0] < 60:
        return c[1]
    try:
        async with httpx.AsyncClient(timeout=20, headers={"User-Agent": USER_AGENT}) as cl:
            r = await cl.get(RAINVIEWER)
            r.raise_for_status()
            d = r.json()
        res = {"host": d["host"], "frames": d["radar"]["past"] + d["radar"].get("nowcast", [])}
    except Exception:  # noqa: BLE001
        res = None
    _cache["radar"] = (time.time(), res)
    return res


async def layers() -> dict:
    caps = await _capabilities()
    sat = [{"id": i, "layer": lay, "style": st, "name": n, "group": g, **caps[lay]}
           for i, lay, st, n, g in SAT_LAYERS if lay in caps]
    ovl = [{"id": i, "layer": lay, "style": st, "name": n, **caps[lay]}
           for i, lay, st, n in OVERLAYS if lay in caps]
    return {"wms": WMS, "sat": sat, "overlays": ovl,
            "static": {k: {"layer": v[0], "style": v[1]} for k, v in STATIC.items()},
            "radar": await _radar()}
