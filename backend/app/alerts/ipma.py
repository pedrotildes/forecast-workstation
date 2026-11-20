"""Official IPMA warnings (open data): relayed when they reach the configured levels."""
from __future__ import annotations

import httpx

from ..config import USER_AGENT

URL = "https://api.ipma.pt/open-data/forecast/warnings/warnings_www.json"


async def fetch() -> list[dict]:
    """Warnings above green: [{"area", "type", "level", "start", "end", "text"}]."""
    async with httpx.AsyncClient(timeout=30, headers={"User-Agent": USER_AGENT}) as c:
        r = await c.get(URL)
        r.raise_for_status()
        data = r.json()
    out = []
    for w in data:
        if w.get("awarenessLevelID") in (None, "green"):
            continue
        out.append({"area": w.get("idAreaAviso", ""), "type": w.get("awarenessTypeName", ""),
                    "level": w["awarenessLevelID"], "start": w.get("startTime", ""),
                    "end": w.get("endTime", ""), "text": (w.get("text") or "").strip()})
    return out


def key(w: dict) -> str:
    # a later change of the end time is not news; a new start or a new level is
    return f"{w['area']}|{w['type']}|{w['level']}|{w['start'][:13]}"


def new_groups(st: dict, warnings: list[dict], levels: list[str]) -> list[list[dict]]:
    """Warnings not relayed yet, grouped as IPMA writes them (same type, level, period, text)."""
    seen = set(st["ipma"].get("seen", []))
    fresh = [w for w in warnings if w["level"] in levels and key(w) not in seen]
    groups: dict[tuple, list[dict]] = {}
    for w in fresh:
        groups.setdefault((w["type"], w["level"], w["start"], w["end"], w["text"]), []).append(w)
    order = {"red": 0, "orange": 1, "yellow": 2}
    return [g for _, g in sorted(groups.items(), key=lambda kv: (order.get(kv[0][1], 9), kv[0][2]))]


def mark_seen(st: dict, group: list[dict]) -> None:
    seen = st["ipma"].setdefault("seen", [])
    seen += [key(w) for w in group]
    st["ipma"]["seen"] = seen[-2000:]
