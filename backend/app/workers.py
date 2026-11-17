"""Functions executed in the rendering process pool (CPU-bound work)."""
from __future__ import annotations


def render_map_worker(job: dict):
    from .maps.render import render_job
    return render_job(job)


def render_sounding_worker(snd: dict, title: str, subtitle: str, parcel: str):
    from .sounding.analysis import analyse
    from .sounding.skewt import render_skewt
    an = analyse(snd["profile"])
    png = render_skewt(snd, an, title, subtitle, parcel)
    data = {"lat": snd["lat"], "lon": snd["lon"], "surface": snd["surface"],
            "profile": snd["profile"], "parcels": an["parcels"], "indices": an["indices"],
            "kinematics": an["kinematics"], "model_diag": snd.get("model_diag", {})}
    return png, data


def render_meteogram_worker(data: dict, place: str):
    from .meteogram import render_meteogram
    return render_meteogram(data, place)


def render_multi_worker(datas: list[dict], place: str):
    from .meteogram import render_multi
    return render_multi(datas, place)


def compare_sounding_worker(items: list[dict], title: str, subtitle: str):
    from .sounding.analysis import analyse
    from .sounding.compare import render_skewt_multi
    for it in items:
        it["an"] = analyse(it["snd"]["profile"])
    return render_skewt_multi(items, title, subtitle)


def render_thermogram_worker(data: dict, place: str):
    from .thermogram import render_thermogram
    return render_thermogram(data, place)


def render_case_worker(kind: str, data: dict):
    from . import cases_render
    if kind == "sounding":
        from .sounding.analysis import analyse
        from .sounding.compare import render_skewt_multi
        for it in data["items"]:
            it["an"] = analyse(it["snd"]["profile"])
        return render_skewt_multi(data["items"], data["title"], data["subtitle"])
    return {"pressure": cases_render.render_pressure, "sequence": cases_render.render_sequence,
            "point": cases_render.render_point,
            "meteogram": cases_render.render_case_meteogram,
            "atcf_tracks": cases_render.render_atcf_tracks,
            "atcf_landfall": cases_render.render_atcf_landfall}[kind](data)


def render_xsection_worker(sec: dict, meta: dict):
    from .xsection import render
    return render(sec, meta)


def render_xsection_multi_worker(items: list, meta: dict):
    from .xsection import render_multi
    return render_multi(items, meta)
