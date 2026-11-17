"""Figuras de satélite da depressão Kristin (27–28/01/2026): EUMETSAT EUMETView (MSG massa de ar,
MTG-I1 infravermelho e vapor de água) com a pressão ao nível do mar da ERA5 e o centro seguido pela app.

Corre a partir de backend/:  ../.venv/bin/python ../docs/casos/kristin/make_sat_kristin.py
"""
import asyncio
import datetime as dt
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "backend"))
spec = importlib.util.spec_from_file_location("sr", HERE.parent / "bruna-charles" / "make_sat_radar.py")
src = (HERE.parent / "bruna-charles" / "make_sat_radar.py").read_text().replace("\nasyncio.run(main())\n", "\n").replace(
    "análise ECMWF IFS (arquivo Forecast Workstation)", "reanálise ERA5 (ECMWF/C3S) · Forecast Workstation")
sr = type(sys)("sr")
sr.__file__ = str(HERE.parent / "bruna-charles" / "make_sat_radar.py")
exec(compile(src, sr.__file__, "exec"), sr.__dict__)
sr.OUT = HERE / "figuras"

from app import archive  # noqa: E402
from app.cases import _use, all_tracks  # noqa: E402

# the ERA5 analyses are the reference for the isobars here
from app.models import MODELS  # noqa: E402
import app.cases as cases  # noqa: E402

_orig = cases.analysis_field


async def _era5_field(model, t, key):
    return await _orig(MODELS["era5"], t, key)


sr.analysis_field = _era5_field


async def main():
    ev = archive.get_event("depresso-kristin")
    _use(ev)
    tr = await all_tracks(ev, ["era5"])
    tracks = {k: v.get("era5") or [] for k, v in tr.items()}
    T = dt.datetime
    box = (-24, 4, 33, 50)
    await sr.satellite("k2_massa_ar_2721.png", "msg_fes:rgb_airmass", "", T(2026, 1, 27, 21), box,
                       "Kristin a formar-se — 27/01/2026, 21:00 UTC (massa de ar)",
                       "Compósito RGB «massa de ar» (MSG/SEVIRI) · isóbaras da ERA5 (4 em 4 hPa) · ✕ = centro seguido pela app",
                       tracks, "Massa de ar: vermelho/laranja = ar seco de origem estratosférica (PV elevada); verde = ar "
                       "quente e húmido; azul/violeta = ar frio; branco/ocre = nuvens altas e espessas.")
    await sr.satellite("k3_massa_ar_2803.png", "msg_fes:rgb_airmass", "", T(2026, 1, 28, 3), box,
                       "Kristin a chegar a Portugal — 28/01/2026, 03:00 UTC (massa de ar)",
                       "Compósito RGB «massa de ar» (MSG/SEVIRI) · isóbaras da ERA5 (4 em 4 hPa) · ✕ = centro seguido pela app",
                       tracks, "A cabeça de nuvens enrola-se à volta do centro; a intrusão seca (vermelho) entra por oeste "
                       "e sul — o padrão em que se forma um sting jet na ponta da cabeça de nuvens.")
    await sr.satellite("k4_ir_2804.png", "mtg_fd:ir105_hrfi", "mtg_fd:mtg_fd_ir105_hrfi_style_01", T(2026, 1, 28, 4),
                       (-14, -2, 36, 44), "Kristin sobre Portugal — 28/01/2026, 04:00 UTC (infravermelho)",
                       "Infravermelho 10,5 µm realçado (MTG-I1/FCI, 1 km) · isóbaras da ERA5 · ✕ = centro seguido pela app",
                       tracks, "Infravermelho realçado: azul → violeta → rosa = topos de nuvem cada vez mais frios e altos; "
                       "cinzento = nuvens baixas e médias. A cabeça de nuvens enrola-se a norte do centro, com a ponta em gancho junto à "
                       "costa centro; a sul e oeste, a intrusão seca deixa só nuvens baixas.")


asyncio.run(main())
