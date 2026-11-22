"""IPMA warning criteria (orange and red levels) used by the detector.

Source: IPMA, «Critérios de Emissão dos Avisos Meteorológicos»
https://www.ipma.pt/opencms/pt/enciclopedia/otempo/sam/index.html?page=criterios.xml
(read 2026-11-20; IPMA announced further revisions — check it from time to time).

IPMA writes integer ranges for observed values ("91 a 130 km/h"); model values are
continuous, so a range "a–b" becomes ≥ a − 0.5 and "> b" becomes ≥ b + 0.5.

Not used (the models cannot give them with confidence, or the data are missing):
1-h rain (convective), thunderstorms, fog and the Azores mean-wind criterion. Snow is
evaluated on the mainland and in the mountains of Madeira (not in the Azores nor on the
Madeira coasts). Heat and cold need ≥ 48 h: here, two consecutive days.
"""
from __future__ import annotations

LEVELS = ("laranja", "vermelho")
LEVEL_NAME = {"laranja": "Laranja", "vermelho": "Vermelho", "amarelo": "Amarelo"}

PHENOMENA = {
    "vento": {"name": "Vento forte", "unit": "km/h", "what": "rajada máxima", "icon": "💨"},
    "chuva": {"name": "Precipitação forte", "unit": "mm/6 h", "what": "precipitação em 6 h", "icon": "🌧️"},
    "calor": {"name": "Tempo quente", "unit": "°C", "what": "temperatura máxima", "icon": "🌡️"},
    "frio": {"name": "Tempo frio", "unit": "°C", "what": "temperatura mínima", "icon": "🥶"},
    "mar": {"name": "Agitação marítima", "unit": "m", "what": "altura significativa das ondas", "icon": "🌊"},
    "neve": {"name": "Neve", "unit": "cm", "what": "neve acumulada em 24 h", "icon": "❄️"},
}
# IPMA "awarenessTypeName" of each phenomenon (official warnings JSON)
IPMA_TYPE = {"vento": "Vento", "chuva": "Precipitação", "calor": "Tempo Quente",
             "frio": "Tempo Frio", "mar": "Agitação Marítima", "neve": "Neve"}

# ---------------------------------------------------------------- wind (max gust, km/h)
GUST = {"laranja": 90.5, "vermelho": 130.5}
GUST_HIGH = {"laranja": 110.5, "vermelho": 130.5}       # "nas terras altas" (mainland, Madeira)
GUST_AZORES = {"laranja": 100.5, "vermelho": 130.5}
# IPMA does not define "terras altas". The 0.25° model orography is smooth (Serra da Estrela
# shows as ~790 m), so ≥ 700 m here picks the high plateaus of Trás-os-Montes and the Beiras.
HIGHLAND_M = 700.0

# ---------------------------------------------------------------- rain (mm in 6 h)
RAIN6 = {"laranja": 40.5, "vermelho": 60.5}             # same on the mainland and the islands

# ---------------------------------------------------------------- waves (Hs, m)
WAVES = {"laranja": 5.0, "vermelho": 7.0}
WAVES_AZORES = {"laranja": 7.0, "vermelho": 9.0}

# ---------------------------------------------------------------- snow (cm accumulated)
# «Queda de neve com acumulação» by altitude band (same table for the mainland, Madeira and
# the Azores): below 1000 m orange 6–15, red > 15; 1000–1600 m orange 11–30, red > 30;
# above 1600 m orange 26–50, red > 50. IPMA gives no period: here, 24 h (snow.py).
SNOW_BANDS = (  # (lower altitude, upper altitude, thresholds)
    (0.0, 1000.0, {"laranja": 5.5, "vermelho": 15.5}),
    (1000.0, 1600.0, {"laranja": 10.5, "vermelho": 30.5}),
    (1600.0, 9000.0, {"laranja": 25.5, "vermelho": 50.5}),
)


def snow(alt: float) -> dict:
    """Orange/red snow thresholds (cm) at an altitude (m); a band boundary belongs to the lower band."""
    for lo, hi, t in SNOW_BANDS:
        if alt <= hi:
            return t
    return SNOW_BANDS[-1][2]


# ---------------------------------------------------------------- heat / cold (°C)
# area: (Tmin orange upper bound, Tmin red bound, Tmax orange lower bound, Tmax red bound)
# e.g. Lisboa Tmin orange "0 a -1", red "< -1"; Tmax orange "38–40", red "> 40".
_TEMP = {
    "VCT": (-4, -5, 37, 38), "BRG": (-4, -5, 37, 38), "PTO": (-2, -3, 37, 38),
    "VRL": (-4, -5, 38, 39), "BGC": (-8, -10, 37, 38), "VIS": (-5, -7, 39, 40),
    "GDA": (-7, -10, 34, 36), "CBO": (-3, -4, 40, 41), "AVR": (0, -1, 36, 38),
    "CBR": (-4, -5, 39, 40), "LRA": (-4, -6, 36, 40), "LSB": (0, -1, 38, 40),
    "STB": (-3, -4, 40, 42), "STM": (-2, -3, 42, 43), "PTG": (-2, -5, 40, 41),
    "EVR": (-2, -4, 41, 43), "BJA": (-2, -4, 41, 43), "FAR": (0, -1, 38, 41),
    "MCN": (6, 5, 27, 30), "MRM": (-2, -4, 28, 29), "MCS": (9, 8, 30, 35), "MPS": (9, 8, 28, 31),
    "AOR": (4, 0, 30, 31), "ACE": (4, 0, 30, 31), "AOC": (4, 0, 30, 31),
}


def heat(code: str) -> dict:
    _, _, o, r = _TEMP[code]
    return {"laranja": o - 0.5, "vermelho": r + 0.5}


def cold(code: str) -> dict:
    o, r, _, _ = _TEMP[code]
    return {"laranja": o + 0.5, "vermelho": r - 0.5}


def gust(code: str, group: str, highland: bool = False) -> dict:
    if group == "acores":
        return GUST_AZORES
    if highland or code == "MRM":
        return GUST_HIGH
    return GUST


def waves(group: str) -> dict:
    return WAVES_AZORES if group == "acores" else WAVES


def thresholds(phen: str, code: str, group: str) -> dict:
    """Orange/red thresholds of an area (for wind: the general, non-highland values)."""
    if phen == "vento":
        return gust(code, group)
    if phen == "chuva":
        return RAIN6
    if phen == "calor":
        return heat(code)
    if phen == "frio":
        return cold(code)
    if phen == "mar":
        return waves(group)
    if phen == "neve":
        return snow(0.0)
    raise KeyError(phen)


def fmt_threshold(phen: str, v: float) -> str:
    """Threshold as IPMA writes it (e.g. 90.5 -> "91 km/h", 37.5 -> "38 °C")."""
    if phen == "frio":
        return f"{v - 0.5:.0f} °C"
    if phen == "calor":
        return f"{v + 0.5:.0f} °C"
    if phen == "mar":
        return f"{v:.0f} m"
    if phen == "neve":
        return f"{v + 0.5:.0f} cm"
    unit = PHENOMENA[phen]["unit"]
    return f"{v + 0.5:.0f} {unit}"
