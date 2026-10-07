from .base import Domain, FieldUnavailable, Key, Model
from .ecmwf import ECMWF_AIFS, ECMWF_IFS
from .gfs import GFS
from .icon import ICON_EU

MODELS: dict[str, Model] = {m.id: m for m in (GFS(), ECMWF_IFS(), ECMWF_AIFS(), ICON_EU())}


def get_model(model_id: str) -> Model:
    try:
        return MODELS[model_id]
    except KeyError:
        raise KeyError(f"modelo desconhecido: {model_id}") from None


__all__ = ["MODELS", "get_model", "Domain", "Key", "Model", "FieldUnavailable"]
