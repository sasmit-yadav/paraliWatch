"""State and district boundaries for Punjab and Haryana (geoBoundaries, open licence)."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from parali import config
from parali.pipeline.download import download


def plain(name: str) -> str:
    """'Haryāna' -> 'Haryana': geoBoundaries names carry diacritics."""
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()


def _load(name: str) -> dict:
    path = download(config.BOUNDARY_URLS[name], config.RAW_DIR / f"ind_{name}.geojson")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_districts() -> dict:
    """Return a GeoJSON FeatureCollection of districts inside the study states.

    geoBoundaries' district layer has no parent-state field, so each district is assigned to the
    state polygon that contains its representative point.
    """
    states = {
        plain(f["properties"]["shapeName"]): shape(f["geometry"])
        for f in _load("adm1")["features"]
        if plain(f["properties"]["shapeName"]) in config.STATES
    }
    missing = set(config.STATES) - set(states)
    if missing:
        raise ValueError(f"States not found in boundary file: {missing}")

    features = []
    for f in _load("adm2")["features"]:
        geom = shape(f["geometry"])
        if not geom.is_valid:
            geom = geom.buffer(0)
        point = geom.representative_point()
        for state_name, state_geom in states.items():
            if state_geom.contains(point):
                features.append(
                    {
                        "type": "Feature",
                        "properties": {"district": plain(f["properties"]["shapeName"]), "state": state_name},
                        "geometry": mapping(geom),
                    }
                )
                break
    return {"type": "FeatureCollection", "features": features}


def study_area(districts: dict):
    """Union of all district polygons: the exact area we model."""
    return unary_union([shape(f["geometry"]) for f in districts["features"]])
