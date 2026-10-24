"""H3 hexagon grid over the study area, with a district label per cell."""

from __future__ import annotations

import h3
import pandas as pd
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

from parali import config


def _polygons(geom):
    return list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]


def cells_for_area(area, res: int = config.H3_RES) -> list[str]:
    """All H3 cells whose centre falls inside the area polygon(s)."""
    cells: set[str] = set()
    for poly in _polygons(area):
        outer = [(y, x) for x, y in poly.exterior.coords]
        holes = [[(y, x) for x, y in ring.coords] for ring in poly.interiors]
        cells.update(h3.polygon_to_cells(h3.LatLngPoly(outer, *holes), res))
    return sorted(cells)


def build_grid(districts: dict, res: int = config.H3_RES) -> pd.DataFrame:
    """One row per cell: id, centre lat/lon, district, state."""
    geoms = [shape(f["geometry"]) for f in districts["features"]]
    props = [f["properties"] for f in districts["features"]]
    tree = STRtree(geoms)

    from parali.pipeline.boundaries import study_area

    rows = []
    for cell in cells_for_area(study_area(districts), res):
        lat, lon = h3.cell_to_latlng(cell)
        pt = Point(lon, lat)
        district = state = None
        for idx in tree.query(pt):
            if geoms[idx].contains(pt):
                district, state = props[idx]["district"], props[idx]["state"]
                break
        if district is None:  # centre sits on a border sliver: use the nearest district
            idx = tree.nearest(pt)
            district, state = props[idx]["district"], props[idx]["state"]
        rows.append({"cell": cell, "lat": lat, "lon": lon, "district": district, "state": state})
    return pd.DataFrame(rows)


def neighbours(cell: str, k: int) -> list[str]:
    """Cells exactly k rings away (excludes the cell itself for k >= 1)."""
    return [c for c in h3.grid_ring(cell, k)] if k > 0 else [cell]
