"""Sentinel-2 L2A reflectance for a small area, read straight from the public COGs on AWS
(Element 84 Earth Search STAC; no account needed).

Bands match NASA HLS / Prithvi order: Blue, Green, Red, NIR narrow (B8A), SWIR1 (B11), SWIR2 (B12).
Everything is resampled to one 20 m UTM grid so pre- and post-fire images line up pixel for pixel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STAC_URL = "https://earth-search.aws.element84.com/v1"
BANDS = ["blue", "green", "red", "nir08", "swir16", "swir22"]
SCL_BAD = {0, 1, 3, 8, 9, 10, 11}  # no data, saturated, cloud shadow, clouds, cirrus, snow
UTM43 = "EPSG:32643"
RES = 20.0


@dataclass
class Grid:
    crs: str
    transform: object  # affine.Affine
    width: int
    height: int

    @classmethod
    def for_bbox(cls, bbox: tuple[float, float, float, float], crs: str = UTM43, res: float = RES) -> Grid:
        from rasterio.transform import from_origin
        from rasterio.warp import transform_bounds

        left, bottom, right, top = transform_bounds("EPSG:4326", crs, *bbox)
        width, height = int((right - left) // res), int((top - bottom) // res)
        return cls(crs, from_origin(left, top, res, res), width, height)

    def lonlat_to_rowcol(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from rasterio.warp import transform

        xs, ys = transform("EPSG:4326", self.crs, list(lon), list(lat))
        inv = ~self.transform
        cols, rows = inv * (np.asarray(xs), np.asarray(ys))
        return np.floor(rows).astype(int), np.floor(cols).astype(int)


def search(bbox, start: str, end: str, max_cloud: float = 10.0, max_nodata: float = 5.0) -> list:
    """Scenes over the area, clearest first per day, keeping only ones that cover the tile fully."""
    from pystac_client import Client

    items = (
        Client.open(STAC_URL)
        .search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{start}/{end}",
            query={"eo:cloud_cover": {"lt": max_cloud}},
        )
        .items()
    )
    keep = [i for i in items if (i.properties.get("s2:nodata_pixel_percentage") or 0) <= max_nodata]
    return sorted(keep, key=lambda i: i.datetime)


def _read(href: str, grid: Grid, resampling) -> np.ndarray:
    import rasterio
    from rasterio.vrt import WarpedVRT

    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES"):
        with (
            rasterio.open(href) as src,
            WarpedVRT(
                src,
                crs=grid.crs,
                transform=grid.transform,
                width=grid.width,
                height=grid.height,
                resampling=resampling,
            ) as vrt,
        ):
            return vrt.read(1)


def read_scene(item, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """(6, H, W) reflectance in 0..1 and a (H, W) mask of usable pixels."""
    from rasterio.enums import Resampling

    # Since processing baseline 04.00 (Jan 2022) L2A digital numbers carry a +1000 offset; remove it
    # unless the catalogue says it was already removed, so reflectance matches HLS / Prithvi inputs.
    offset = 0.0
    if not item.properties.get("earthsearch:boa_offset_applied", False):
        baseline = float(item.properties.get("s2:processing_baseline", "0") or 0)
        offset = 1000.0 if baseline >= 4.0 else 0.0
    bands = [
        np.clip(
            (_read(item.assets[b].href, grid, Resampling.average).astype(np.float32) - offset) / 10000.0, 0, 1
        )
        for b in BANDS
    ]
    scl = _read(item.assets["scl"].href, grid, Resampling.nearest)
    ok = ~np.isin(scl, list(SCL_BAD)) & (bands[0] > 0)
    return np.stack(bands), ok


def nbr(img: np.ndarray) -> np.ndarray:
    """Normalised burn ratio from B8A and B12."""
    nir, swir2 = img[3], img[5]
    return (nir - swir2) / np.clip(nir + swir2, 1e-4, None)
