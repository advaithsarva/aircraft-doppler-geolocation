"""Sample raster DEMs with explicit vertical datum and coordinate conversion."""

import numpy as np
from pyproj import Transformer


class RasterTerrain:
    def __init__(self, path, frame, *, datum, geoid_offset_m=None):
        import rasterio

        if datum not in {"ellipsoid", "orthometric"}:
            raise ValueError("DEM datum must be explicit")
        if datum == "orthometric" and geoid_offset_m is None:
            raise ValueError("Orthometric DEM requires geoid offset (ellipsoid minus MSL)")
        self.dataset = rasterio.open(path)
        if self.dataset.crs is None:
            self.dataset.close()
            raise ValueError("DEM needs a CRS")
        self.frame = frame
        self.offset = geoid_offset_m or 0.0
        self.transformer = Transformer.from_crs(4326, self.dataset.crs, always_xy=True)

    def __call__(self, x, y):
        lat, lon, _ = self.frame.to_geodetic([x, y, 0])
        rx, ry = self.transformer.transform(lon, lat)
        if not (
            self.dataset.bounds.left <= rx < self.dataset.bounds.right
            and self.dataset.bounds.bottom < ry <= self.dataset.bounds.top
        ):
            raise ValueError("Candidate outside DEM coverage")
        value = next(self.dataset.sample([(rx, ry)], masked=True))[0]
        if np.ma.is_masked(value) or not np.isfinite(value):
            raise ValueError("Candidate intersects DEM nodata")
        return float(self.frame.to_enu(lat, lon, float(value) + self.offset)[2])

    def close(self):
        self.dataset.close()
