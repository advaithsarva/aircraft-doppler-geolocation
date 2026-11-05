"""WGS84 geodetic/ellipsoidal height <-> Earth centered local ENU."""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from pyproj import Transformer


@dataclass(frozen=True)
class LocalFrame:
    latitude: float
    longitude: float
    height_m: float = 0.0

    def __post_init__(self):
        if not np.isfinite(self.height_m):
            raise ValueError("Frame height must be finite")
        if not (-90 <= self.latitude <= 90 and -180 <= self.longitude <= 180):
            raise ValueError("Invalid frame coordinates")

    @cached_property
    def _setup(self):
        transformer = Transformer.from_crs(4979, 4978, always_xy=True)
        origin = np.array(transformer.transform(self.longitude, self.latitude, self.height_m))
        p, longitude_radians = np.radians([self.latitude, self.longitude])
        rotation = np.array(
            [
                [-np.sin(longitude_radians), np.cos(longitude_radians), 0],
                [
                    -np.sin(p) * np.cos(longitude_radians),
                    -np.sin(p) * np.sin(longitude_radians),
                    np.cos(p),
                ],
                [
                    np.cos(p) * np.cos(longitude_radians),
                    np.cos(p) * np.sin(longitude_radians),
                    np.sin(p),
                ],
            ]
        )
        return transformer, origin, rotation

    def to_enu(self, latitudes, longitudes, heights):
        transformer, origin, rotation = self._setup
        xyz = np.stack(transformer.transform(longitudes, latitudes, heights), axis=-1)
        return (xyz - origin) @ rotation.T

    def to_geodetic(self, enu):
        _, origin, rotation = self._setup
        xyz = np.asarray(enu) @ rotation + origin
        lon, lat, h = Transformer.from_crs(4978, 4979, always_xy=True).transform(
            xyz[..., 0], xyz[..., 1], xyz[..., 2]
        )
        return np.stack([lat, lon, h], axis=-1)
