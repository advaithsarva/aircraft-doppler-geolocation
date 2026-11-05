"""Small shared validators for public numerical and UTC inputs."""

import datetime as dt
from numbers import Integral, Real

import numpy as np


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def finite_scalar(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f"{name} has an invalid sign")
    return float(value)


def utc_datetime(value):
    if not isinstance(value, str):
        raise ValueError("UTC timestamps must be ISO 8601 strings")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Invalid ISO 8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("Timestamps must be timezone-aware")
    return parsed.astimezone(dt.timezone.utc)


def spatial_bbox(bbox):
    if len(bbox) != 4 or not np.isfinite(bbox).all():
        raise ValueError("A finite west/south/east/north bbox is required")
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("Invalid bbox; split antimeridian regions explicitly")
    return west, south, east, north
