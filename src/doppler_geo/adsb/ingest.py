"""Explicit canonical schema, UTC normalization, ellipsoidal altitude only."""

from pathlib import Path

import duckdb
import numpy as np
import polars as pl
from scipy.spatial import cKDTree

from doppler_geo.physics.model import Trajectory
from doppler_geo.validation import finite_scalar, positive_integer, spatial_bbox, utc_datetime

REQUIRED = {"flight_id", "aircraft_id", "timestamp", "latitude", "longitude", "altitude_m"}


def load_tracks(
    path,
    frame,
    bbox,
    start_utc,
    end_utc,
    *,
    altitude_datum,
    time_origin_utc,
    altitude_offset_m=None,
    max_gap_s=30.0,
    max_rows=1_000_000,
):
    """Filter lazily before collecting; input times must include UTC offset or Z.

    bbox=(west,south,east,north). Operator must convert barometric/orthometric heights
    with an explicit, documented offset or pre-process with a proper geoid model.
    Optional metadata columns remain nullable and are never used to infer type.
    """
    positive_integer(max_rows, "max_rows")
    finite_scalar(max_gap_s, "max_gap_s", positive=True)
    if altitude_offset_m is not None:
        finite_scalar(altitude_offset_m, "altitude_offset_m")
    if altitude_datum not in {"ellipsoid", "orthometric", "barometric"}:
        raise ValueError("Specify altitude datum")
    if altitude_datum != "ellipsoid" and altitude_offset_m is None:
        raise ValueError("A documented altitude conversion is required")
    path = Path(path)
    if path.suffix == ".parquet":
        lazy = pl.scan_parquet(path)
    elif path.suffix == ".csv":
        lazy = pl.scan_csv(path, schema_overrides={"timestamp": pl.String})
    else:
        raise ValueError("Use canonical CSV or Parquet")
    if not REQUIRED.issubset(lazy.collect_schema().names()):
        raise ValueError(f"Required columns: {sorted(REQUIRED)}")
    west, south, east, north = spatial_bbox(bbox)
    start, end, origin = [utc_datetime(v) for v in (start_utc, end_utc, time_origin_utc)]
    if end <= start:
        raise ValueError("Invalid UTC window")
    filtered_lazy = lazy.filter(
        pl.col("longitude").is_between(west, east) & pl.col("latitude").is_between(south, north)
    )
    # Require string canonical timestamps even in Parquet to avoid implicit timezone assumptions.
    invalid_timezone = (
        filtered_lazy.filter(~pl.col("timestamp").str.contains(r"(Z|[+-]\d{2}:\d{2})$"))
        .limit(1)
        .collect()
    )
    if invalid_timezone.height:
        raise ValueError("Input timestamps must include UTC offset or Z")
    filtered_lazy = filtered_lazy.with_columns(
        pl.col("timestamp").str.to_datetime(time_zone="UTC", strict=True)
    )
    if max_rows < 1 or max_rows > 1_000_000:
        raise ValueError("Row budget must be between 1 and 1000000")
    filtered = (
        filtered_lazy.filter(pl.col("timestamp").is_between(start, end))
        .limit(max_rows + 1)
        .collect()
    )
    if filtered.height > max_rows:
        raise ValueError("Row budget exceeded; partition data or narrow the UTC/spatial window")
    initial = filtered.height
    filtered = filtered.drop_nulls(list(REQUIRED)).filter(
        pl.all_horizontal([pl.col(c).is_finite() for c in ["latitude", "longitude", "altitude_m"]])
    )
    # Conflicting simultaneous samples are excluded, not averaged into fictitious positions.
    filtered = filtered.unique(subset=sorted(REQUIRED))
    filtered = filtered.with_columns(pl.len().over(["flight_id", "timestamp"]).alias("n"))
    filtered = filtered.filter(pl.col("n") == 1).sort(["flight_id", "timestamp"])
    tracks = []
    for group in filtered.partition_by("flight_id", maintain_order=True):
        stamps = np.array([(t - origin).total_seconds() for t in group["timestamp"].to_list()])
        # Split at gaps or aircraft identifier changes, never interpolate across them.
        ids = group["aircraft_id"].to_numpy()
        cuts = np.flatnonzero((np.diff(stamps) > max_gap_s) | (ids[1:] != ids[:-1])) + 1
        positions = frame.to_enu(
            group["latitude"].to_numpy(),
            group["longitude"].to_numpy(),
            group["altitude_m"].to_numpy() + (altitude_offset_m or 0),
        )
        for number, indices in enumerate(np.split(np.arange(len(stamps)), cuts), 1):
            if len(indices) >= 3:
                tracks.append(
                    Trajectory(
                        stamps[indices],
                        positions[indices],
                        str(group["flight_id"][0]),
                        max_gap_s,
                        f"{group['flight_id'][0]}/segment-{number}",
                    )
                )
    return tracks, {
        "rows_in_window": initial,
        "clean_rows": filtered.height,
        "segments": len(tracks),
        "available_segments": [
            {
                "flight_id": f.flight_id,
                "segment_id": f.segment_id,
                "start_s": float(f.times[0]),
                "end_s": float(f.times[-1]),
            }
            for f in tracks
        ],
        "altitude_datum": altitude_datum,
        "altitude_offset_m": altitude_offset_m,
        "warning": "ADS-B coverage is incomplete; aircraft type may be unknown.",
    }


def parquet_window(path, bbox, start_utc, end_utc, limit=1_000_000):
    """Bounded DuckDB query; identifiers are fixed, all user values parameterized."""
    west, south, east, north = spatial_bbox(bbox)
    positive_integer(limit, "limit")
    start, end = utc_datetime(start_utc), utc_datetime(end_utc)
    if end <= start:
        raise ValueError("Invalid UTC window")
    if limit <= 0 or limit > 1_000_000:
        raise ValueError("Invalid row limit")
    with duckdb.connect() as db:
        db.execute("SET memory_limit='512MB'")
        return pl.from_arrow(
            db.sql(
                """SELECT * FROM read_parquet(?) WHERE longitude BETWEEN ? AND ?
          AND latitude BETWEEN ? AND ? AND CAST(timestamp AS TIMESTAMPTZ)
          BETWEEN CAST(? AS TIMESTAMPTZ) AND CAST(? AS TIMESTAMPTZ) LIMIT ?""",
                params=[str(path), west, east, south, north, start, end, limit],
            ).to_arrow_table()
        )


class FlightIndex:
    """Conservative bounding-sphere KD-tree; no false rejection from sparse samples."""

    def __init__(self, flights):
        self.flights = list(flights)
        if not self.flights:
            raise ValueError("No flights to index")
        lower = np.array([f.positions.min(axis=0) for f in self.flights])
        upper = np.array([f.positions.max(axis=0) for f in self.flights])
        self.centers = (lower + upper) / 2
        self.radii = np.linalg.norm(upper - lower, axis=1) / 2
        self.tree = cKDTree(self.centers)

    def within_range(self, observer, distance_m):
        if not np.isfinite(distance_m) or distance_m <= 0:
            raise ValueError("Distance must be positive")
        indices = self.tree.query_ball_point(observer, distance_m + self.radii.max())
        return [
            self.flights[i]
            for i in indices
            if np.linalg.norm(self.centers[i] - observer) <= distance_m + self.radii[i]
        ]
