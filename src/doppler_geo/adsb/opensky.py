"""Normalize an operator-provided OpenSky state-vector export; no API credentials required."""

import polars as pl


def normalize_export(frame: pl.DataFrame, *, altitude_field: str) -> pl.DataFrame:
    """OpenSky-compatible column names; SI meters/seconds, epoch UTC time.

    geoaltitude/geo_altitude requires provider datum verification before ingestion.
    A callsign alone is not a globally unique flight ID. Segment ID here is ICAO24 + UTC day;
    downstream ingestion additionally splits data gaps. Operators may provide flight_id.
    """
    if altitude_field not in {"geoaltitude", "geo_altitude", "baroaltitude", "baro_altitude"}:
        raise ValueError("Choose an explicit OpenSky altitude column")
    required = {"icao24", "time", "lat", "lon", altitude_field}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing OpenSky columns: {required - set(frame.columns)}")
    result = frame.with_columns(
        pl.from_epoch(pl.col("time"), time_unit="s").dt.replace_time_zone("UTC").alias("utc")
    )
    result = result.with_columns(
        pl.col("icao24").alias("aircraft_id"),
        pl.col("lat").alias("latitude"),
        pl.col("lon").alias("longitude"),
        pl.col(altitude_field).alias("altitude_m"),
        pl.col("utc").dt.strftime("%Y-%m-%dT%H:%M:%SZ").alias("timestamp"),
    )
    if "flight_id" not in result.columns:
        result = result.with_columns(
            pl.concat_str(
                [pl.col("icao24"), pl.col("utc").dt.strftime("%Y-%m-%d")], separator="/"
            ).alias("flight_id")
        )
    return result.drop("utc")
