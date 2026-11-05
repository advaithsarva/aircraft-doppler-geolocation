# Dependencies and legitimate data access

Python 3.11+; core: NumPy/SciPy for vectors, interpolation, signal transforms and spatial
indices, pyproj for coordinate reference systems, soundfile for audio, Polars/PyArrow
for columnar local input, DuckDB for bounded Parquet SQL. Optional Plotly/Streamlit
provide diagnostics and the research interface. Optional rasterio reads local DEMs.
FFmpeg is an external executable needed for video input, not required for WAV/FLAC.

Numba, librosa, GeoPandas, shapely and traffic are deliberately not mandatory. The
current CPU workload does not justify their deployment or compilation costs. Numba
should follow profiling; GPU work is deferred. See `benchmarks/` for measured results.
Development dependencies: pytest, Hypothesis, Ruff and mypy. `requirements-tested.txt`
records tested direct dependency versions; it is not a universal platform lockfile.

## ADS-B schema

Required canonical columns:

| Column | Type / convention |
|---|---|
| flight_id | operator-assigned flight/segment label |
| aircraft_id | ICAO24 or explicitly supplied identifier |
| timestamp | ISO 8601 string with Z or timezone offset, including in Parquet |
| latitude / longitude | finite degrees, WGS84 |
| altitude_m | meters; altitude datum supplied separately |

Optional provider columns such as callsign, aircraft_type, groundspeed_mps,
vertical_rate_mps and track_deg are retained during normalization but do not create
confident aircraft classifications. Position-derived velocity is authoritative in the
interpolated model. Altitude missing values are dropped, never replaced with ground level.
Conflicting positions at the same flight/timestamp are excluded; exact duplicates are
removed; changes of identifier and long gaps split tracks. Segments shorter than three
points cannot be interpolated. Ingestion reports the remaining rows and segments.

`adsb.opensky.normalize_export` maps an operator-provided state-vector table. The selected
altitude field is explicit. Its provisional flight label is ICAO24/UTC-day unless provided;
this does not claim two same-day sorties are the same flight. Supply proper flight IDs
or segment by gaps. Verify the provider's geometric height reference; the label
"geometric" alone does not establish a universal ellipsoid datum.

## Historical sources

OpenSky access conditions, coverage and interfaces can change. The official
[historical Trino documentation](https://openskynetwork.github.io/opensky-api/trino)
and [data catalog](https://opensky-network.org/data) are the starting points for eligible
operators; historical access is not promised by installing this package. The
[REST schema](https://openskynetwork.github.io/opensky-api/rest.html) distinguishes
barometric and geometric altitude. Export only a bounded authorized window using your
legitimate access and convert it to canonical Parquet. No credentials, paid API dependency,
archive download, fabricated records, or claimed real-flight evaluation is bundled.
Aircraft without ADS-B or outside receiver coverage are missing, not evidence of absence.

## DEM and atmosphere

Use legitimately obtained local USGS 3DEP, Copernicus DEM or comparable rasters and retain
the source's license, resolution, CRS, vertical datum and nodata metadata. Dataset names
alone do not establish height datum; check the actual product. Ellipsoid raster heights
are accepted directly. Orthometric heights require an explicit local geoid separation;
large regions require preprocessing with an appropriate geoid grid, not one constant.
Wind/refraction and spatial atmospheric fields are not implemented in this release.

## Provenance and inspiration

This project is authored for Sarva Advaith Narayana. It is inspired by the creator's
[YouTube demonstration supplied by the project owner](https://youtu.be/UytYrhYLOKI).
Credit belongs to that creator for the motivating demonstration. This repository is an
independent implementation of the supplied research brief; no affiliation, endorsement,
access to the creator's code or reproduction of the creator's measured result is claimed.
