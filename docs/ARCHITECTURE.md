# Technical architecture

Project owner and author: Sarva Advaith Narayana.

## Repository inspection

Inspected `main` at `f2a8eb04e9e3a2c7c3b35c86c4dc0bf7891cd089` before implementation.
The repository had a 31-byte README, no code, data, dependency manifest, tests, or
AGENTS.md. There was no reusable implementation or architecture constraint beyond
preserving history. Existing README content is retained as the first heading.

## Data flow and boundaries

```
Authorized WAV/FLAC/video -> audio/STFT -> tonal proposals -> operator annotations
                                                          |
Local CSV/Parquet -> UTC/datum validation -> trajectory ----+-> bounded ENU search
DEM -> ellipsoid conversion -> observer elevation ---------+          |
                                                   retarded-time model
                                                          |
                                   joint profiled scores + shared UTC offset
                                                          |
                           ranked candidates + residuals + sensitivity diagnostics
```

All physics uses SI meters/seconds and one local Earth-centered ENU frame. All
trajectories use seconds since an explicit UTC origin. Audio annotations use seconds
since recording start. The shared offset maps recording seconds to trajectory seconds.
No location truth is passed to the optimizer; synthetic truth is used only to evaluate
outputs. Synthetic synthesis uses analytic paths and retarded source phase, independently
of production interpolation/prediction.

- `physics`: coordinate transformations, PCHIP trajectories, sound speed, delay and CPA.
- `audio`: bounded file loading, FFmpeg extraction, resampling, STFT, annotation JSON.
- `detection`: temporal tonal excess and dynamic-programming ridge proposals. These
  are signal structures, not aircraft identities. Band selection is a research input.
- `adsb`: canonical CSV/Parquet ingestion, OpenSky export normalization, duplicate and
  gap handling, conservative KD-tree index, bounded DuckDB queries.
- `terrain`: optional raster sampling, CRS conversion and explicit vertical datum.
- `scoring`: frequency uncertainty, unknown source tone profiling, robust and Gaussian
  losses, harmonic hypothesis and spectrogram-energy evidence.
- `search`: authorized rectangle, CPA/range filtering, shared offset grid,
  coarse-to-fine candidates, bounded flight assignments, perturbation ensembles.
- `visualization`: portable Plotly report; `app/research.py` exposes actual pipeline tools.
- `experiment`/`cli`: reproducible synthetic experiment and local-data configuration.

## Computation and memory

Polars lazy spatial filtering precedes collection, with a one-million-row upper budget.
DuckDB has a 512 MB memory limit for standalone window queries. Time and spatial
partitions in Parquet are strongly recommended: the loader supports a local file or
Parquet glob, not acquisition of a national archive. A row limit bounds collected memory,
not the time to scan badly partitioned source data. Use DuckDB to prepare a small window
first. FlightIndex is available for shortlisting; search currently receives a shortlist
and performs CPA screening directly. It does not pretend to index a whole country.

Search rejects an oversized coarse grid before constructing it, counts each
location/offset evaluation, caps elapsed runtime, retains only bounded evaluated rows,
and marks interrupted runs incomplete. Fine search explores neighborhoods of the best
coarse candidates. It can miss a narrow distant mode; this is an approximation, not a
proof of the global optimum. Parameter sweeps and multiple coarse resolutions are needed
for demanding experiments.

Frequency tracking is O(FKT), where K is the bounded predecessor-band width; a work
budget rejects oversized tracking requests before constructing transition arrays. Do not run wideband ridge
tracking on hour-long audio at fine resolution. Audio defaults to a 600 s duration cap.
CPU NumPy/SciPy profiling comes before Numba/GPU work. No GPU dependency is needed.

## Scientific scope

A working research prototype is provided, not an externally validated operational
locator. Correctness is tested under homogeneous, subsonic, stationary-observer
assumptions. Recorded engine tones can vary in time, aircraft can be absent from ADS-B,
and automated ridge proposals can be wrong. The UI never turns a score into probability.
No authorized real recording or historical archive was supplied for real-world evaluation.

The package supports manual flight associations and bounded automatic local assignment
ranking. Aircraft type taxonomy, arbitrary source-frequency variation, missing-aircraft
probabilistic evidence, atmospheric ray tracing, wind/refraction, terrain occlusion,
reflection/multipath and calibrated false-positive rates remain research extensions.
