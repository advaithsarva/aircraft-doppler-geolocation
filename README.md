# aircraft-doppler-geolocation

**Aircraft Doppler Acoustic Geolocation Engine — by Sarva Advaith Narayana**

A working Python research prototype that simulates aircraft acoustics, extracts tonal
ridges from audio, and scores possible observer locations within an explicitly authorized
test area. Single and multiple aircraft share one physically consistent recording-time
alignment, including sound propagation delay.

**Inspiration:** We took inspiration from the creator's [YouTube demonstration](https://youtu.be/UytYrhYLOKI).
Thank you for the motivating work. This is an independent implementation; we do not claim
access to the creator's code, their experimental results, affiliation or endorsement.
See [attribution](docs/INSPIRATION.md).

## Run the complete synthetic experiment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,app]'
doppler-geo demo --output outputs/demo
doppler-geo demo --held-out --seed 73 --snr-db -15 --output outputs/held-out
pytest -q
streamlit run app/research.py
```

Python 3.11+; CPU only. The demo writes a WAV, extracted annotation curves, JSON metrics
and a standalone interactive HTML report. The pipeline is:

**Analytic aircraft motion → retarded-phase noisy audio → STFT → observed ridge extraction
→ bounded candidate search → measured error against known observer truth.**

Truth coordinates are used only by the generator and evaluator. The optimizer receives
the same complete rectangular search area for every scenario. Source-band and active-time
windows and event-to-flight associations are explicit synthetic priors. This demo tests
the inference pipeline; it is not an automatic aircraft classifier or real-world accuracy claim.

## Analyze local authorized recordings

1. In the dashboard, load a recording you own or have explicit permission to analyze.
   Inspect the spectrogram and tonal proposals. Edit time/frequency samples and export
   annotation JSON. Do not assume every curved ridge is an aircraft.
2. Export a bounded historical flight window through your legitimate data access. Supply
   canonical CSV/Parquet with an aware UTC timestamp and an explicit height datum.
3. Copy [the example config](configs/local.example.json), select flight IDs, UTC origin,
   timing uncertainty grid and authorized test boundary. Set `authorized: true` only for
   recordings and areas you are authorized to investigate. Supply a DEM for actual terrain.
4. Run `doppler-geo search your-config.json`, or use the local-data workflow in the dashboard.

FFmpeg must be installed separately for video files. No remote ADS-B archive acquisition,
paid API, nationality-scale location sweep or person tracking is implemented.

## Scientific capabilities

- WGS84/ENU geometry, 3D slant range, altitude, vertical motion, PCHIP interpolation.
- CPA and delayed CPA, monotonic arrival-time mapping, configurable sound speed.
- WAV/FLAC and video audio, resampling, band-limited STFT and manual annotations.
- Tonal proposals and dynamic-programming ridge extraction with heuristic quality scores.
- Parquet/CSV validation, duplicate conflicts, flight segmentation, lazy bounded filtering.
- Local OpenSky export normalization, DuckDB window queries and conservative spatial index.
- Profiled unknown fundamental frequency, explicit harmonics, robust/Gaussian curve losses,
  optional spectrogram-energy evidence, joint matching and bounded flight assignments.
- Coarse-to-fine search, computation budgets, residual diagnostics and conditional sensitivity.
- Optional raster DEM sampling with explicit CRS and vertical datum.

**Limits:** Homogeneous still air and subsonic aircraft only. Wind, refraction, multipath,
terrain occlusion and varying engine tones are not modeled. ADS-B is incomplete. A single
flight can be ambiguous. Candidate scores and sensitivity spreads are **not calibrated
probabilities or confidence regions**. No authorized real-world data was supplied, so no
real-world localization accuracy is claimed. See [phase status](docs/IMPLEMENTATION_CHECKLIST.md).

## Reproduce and assess

```bash
ruff check .
ruff format --check .
mypy src/doppler_geo
pytest -q
python benchmarks/run_validation.py --output outputs/validation
```

[Measured synthetic results](benchmarks/RESULTS.md) ·
[architecture](docs/ARCHITECTURE.md) · [mathematics](docs/MATHEMATICS.md) ·
[data/dependency plan](docs/DATA_AND_DEPENDENCIES.md) · [validation protocol](docs/VALIDATION.md).

MIT license © 2026 Sarva Advaith Narayana. Data licenses are separate from the software license.
