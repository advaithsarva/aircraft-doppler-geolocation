# Measured synthetic validation results

Measured on 2026-11-05 using Python 3.12.14, NumPy 2.3.5 and the direct versions in `requirements-tested.txt`. Linux x86_64; CPU only. Command:

```bash
python benchmarks/run_validation.py --output outputs/final-validation
```

The six cases use 150-second, 8 kHz independently synthesized audio, broad declared source-band priors, a common active window of 35–105 s, and operator-assigned flight identities. Search covers 2 km × 2 km with a 500 m coarse step and 25 m final step. No known observer location is passed to search. These are controlled synthetic results, not real-world accuracy.

| Case | SNR dB | Error m | Frequency RMS Hz (aircraft 1 / 2) | End-to-end s | Peak RSS MiB |
|---|---:|---:|---|---:|---:|
| baseline_0_11 | 0 | 6.25 | 0.087 / 0.150 | 3.021 | 301.1 |
| baseline_-5_42 | -5 | 6.25 | 0.115 / 0.171 | 2.713 | 303.9 |
| baseline_-15_73 | -15 | 6.25 | 0.252 / 0.791 | 2.988 | 303.9 |
| held-out_0_19 | 0 | 6.25 | 0.080 / 0.160 | 2.773 | 303.9 |
| held-out_-5_73 | -5 | 6.25 | 0.097 / 0.192 | 3.368 | 303.9 |
| held-out_-15_91 | -15 | 12.50 | 0.244 / 2.987 | 3.030 | 303.9 |

Every case scored 399 location/offset hypotheses. Runtime includes generation and extraction, excludes plotting. RSS is the process high-water mark and can carry over earlier allocations.

| Diagnostic | Error m | Shared offset s | Loss | Runtime s |
|---|---:|---:|---:|---:|
| missing_samples | 6.25 | 0.0 | 0.0058 | 0.366 |
| incorrect_altitude_plus_200m | 224.39 | 0.0 | 0.7465 | 0.337 |
| shared_recording_offset | 6.25 | 2.5 | 0.0058 | 0.895 |
| frequency_interference | 31.25 | 0.0 | 3.3490 | 0.336 |

Correct flight ordering was recovered in one two-flight synthetic permutation problem. White-noise controls produced 0/10 tonal-proposal trials at the configured 12 dB threshold. Neither result calibrates real-world aircraft recognition or geolocation false positives.

**Important counterexample:** a +200 m altitude error yielded about 224 m location error while retaining a relatively small frequency loss. A small residual alone therefore does not establish a correct location or trustworthy height datum.

The interference case uses the approximate robust score. A -15 dB held-out case has noticeably larger aircraft-2 residuals; not all noisy ridges are equally reliable.

Earlier failed runs (clipped bands, pre-arrival noise and overly permissive tonal proposals) are described in [the validation protocol](../docs/VALIDATION.md). No real-world accuracy, calibrated uncertainty coverage or national-archive performance is claimed.

[Machine-readable measurements](measured-results.json) retain seeds, configurations, errors, timing, memory, and negative controls.

Local verification: **36 tests passed** in 18.48 seconds, including a Streamlit AppTest
that ran the complete synthetic workflow and a real FFmpeg extraction check. Ruff lint,
Ruff format checks and mypy passed. A third-party anyio/pytest deprecation warning was
emitted; no tests were hidden or skipped in that run. CI is configured for Python 3.11
and 3.12; the local run used 3.12, and remote CI has not yet been verified.
