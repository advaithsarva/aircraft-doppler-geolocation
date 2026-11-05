"""Command line experiment and authorized local-data search."""

import argparse
import json
import logging
from glob import glob
from pathlib import Path

import numpy as np

from doppler_geo.adsb.ingest import load_tracks
from doppler_geo.audio.processing import load_annotations
from doppler_geo.physics.coordinates import LocalFrame
from doppler_geo.physics.model import predict
from doppler_geo.provenance import environment, file_digest
from doppler_geo.search.engine import SearchArea, SearchConfig, locate


def run_config(path):
    """No remote data acquisition: local authorized recordings/trajectories only."""
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text())

    def resolve(value):
        candidate = Path(value)
        return str(candidate if candidate.is_absolute() else config_path.parent / candidate)

    for key in ("tracks", "annotations"):
        config[key] = resolve(config[key])
    if config.get("dem"):
        config["dem"]["path"] = resolve(config["dem"]["path"])
    output_path = Path(resolve(config.get("output", "result.json")))
    track_inputs = sorted(glob(config["tracks"]))
    input_paths = [config_path, Path(config["annotations"]), *map(Path, track_inputs)]
    if config.get("dem"):
        input_paths.append(Path(config["dem"]["path"]))
    if output_path.resolve() in {p.resolve() for p in input_paths}:
        raise ValueError("Output must not overwrite a configuration, annotation, trajectory or DEM")
    area = SearchArea(**config["area"])
    frame = LocalFrame(**config["frame"])
    tracks, ingestion = load_tracks(config["tracks"], frame, **config["ingestion"])
    events = load_annotations(config["annotations"])
    selected = []
    for _event, flight_id in zip(events, config["flight_ids"], strict=True):
        matches = [t for t in tracks if t.segment_id == flight_id]
        if not matches:
            matches = [t for t in tracks if t.flight_id == flight_id]
        if len(matches) != 1:
            available = [t.segment_id for t in matches] or [t.segment_id for t in tracks]
            raise ValueError(
                f"{flight_id}: select a unique, continuous flight segment; "
                f"available segment IDs: {available}"
            )
        selected.append(matches[0])
    search_config = SearchConfig(**config.get("search", {}))
    terrain = None
    if config.get("dem"):
        from doppler_geo.terrain.dem import RasterTerrain

        terrain = RasterTerrain(frame=frame, **config["dem"])
    try:
        result = locate(events, selected, area, search_config, elevation=terrain)
        if config.get("uncertainty"):
            from doppler_geo.search.uncertainty import perturbation_ensemble

            result["uncertainty"] = perturbation_ensemble(
                events, selected, area, search_config, elevation=terrain, **config["uncertainty"]
            )
    finally:
        if terrain:
            terrain.close()
    result["ingestion"] = ingestion
    for row in result["candidates"]:
        row["latitude"], row["longitude"], row["ellipsoid_height_m"] = frame.to_geodetic(
            np.array([row["x_m"], row["y_m"], row["z_m"]])
        ).tolist()
    result["frame"] = config["frame"]
    result["area"] = config["area"]
    result["provenance"] = {
        "config_sha256": file_digest(config_path),
        "tracks_sha256": {p: file_digest(p) for p in track_inputs},
        "annotations_sha256": file_digest(config["annotations"]),
        "environment": environment(),
        "config": config,
    }
    result["flight_visualization"] = [
        {
            "flight_id": f.flight_id,
            "segment_id": f.segment_id,
            "emission_times": f.times.tolist(),
            "enu_positions": f.positions.tolist(),
            "geodetic_positions": frame.to_geodetic(f.positions).tolist(),
        }
        for f in selected
    ]
    result["curve_diagnostics"] = []
    if result["candidates"]:
        best = result["candidates"][0]
        for event, flight, component in zip(events, selected, best["components"], strict=True):
            expected = predict(
                flight,
                [best["x_m"], best["y_m"], best["z_m"]],
                event.times + best["offset_s"],
                search_config.c,
            ) * (component["f0_hz"] * event.harmonic)
            result["curve_diagnostics"].append(
                {
                    "flight_id": flight.flight_id,
                    "recording_times": event.times.tolist(),
                    "observed_hz": event.frequencies.tolist(),
                    "modeled_hz": expected.tolist(),
                    "residual_hz": (event.frequencies - expected).tolist(),
                    "sigma_hz": event.sigma_hz,
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description="Aircraft Doppler Geolocation research engine")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run independent synthetic audio validation")
    demo.add_argument("--output", default="outputs/demo")
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--snr-db", type=float, default=-5)
    demo.add_argument("--held-out", action="store_true")
    demo.add_argument("--no-plots", action="store_true")
    search = sub.add_parser("search", help="Search a configured authorized test area")
    search.add_argument("config")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.command == "demo":
        from doppler_geo.experiment import run_experiment

        result = run_experiment(
            args.output, args.seed, args.snr_db, args.held_out, not args.no_plots
        )
        print(json.dumps(result["experiment"], indent=2))
    else:
        result = run_config(args.config)
        print(json.dumps(result["candidates"], indent=2))


if __name__ == "__main__":
    main()
