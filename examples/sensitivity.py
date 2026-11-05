"""Run a conditional uncertainty ensemble on the reproducible synthetic observation curves."""

from pathlib import Path

from doppler_geo.audio.processing import load_annotations
from doppler_geo.experiment import run_experiment
from doppler_geo.search.engine import SearchArea, SearchConfig
from doppler_geo.search.uncertainty import perturbation_ensemble
from doppler_geo.synthetic import example_scenario

if __name__ == "__main__":
    import json

    output = Path("outputs/sensitivity")
    run_experiment(output, write_plots=False)
    _, sources = example_scenario()
    result = perturbation_ensemble(
        load_annotations(output / "annotations.json"),
        [s.trajectory() for s in sources],
        SearchArea(-1000, 1000, -1000, 1000, True),
        SearchConfig(coarse_step_m=500, fine_step_m=50),
        draws=10,
        seed=31,
    )
    (output / "sensitivity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result["interpretation"])
