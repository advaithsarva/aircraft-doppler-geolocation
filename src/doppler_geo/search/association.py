"""Budgeted candidate flight assignments within an already authorized local dataset."""

from dataclasses import replace
from itertools import islice, permutations
from math import perm
from time import monotonic

from doppler_geo.search.engine import SearchConfig, locate
from doppler_geo.validation import finite_scalar, positive_integer


def rank_assignments(
    events,
    flights,
    area,
    config=None,
    max_assignments=20,
    total_evaluations=10000,
    total_runtime_s=120,
):
    """Enumerate small local candidate lists; require explicit budgets.

    This does not download flight archives or infer aircraft types. Each event is assigned
    a distinct trajectory; harmonic ridges from one aircraft should first be consolidated.
    """
    positive_integer(max_assignments, "max_assignments")
    positive_integer(total_evaluations, "total_evaluations")
    finite_scalar(total_runtime_s, "total_runtime_s", positive=True)
    if not 1 <= max_assignments <= 100 or len(flights) < len(events) or not events:
        raise ValueError("Invalid local association problem or assignment budget")
    count = perm(len(flights), len(events))
    attempts = min(count, max_assignments)
    if total_evaluations < attempts or total_runtime_s <= 0:
        raise ValueError("Insufficient association computation budget")
    config = config or SearchConfig()
    per_assignment = replace(
        config, max_evaluations=min(config.max_evaluations, total_evaluations // attempts)
    )
    started = monotonic()
    results = []
    searched = 0
    complete_searches = 0
    for selected in islice(permutations(flights, len(events)), attempts):
        remaining = total_runtime_s - (monotonic() - started)
        if remaining <= 0:
            break
        result = locate(
            events,
            selected,
            area,
            replace(per_assignment, max_runtime_s=min(config.max_runtime_s, remaining)),
        )
        searched += 1
        complete_searches += bool(result["complete"])
        if result["candidates"]:
            results.append(
                {
                    "flight_ids": [f.flight_id for f in selected],
                    "loss": result["candidates"][0]["loss"],
                    "result": result,
                }
            )
    results.sort(key=lambda r: r["loss"])
    return {
        "assignments": results,
        "possible_assignments": count,
        "exhaustive": complete_searches == count,
        "searched_assignments": searched,
        "complete_assignments": complete_searches,
        "warning": "Assignment ranking is conditional on the supplied local flight shortlist.",
    }
