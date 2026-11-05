"""Self-contained scientific diagnostic report, observed vs modeled data explicit."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from doppler_geo.physics.model import predict


def write_report(path, spec, events, tracks, result, c=343.0):
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Observed spectrogram (dB)",
            "Candidate scores (lower better)",
            "Observed and modeled curves",
            "Frequency residuals",
        ),
    )
    fig.add_trace(
        go.Heatmap(
            x=spec.times[::2],
            y=spec.frequencies,
            z=10 * np.log10(spec.power[:, ::2] + 1e-15),
            showscale=False,
            colorscale="Viridis",
        ),
        row=1,
        col=1,
    )
    rows = result["evaluated_locations"]
    fig.add_trace(
        go.Scatter(
            x=[r["x_m"] for r in rows],
            y=[r["y_m"] for r in rows],
            mode="markers",
            marker={"color": [r["loss"] for r in rows], "colorscale": "Viridis", "size": 6},
            name="inferred candidates",
        ),
        row=1,
        col=2,
    )
    if result["candidates"]:
        best = result["candidates"][0]
        observer = [best["x_m"], best["y_m"], best["z_m"]]
        for event, track, component in zip(events, tracks, best["components"], strict=True):
            modeled = predict(track, observer, event.times + best["offset_s"], c) * (
                component["f0_hz"] * event.harmonic
            )
            fig.add_trace(
                go.Scatter(
                    x=event.times, y=event.frequencies, mode="lines", name=event.label + " observed"
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=event.times,
                    y=modeled,
                    mode="lines",
                    line={"dash": "dash"},
                    name=event.label + " modeled",
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=event.times, y=event.frequencies - modeled, name=event.label + " residual Hz"
                ),
                row=2,
                col=2,
            )
    fig.update_layout(title="Aircraft Doppler Geolocation — Sarva Advaith Narayana", height=850)
    fig.update_xaxes(title_text="Recording time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1)
    fig.update_xaxes(title_text="East (m)", row=1, col=2)
    fig.update_yaxes(title_text="North (m)", row=1, col=2, scaleanchor="x2", scaleratio=1)
    fig.write_html(str(path), include_plotlyjs=True, full_html=True)
