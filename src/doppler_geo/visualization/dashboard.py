"""Actual trajectory playback and geolocation diagnostics for the Streamlit interface."""

import numpy as np
import plotly.graph_objects as go


def show_local_result(st, result):
    """Only plots the observations, trajectories and candidates supplied by a search."""
    flights = result["flight_visualization"]
    if not flights:
        return
    start = min(f["emission_times"][0] for f in flights)
    end = max(f["emission_times"][-1] for f in flights)
    playback = st.slider(
        "Aircraft emission-time playback (seconds since UTC origin)",
        float(start),
        float(end),
        float((start + end) / 2),
    )
    fig = go.Figure()
    for flight in flights:
        positions = np.array(flight["enu_positions"])
        times = np.array(flight["emission_times"])
        fig.add_trace(
            go.Scatter(
                x=positions[:, 0],
                y=positions[:, 1],
                mode="lines",
                name=flight["flight_id"] + " trajectory",
            )
        )
        if times[0] <= playback <= times[-1]:
            x = np.interp(playback, times, positions[:, 0])
            y = np.interp(playback, times, positions[:, 1])
            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers",
                    marker={"symbol": "triangle-up", "size": 12},
                    name=flight["flight_id"] + " at emission time",
                )
            )
    rows = result["evaluated_locations"]
    if rows:
        fig.add_trace(
            go.Scatter(
                x=[r["x_m"] for r in rows],
                y=[r["y_m"] for r in rows],
                mode="markers",
                marker={"size": 5, "color": [r["loss"] for r in rows], "colorscale": "Viridis"},
                name="Inferred location scores",
            )
        )
    if rows and st.checkbox("Show configured geometric acoustic range around best candidate"):
        best = rows[0]
        radius = result["provenance"]["config"].get("search", {}).get("max_range_m", 25000)
        fig.add_shape(
            type="circle",
            x0=best["x_m"] - radius,
            x1=best["x_m"] + radius,
            y0=best["y_m"] - radius,
            y1=best["y_m"] + radius,
            line={"dash": "dot", "color": "gray"},
        )
        st.caption("Geometric screening range only; detectability is not an audibility guarantee.")
    area = result["area"]
    fig.add_shape(
        type="rect",
        x0=area["xmin"],
        x1=area["xmax"],
        y0=area["ymin"],
        y1=area["ymax"],
        line={"color": "red", "dash": "dash"},
    )
    fig.update_layout(
        title="Aircraft trajectories and authorized search boundary",
        xaxis_title="East (m)",
        yaxis_title="North (m)",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig, use_container_width=True)
    for diagnostic in result["curve_diagnostics"]:
        curves = go.Figure()
        curves.add_trace(
            go.Scatter(
                x=diagnostic["recording_times"], y=diagnostic["observed_hz"], name="Observed Hz"
            )
        )
        curves.add_trace(
            go.Scatter(
                x=diagnostic["recording_times"],
                y=diagnostic["modeled_hz"],
                name="Modeled Hz",
                line={"dash": "dash"},
            )
        )
        curves.update_layout(
            title=diagnostic["flight_id"],
            xaxis_title="Recording time (s)",
            yaxis_title="Frequency (Hz)",
        )
        st.plotly_chart(curves, use_container_width=True)
        residual = go.Figure(
            go.Scatter(
                x=diagnostic["recording_times"], y=diagnostic["residual_hz"], name="Residual Hz"
            )
        )
        residual.update_layout(title="Observed minus modeled frequency", yaxis_title="Hz")
        st.plotly_chart(residual, use_container_width=True)
    st.json(
        {
            "frame": result["frame"],
            "provenance": result["provenance"],
            "components_and_delayed_cpa": result["candidates"],
            "near_optimal_spread_m": result["near_optimal_spread_m"],
        }
    )

    if result.get("uncertainty"):
        sensitivity = np.array(result["uncertainty"]["samples_east_north_offset"])
        if len(sensitivity):
            uncertainty_figure = go.Figure(
                go.Scatter(
                    x=sensitivity[:, 0],
                    y=sensitivity[:, 1],
                    mode="markers",
                    name="Conditional perturbation samples",
                )
            )
            uncertainty_figure.update_layout(
                title="Sensitivity, not a calibrated confidence region",
                xaxis_title="East (m)",
                yaxis_title="North (m)",
            )
            st.plotly_chart(uncertainty_figure, use_container_width=True)
    if rows:
        elevation_figure = go.Figure(
            go.Scatter(
                x=[r["x_m"] for r in rows],
                y=[r["y_m"] for r in rows],
                mode="markers",
                marker={
                    "color": [r["z_m"] for r in rows],
                    "colorscale": "Earth",
                    "showscale": True,
                    "colorbar": {"title": "ENU up (m)"},
                },
            )
        )
        elevation_figure.update_layout(
            title="Observer surface used by the model",
            xaxis_title="East (m)",
            yaxis_title="North (m)",
        )
        st.plotly_chart(elevation_figure, use_container_width=True)
