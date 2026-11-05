"""Run: streamlit run app/research.py. Research data stays on the running host."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from doppler_geo.audio.processing import load_audio, spectrogram
from doppler_geo.cli import run_config
from doppler_geo.detection.tracking import detect_events, track_ridge
from doppler_geo.experiment import run_experiment
from doppler_geo.visualization.dashboard import show_local_result

st.set_page_config(page_title="Aircraft Doppler Geolocation", layout="wide")
st.title("Aircraft Doppler Geolocation")
st.caption("By Sarva Advaith Narayana · Observations, model predictions and inferences labeled")
st.info(
    "Research prototype. Use synthetic or explicitly authorized recordings in a bounded test area. "
    "Match scores are not calibrated probabilities."
)
mode = st.sidebar.radio(
    "Research workflow",
    ["Synthetic validation", "Audio annotation", "Authorized local-data search"],
)
if mode == "Synthetic validation":
    snr = st.slider("Synthetic SNR (dB)", -20, 20, -5)
    held = st.checkbox("Independent held-out geometry")
    if st.button("Run complete synthetic experiment"):
        with st.spinner("Synthesizing audio, extracting ridges and scoring locations"):
            with TemporaryDirectory() as temp:
                result = run_experiment(temp, snr_db=snr, held_out=held)
                st.session_state["result"] = result
                st.session_state["report"] = (Path(temp) / "report.html").read_bytes()
                st.session_state["audio"] = (Path(temp) / "synthetic.wav").read_bytes()
    if "result" in st.session_state:
        result = st.session_state["result"]
        st.metric(
            "Measured synthetic localization error",
            f"{result['experiment']['localization_error_m']:.1f} m",
        )
        st.audio(st.session_state["audio"])
        st.dataframe(pd.DataFrame(result["candidates"]).drop(columns=["components", "cpa"]))
        st.download_button(
            "Download interactive diagnostic report",
            st.session_state["report"],
            "report.html",
            "text/html",
        )
        components.html(st.session_state["report"].decode(), height=900, scrolling=True)
        st.json(result["experiment"])
elif mode == "Audio annotation":
    authorized = st.checkbox("I own or am authorized to analyze this recording")
    upload = st.file_uploader("Authorized recording", type=["wav", "flac", "mp4", "mov", "mkv"])
    if upload and authorized:
        import hashlib

        identity = hashlib.sha256(upload.getvalue()).hexdigest()
        if st.session_state.get("recording_identity") != identity:
            st.session_state.pop("annotation", None)
            st.session_state["recording_identity"] = identity
        with TemporaryDirectory() as temp:
            path = Path(temp) / ("recording" + Path(upload.name).suffix.lower())
            path.write_bytes(upload.getvalue())
            try:
                audio, fs = load_audio(path)
                spec = spectrogram(audio, fs)
            except (ValueError, RuntimeError, OSError) as exc:
                st.error(str(exc))
                st.stop()
        st.audio(upload)
        st.plotly_chart(
            go.Figure(
                go.Scatter(
                    x=np.arange(0, len(audio), 16) / fs, y=audio[::16], name="Observed waveform"
                )
            )
        )
        fig = go.Figure(
            go.Heatmap(
                x=spec.times[::4],
                y=spec.frequencies,
                z=10 * np.log10(spec.power[:, ::4] + 1e-15),
                colorscale="Viridis",
            )
        )
        fig.update_layout(
            title="Observed spectrogram — zoom to inspect tonal structures",
            xaxis_title="Recording time (s)",
            yaxis_title="Frequency (Hz)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.write("Automated proposals (unverified):", detect_events(spec))
        low, high = st.slider("Ridge frequency band (Hz)", 40, 1500, (100, 350))
        begin, finish = st.slider(
            "Active event interval (recording seconds)",
            float(spec.times[0]),
            float(spec.times[-1]),
            (float(spec.times[0]), float(spec.times[-1])),
        )
        if st.button("Propose ridge"):
            try:
                from doppler_geo.audio.processing import Spectrogram

                active = (spec.times >= begin) & (spec.times <= finish)
                cropped = Spectrogram(spec.times[active], spec.frequencies, spec.power[:, active])
                event, quality = track_ridge(cropped, (low, high))
                st.session_state["annotation"] = event.to_dict()
                st.write("Heuristic ridge contrast:", quality)
            except ValueError as exc:
                st.error(str(exc))
        if "annotation" in st.session_state:
            st.write(
                "Manually edit observed time/frequency samples; harmonic is an explicit hypothesis."
            )
            data = st.session_state["annotation"]
            edited = st.data_editor(
                pd.DataFrame({"time_s": data["times"], "frequency_hz": data["frequencies"]}),
                num_rows="dynamic",
            )
            harmonic = st.number_input("Harmonic order", min_value=1, value=1)
            annotation = {
                **data,
                "times": edited["time_s"].tolist(),
                "frequencies": edited["frequency_hz"].tolist(),
                "harmonic": harmonic,
            }
            st.download_button(
                "Save annotation JSON",
                json.dumps([annotation], indent=2),
                "annotations.json",
                "application/json",
            )
else:
    st.write(
        "Run a JSON configuration against local canonical CSV/Parquet and annotation files. "
        "Paths resolve relative to the configuration file on this host. "
        "See configs/local.example.json."
    )
    config_path = st.text_input("Local configuration path", "configs/local.example.json")
    if st.button("Search authorized area"):
        try:
            with st.spinner("Searching"):
                result = run_config(config_path)
                st.session_state["local_result"] = result
            st.dataframe(
                pd.DataFrame(result["candidates"]).drop(
                    columns=["components", "cpa"], errors="ignore"
                )
            )
            for warning in result["warnings"]:
                st.warning(warning)
            if result["candidates"]:
                candidates = pd.DataFrame(result["candidates"])
                st.map(candidates.rename(columns={"latitude": "lat", "longitude": "lon"}))
            st.json({"evaluations": result["evaluations"], "runtime_s": result["runtime_s"]})
        except Exception as exc:
            st.error(str(exc))

    if "local_result" in st.session_state:
        show_local_result(st, st.session_state["local_result"])
