"""Run with: streamlit run app.py"""
import io
import json
import os
from pathlib import Path
import zipfile

from breakpoint.detector import configure_runtime
configure_runtime()

import cv2
import numpy as np
from PIL import UnidentifiedImageError
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from breakpoint.core import KINDS, detections, perturb
from breakpoint.detector import make_detector, normalize_model_id, inference_error_message
from breakpoint.experiment import ExperimentConfig, analyze, decode_image, run_experiment
from breakpoint.inputs import load_input
from breakpoint.report import aggregate, boundaries, markdown_report, metrics_csv, results_table, RESULT_HELP, results_guide
from breakpoint.visuals import annotate_pair
from breakpoint.presentation import boundary_display, condition_display, CONDITION_HELP, status_matrix

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=True)
st.set_page_config(page_title="Breakpoint · detector robustness lab", page_icon="◩", layout="wide")
st.markdown("""<style>
.block-container {max-width:1440px; padding-top:4.5rem; padding-bottom:3rem}
h1 {font-size:3.8rem!important; font-weight:650!important; letter-spacing:-.065em; line-height:1.04!important}
h2 {letter-spacing:-.035em} h3 {letter-spacing:-.025em}
[data-testid="stMetric"] {background:#171f27; border:1px solid #2b3741; border-radius:10px; padding:16px}
[data-testid="stMetricLabel"] {color:#a4b1be}
[data-testid="stMetricValue"] {font-variant-numeric:tabular-nums; letter-spacing:-.045em}
.eyebrow {font-family:monospace; font-size:12px; color:#b7f36b; letter-spacing:.14em; margin-bottom:20px}
.intro {font-size:1.1rem; max-width:780px; color:#aebac6; margin:18px 0 24px}
.runline {font-family:monospace; font-size:12px; color:#91a1af; padding:12px 0 20px}
.legend {font-family:monospace; font-size:12px; color:#aebac6; padding:8px 0 16px}
button[kind="primary"] {background:#b7f36b!important; color:#10171d!important; border:0!important}
[data-testid="stSidebar"] {border-right:1px solid #28313b}
@media (max-width:640px) {h1 {font-size:2.7rem!important} .eyebrow {font-size:10px; letter-spacing:.08em}}
</style>""", unsafe_allow_html=True)


@st.cache_data
def recorded_experiment(path, modified):
    return json.loads(Path(path).read_text())


def curve_figure(summary, cutoff, shape):
    colors = {"blur": "#72B5FF", "darkness": "#B7F36B", "noise": "#FFB56B"}
    fig = go.Figure()
    for kind, title in KINDS.items():
        subset = summary[summary.kind == kind].sort_values("severity")
        if kind == "noise":
            fig.add_trace(go.Scatter(x=subset.severity, y=subset.retention_max, mode="lines",
                                     line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=subset.severity, y=subset.retention_min, mode="lines",
                                     line=dict(width=0), fill="tonexty", fillcolor="rgba(255,181,107,.13)",
                                     showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=subset.severity, y=subset.retention, name=title,
                                 mode="lines+markers", line=dict(color=colors[kind], width=3),
                                 marker=dict(size=7), customdata=[[condition_display(kind, float(level), shape)[0], int(n)]
                                                               for level, n in zip(subset.severity, subset.trials)],
                                 hovertemplate=f"{title}<br>Severity %{{x:.2f}}<br>Retention %{{y:.1%}}"
                                               "<br>Actual adjustment %{customdata[0]}"
                                               "<br>Trials %{customdata[1]}<extra></extra>"))
    fig.add_hline(y=cutoff, line_dash="dash", line_color="#FF775E",
                  annotation_text=f"failure criterion · <{cutoff:.0%}", annotation_position="bottom left")
    fig.update_layout(height=340, margin=dict(l=10, r=20, t=20, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#aebac6"), legend=dict(orientation="h", y=1.16, x=0),
                      xaxis=dict(title="Controlled severity →", range=[-0.02, 1.02], gridcolor="#28313b"),
                      yaxis=dict(title="Clean detections retained", tickformat=".0%", range=[-0.04, 1.04], gridcolor="#28313b"))
    return fig


with st.sidebar:
    st.markdown("### ◩ Breakpoint")
    st.caption("A small instrument for finding detector failure boundaries.")
    mode = st.radio("Experiment source", ["Recorded real experiment", "Run live inference"])
    if mode == "Run live inference":
        upload = st.file_uploader("Input image", type=["jpg", "jpeg", "png", "webp"])
        try:
            input_image = load_input(upload if upload is not None else ROOT / "assets/people-walking.jpg")
        except (ValueError, UnidentifiedImageError, OSError) as error:
            st.error(str(error))
            st.stop()
        st.image(input_image, channels="BGR", caption="Actual experimental input · max edge 960 px")
        backend_label = st.selectbox("Inference backend", ["Local · Roboflow Inference", "HTTP · Roboflow SDK"])
        backend = "local" if backend_label.startswith("Local") else "http"
        model_id = st.text_input("Model ID", "rfdetr-small", help="Use project/version, for example vehicle-detection-wvcx6/1. A workspace/project/version ID is shortened automatically.")
        model_error = None
        try:
            effective_model_id = normalize_model_id(model_id)
            if effective_model_id != model_id.strip():
                st.caption(f"Using {effective_model_id} · the workspace prefix is not part of the inference ID.")
        except ValueError as error:
            model_error = str(error)
            st.warning(model_error)
        api_url = "https://serverless.roboflow.com"
        if backend == "http":
            api_url = st.text_input("Inference server", api_url)
            st.caption("Images are sent to this server. " + ("API key loaded." if os.getenv("ROBOFLOW_API_KEY") else "No API key loaded; add ROBOFLOW_API_KEY to .env for hosted inference."))
        else:
            st.caption("Runs on this machine. First run downloads weights. RF-DETR Small needs no key; private models use ROBOFLOW_API_KEY from .env.")
        with st.expander("Sweep settings"):
            steps = st.select_slider("Severity levels", options=[5, 7, 11, 21], value=11)
            trials = st.slider("Noise trials", 1, 5, 3)
            seed = st.number_input("Random seed", 0, 2**31 - 1, 42)
        calls = 1 + (steps - 1) * (2 + trials)
        st.caption(f"{calls} inference calls · three independent perturbations · capture floor 0.05")
        run = st.button("Run robustness experiment", type="primary", width="stretch", disabled=model_error is not None)
        if run:
            config = ExperimentConfig(model_id=effective_model_id, backend=backend, api_url=api_url.strip(),
                                      levels=tuple(round(i / (steps - 1), 6) for i in range(steps)),
                                      noise_trials=trials, seed=int(seed))
            progress = st.progress(0, text="Loading Roboflow model…")
            try:
                infer = make_detector(config, os.getenv("ROBOFLOW_API_KEY"))
                bundle = run_experiment(input_image, config, infer,
                                        lambda n, total: progress.progress(n / total, text=f"Inference {n}/{total}"))
                st.session_state["live_bundle"] = bundle
                progress.empty()
                st.success("Experiment complete. All predictions are from real inference.")
            except Exception as error:
                # SDK exceptions can contain request details. Never show secrets or URLs with keys.
                progress.empty()
                st.error(inference_error_message(error))
    st.divider()
    st.markdown("**Evaluation controls**")
    confidence = st.slider("Detection confidence ≥", 0.05, 0.95, 0.25, 0.05)
    match_iou = st.slider("Matching IoU ≥", 0.10, 0.90, 0.50, 0.05)
    cutoff = st.slider("Failure below retention", 0.10, 0.90, 0.50, 0.05)
    st.caption("These controls reanalyze captured predictions without new inference calls. Changing confidence also changes the clean reference set.")

st.markdown('<div class="eyebrow">ROBOFLOW LABS / EXPERIMENT 001 / DETECTOR CONSISTENCY</div>', unsafe_allow_html=True)
st.title("Every detector has a breaking point.")
st.markdown('<div class="intro">How much blur, darkness, or noise makes a detector lose the objects it could see? '
            'Change one condition. Measure what survives. Inspect where it breaks.</div>', unsafe_allow_html=True)

if mode == "Recorded real experiment":
    path = ROOT / "artifacts/demo.json"
    if not path.exists():
        st.info("No recorded experiment yet. Choose Run live inference, or run python scripts/run_experiment.py.")
        st.stop()
    bundle = recorded_experiment(str(path), path.stat().st_mtime)
    st.caption("● REPLAY · Recorded real Roboflow predictions. Scrubbing and evaluation controls make no inference calls.")
else:
    bundle = st.session_state.get("live_bundle")
    if bundle is None:
        st.info("Choose an image and run the experiment in the sidebar to measure its failure boundaries.")
        st.stop()
    st.caption("● LIVE RUN RESULTS · Real Roboflow predictions from the last completed sweep. Sidebar input changes apply when you run again.")

rows = analyze(bundle, confidence, match_iou)
summary = aggregate(rows)
limits = boundaries(rows, cutoff)
image = decode_image(bundle["image_png"])
baseline = detections(bundle["baseline"], confidence)
st.markdown(f'<div class="runline">MODEL {bundle["config"]["model_id"]} · '
            f'{image.shape[1]} × {image.shape[0]} PX · {bundle["inference_calls"]} REAL CALLS · '
            f'RUN {bundle["run_id"]} · {bundle["created_utc"][:10]}</div>', unsafe_allow_html=True)
if not len(baseline):
    st.warning("No clean detections at this confidence threshold. Retention and boundaries are undefined. Lower the evaluation confidence or try another image/model.")

atlas, traces, protocol = st.tabs(["01  Failure atlas", "02  Object traces", "03  Results & export"])

with atlas:
    st.subheader("The failure envelope")
    st.plotly_chart(curve_figure(summary, cutoff, image.shape), width="stretch", key="envelope")
    st.caption("Points are measured, lines guide the eye. Noise shading is the observed min–max range across seeded trials, not a confidence interval. Severity units differ by perturbation.")
    cards = st.columns(3)
    for column, (kind, title) in zip(cards, KINDS.items()):
        bracket = limits[kind]["bracket"]
        display = boundary_display(kind, bracket, image.shape, bool(len(baseline)))
        column.metric(title + " · first failure", display["value"], help=CONDITION_HELP[kind])
        column.caption(display["label"] + ". " + display["detail"])
    st.caption(f"Failure here means fewer than {cutoff:.0%} of the original detections were kept. "
               "Each arrow goes from the last tested setting before the first failure to the first failing setting. "
               "These are different physical scales, not equal amounts of difficulty.")
    st.divider()
    left, right = st.columns([1, 2])
    with left:
        kind = st.selectbox("Perturbation", list(KINDS), format_func=KINDS.get)
    levels = bundle["config"]["levels"]
    bracket = limits[kind]["bracket"]
    default_level = bracket[1] if bracket else levels[len(levels) // 2]
    with right:
        severity = st.select_slider("Severity · inspect a measured point", options=levels,
                                    value=default_level, key=f"severity_{bundle['run_id']}_{kind}")
    trial_options = sorted({r["trial"] for r in bundle["records"] if r["kind"] == kind and r["severity"] == severity})
    trial = st.selectbox("Noise realization / trial", trial_options, format_func=lambda x: f"Trial {x + 1}", disabled=len(trial_options) == 1)
    record = next(r for r in bundle["records"] if r["kind"] == kind and r["severity"] == severity and r["trial"] == trial)
    transformed = perturb(image, kind, severity, record["seed"])
    show_lost = st.checkbox("Show lost reference boxes in coral", value=True)
    reference_view, current_view, point = annotate_pair(image, transformed, bundle["baseline"], record["raw"], confidence, match_iou, show_lost)
    stat_cols = st.columns(4)
    stat_cols[0].metric("Clean reference", len(baseline))
    stat_cols[1].metric("Retained at this point", f"{point['retention']:.0%}" if point["retention"] is not None else "Undefined")
    stat_cols[2].metric("Lost / changed", point["lost"])
    amount, amount_label = condition_display(kind, severity, image.shape)
    stat_cols[3].metric(amount_label, amount, help=CONDITION_HELP[kind])
    st.markdown('<div class="legend"><span style="color:#72B5FF">■</span> clean / unmatched current '
                '&nbsp; <span style="color:#B7F36B">■</span> retained '
                '&nbsp; <span style="color:#FF775E">■</span> lost reference (overlay only)</div>', unsafe_allow_html=True)
    clean_col, corrupt_col = st.columns(2)
    clean_col.image(reference_view, channels="BGR", caption=f"CONTROL · {len(baseline)} reference detections", width="stretch")
    corrupt_col.image(current_view, channels="BGR", caption=f"{KINDS[kind].upper()} · severity {severity:.2f} · {point['matched']} retained / {point['current_count']} current detections", width="stretch")
    st.caption("Lost means absent, below confidence threshold, changed class, or moved below the IoU criterion. Blue unmatched current boxes may be new objects or shifted predictions; they are not labeled false positives.")

with traces:
    st.subheader("Which objects break first?")
    trace_kind = st.selectbox("Trace perturbation", list(KINDS), format_func=KINDS.get, key="trace_kind")
    if len(baseline):
        labels = [f"#{i+1} {name} · {score:.2f}" for i, (name, score) in enumerate(zip(baseline.data["class_name"], baseline.confidence))]
        confidence_matrix = np.zeros((len(baseline), len(levels)))
        for col, level in enumerate(levels):
            subset = [r for r in rows if r["kind"] == trace_kind and r["severity"] == level]
            raw_subset = [r for r in bundle["records"] if r["kind"] == trace_kind and r["severity"] == level]
            for row, raw_record in zip(subset, raw_subset):
                current = detections(raw_record["raw"], confidence)
                for a, b, _ in row["pairs"]:
                    confidence_matrix[a, col] += float(current.confidence[b]) / len(subset)
        st.markdown("**Detection-status matrix**")
        if trace_kind == "noise":
            n_trials = bundle["config"]["noise_trials"]
            st.caption(f"Each square shows how many of the {n_trials} noise trials retained this object. "
                       "The legend lists the possible counts. The blue checkmarks at severity 0 are the reused original result.")
        else:
            st.caption("One row per original object, one column per tested strength. "
                       "A green ✓ means retained; a coral × means lost. There are no in-between states.")
        matrix_figure = status_matrix([r for r in rows if r["kind"] == trace_kind],
                                      levels, labels, trace_kind, image.shape)
        with st.container(height=min(800, matrix_figure.layout.height + 25), border=False):
            st.plotly_chart(matrix_figure, width="stretch")
        st.caption("Hover over a square for the actual blur, darkness, or noise amount. "
                   "Retained means the label and box position still meet your matching settings.")
        index = st.selectbox("Follow one reference object", range(len(baseline)), format_func=lambda i: labels[i])
        fig = go.Figure(go.Scatter(x=levels, y=confidence_matrix[index], mode="lines+markers",
                                   line=dict(color="#72B5FF", width=3), name="Matched confidence mass"))
        fig.update_layout(height=260, xaxis_title="Severity", yaxis=dict(title="Matched confidence · zero if lost", range=[0, 1]),
                          margin=dict(l=5, r=5, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")
        st.caption("IDs refer to the clean reference boxes in the atlas. Assignment is recomputed independently per image; these are reference correspondences, not tracking IDs. Unmatched objects contribute zero to the confidence trace.")

with protocol:
    st.subheader("Your results")
    st.write("How many of the original detections survived each change? Each row is one tested strength.")
    st.caption("For noise, counts and percentages are averaged across trials. Hover over a column heading for its explanation.")
    table = results_table(rows)
    column_config = {
        "Change": st.column_config.TextColumn(help=RESULT_HELP["Change"]),
        "Severity": st.column_config.NumberColumn(help=RESULT_HELP["Severity"], format="%.2f"),
        "Original objects": st.column_config.NumberColumn(help=RESULT_HELP["Original objects"], format="%d"),
        "Objects kept": st.column_config.NumberColumn(help=RESULT_HELP["Objects kept"], format="%.1f"),
        "Objects lost": st.column_config.NumberColumn(help=RESULT_HELP["Objects lost"], format="%.1f"),
        "Kept (%)": st.column_config.NumberColumn(help=RESULT_HELP["Kept (%)"], format="%.1f%%"),
        "Trials": st.column_config.NumberColumn(help=RESULT_HELP["Trials"], format="%d"),
    }
    st.dataframe(table, column_config=column_config, hide_index=True, width="stretch")
    st.info("These results measure whether the model keeps finding its original detections. "
            "They do not tell us whether those detections are correct.")
    with st.expander("What does each column mean?"):
        for name, description in RESULT_HELP.items():
            st.markdown(f"**{name}.** {description}")
    st.subheader("Save your results")
    st.download_button("Download results CSV", metrics_csv(rows),
                       file_name=f"breakpoint-{bundle['run_id']}-results.csv", mime="text/csv", type="primary")
    st.download_button("Download column guide", results_guide(), file_name="results-guide.md", mime="text/markdown")
    st.caption("The CSV uses the same seven columns as the table. Choose the full experiment below if you need the original predictions and images.")
    with st.expander("How this test works"):
        st.markdown("""1. **Find objects in the original picture.** These detections become the reference set.
2. **Change one condition at a time.** Make separate versions with blur, darkness, or random noise, then run the same model on each.
3. **Count what survives.** An object is kept if the model returns the same label and a box close enough to its original location. Each new box can match only one original object.

The failure point is the first tested strength where the percentage kept falls below your chosen cutoff. It is an approximate interval between tested strengths, not an exact threshold. Objects can disappear and reappear.

Noise has several trials because different random speckle patterns can produce different results. The same seed—a number used to recreate a pattern—makes each trial repeatable.""")
    with st.expander("What do the severity numbers mean?"):
        st.markdown("""**Severity is a slider from 0 (unchanged) to 1 (the strongest change we test).** The scales differ, so 0.5 blur is not necessarily as difficult as 0.5 darkness.

- **Blur:** blend neighboring pixels. The spread is severity × 3.5% of the image's shorter side, measured in pixels.
- **Darkness:** multiply the original pixel values by 10 raised to (−3 × severity). At severity 0.5, about 3.2% of the original values remain.
- **Noise:** add random positive or negative changes to every color channel of every pixel. The typical spread is severity × 100 on the 0–255 pixel scale.

The atlas shows the actual adjustment for the point you select.""")
    with st.expander("What this test cannot tell you"):
        st.markdown("""The original detections may be wrong or incomplete. Without a person marking the correct answers, we cannot measure actual accuracy.

One picture and a few noise patterns cannot establish how a model behaves on all images. These artificial changes also do not reproduce every effect of a real camera in poor conditions.

Changing the confidence setting changes which original detections count. Changing the box-overlap setting changes how close boxes must be to match. Both can affect the results.""")
    with st.expander("Technical settings and Roboflow tools"):
        st.caption("Roboflow Inference runs the detector; Inference SDK connects to a server when selected. Supervision reads the predictions, measures box overlap, and draws annotations. IoU is the box-overlap score: 0 means no overlap and 1 means identical boxes.")
        st.json({**{k: v for k, v in bundle.items() if k not in ("image_png", "baseline", "records", "evaluation")},
                 "evaluation": {"confidence": confidence, "match_iou": match_iou, "failure_cutoff": cutoff}})
    report = markdown_report(bundle, rows, confidence, match_iou, cutoff)
    export = {**bundle, "evaluation": {"confidence": confidence, "match_iou": match_iou, "failure_cutoff": cutoff}}
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("experiment.json", json.dumps(export, indent=2, allow_nan=False))
        z.writestr("results.csv", metrics_csv(rows))
        z.writestr("results-guide.md", results_guide())
        z.writestr("report.md", report)
        for filename, pixels in [("input.png", image), ("clean-annotated.png", reference_view),
                                 (f"{kind}-{severity}-trial{trial}-annotated.png", current_view)]:
            z.writestr(filename, cv2.imencode(".png", pixels)[1].tobytes())
    st.download_button("Download full experiment", archive.getvalue(),
                       file_name=f"breakpoint-{bundle['run_id']}.zip", mime="application/zip")
    st.caption("ZIP with results, a column guide, the original predictions and settings, a detailed report, and images. No API keys are exported.")
    st.markdown("[Inference Python API](https://docs.roboflow.com/reference/inference/inference-python/native-python-api) · "
                "[Inference SDK](https://docs.roboflow.com/reference/inference/inference-sdk) · "
                "[Supervision](https://supervision.roboflow.com/latest/detection/core/)")

st.divider()
st.caption("BREAKPOINT / Built with Roboflow Inference + Supervision / Consistency is evidence. It is not correctness.")
