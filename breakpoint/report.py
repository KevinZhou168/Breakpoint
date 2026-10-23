"""Shared analysis tables and honest, portable summaries."""
import numpy as np
import pandas as pd

from .core import KINDS, first_crossing


def aggregate(rows):
    frame = pd.DataFrame(rows)
    return frame.groupby(["kind", "severity"], sort=False).agg(
        retention=("retention", "mean"), retention_min=("retention", "min"),
        retention_max=("retention", "max"), retention_std=("retention", "std"),
        confidence_mass=("confidence_mass", "mean"), matched=("matched", "mean"),
        physical_value=("physical_value", "first"), trials=("trial", "size"),
    ).reset_index()


def boundaries(rows, cutoff=0.5):
    summary = aggregate(rows)
    result = {}
    for kind in KINDS:
        subset = summary[summary.kind == kind].sort_values("severity")
        crossing = first_crossing(subset.severity, subset.retention, cutoff)
        valid = subset.retention.notna().all()
        result[kind] = {"bracket": crossing,
                        "auc": float(np.trapezoid(subset.retention, subset.severity)) if valid else None}
    return result


RESULT_HELP = {
    "Change": "The alteration applied to the original image: blur, low lighting, or noise.",
    "Severity": "Strength on this change's own scale: 0 is unchanged; 1 is the strongest tested. Equal values across different changes do not mean equal difficulty.",
    "Original objects": "Detections in the unchanged image that meet the current confidence setting. This is the reference count, not a human-verified count of real objects.",
    "Objects kept": "Original detections found again with the same label and enough box overlap. For repeated noise trials, this is the average count, so it can be fractional.",
    "Objects lost": "Original detections without an acceptable match. Includes missing detections, changed labels, low scores, or boxes that moved too far. Averaged across trials.",
    "Kept (%)": "Objects kept divided by original objects, multiplied by 100. Averaged across trials. Blank when there are no original detections. This measures consistency, not accuracy.",
    "Trials": "Number of noisy versions tested at this strength. Blur and lighting use one version. Severity 0 reuses the original result and is not a separate repeat.",
}


def results_table(rows):
    """One readable row per condition, with noise trials averaged explicitly."""
    frame = pd.DataFrame(rows)
    table = frame.groupby(["kind", "severity"], sort=False).agg(
        original=("reference_count", "first"), kept=("matched", "mean"),
        lost=("lost", "mean"), percentage=("retention", "mean"), trials=("trial", "size"),
    ).reset_index()
    table["kind"] = table["kind"].map(KINDS)
    table["percentage"] *= 100
    table.columns = list(RESULT_HELP)
    return table


def metrics_csv(rows):
    return results_table(rows).round({"Objects kept": 1, "Objects lost": 1, "Kept (%)": 1}).to_csv(index=False)


def results_guide():
    return "# Reading the results CSV\n\nOne row per change and severity. Repeated noise trials are averaged.\n\n" + "\n\n".join(
        f"**{name}:** {description}" for name, description in RESULT_HELP.items()
    ) + "\n\nFor exact alteration settings, seeds, and individual trial predictions, see experiment.json in the full download.\n"


def markdown_report(bundle, rows, confidence, iou, cutoff):
    results = boundaries(rows, cutoff)
    lines = ["# Breakpoint · experiment report", "",
             "Research question: How much controlled corruption makes clean-image detections disappear or move?", "",
             f"Run `{bundle['run_id']}` · captured {bundle['created_utc']}",
             f"Model `{bundle['config']['model_id']}` · backend `{bundle['config']['backend']}`",
             f"Image pixel SHA-256: `{bundle['image_sha256']}`",
             f"Reference detections: {rows[0]['reference_count']} at confidence ≥ {confidence:.2f}.",
             f"Class-aware one-to-one matching at IoU ≥ {iou:.2f}; failure = mean retention < {cutoff:.0%}.", "",
             "| Perturbation | First sampled crossing bracket | Retention AUC |",
             "|---|---|---|"]
    for kind, value in results.items():
        bracket = value["bracket"]
        label = (f"({bracket[0]}, {bracket[1]}]" if bracket else
                 "Not observed" if value["auc"] is not None else "Undefined (no clean references)")
        auc = f"{value['auc']:.3f}" if value["auc"] is not None else "undefined"
        lines.append(f"| {KINDS[kind]} | {label} | {auc} |")
    lines.extend(["", "## Interpretation and limits", "",
                  "Retention is consistency against the model's own clean detections, not ground-truth recall or mAP. "
                  "The baseline can be wrong. Unmatched corrupted-image detections are not necessarily false positives.",
                  "The bracket only locates the first observed crossing on a finite grid. "
                  "Recovery at later severities is possible; no monotonicity or continuous boundary is assumed.",
                  "Noise bands show the observed range across seeded trials, not a confidence interval. "
                  "Trials share a noise pattern across severities; clean references are reused.",
                  "Severity axes use different physical units; AUC is schedule-dependent and is not a universal "
                  "ranking of corruption difficulty. One image cannot support dataset-wide conclusions.",
                  "Brightness multiplication operates on encoded pixels and is not a physically accurate camera model. "
                  "Noise is additive, independent per channel, and clipped. Blur preserves geometry.",
                  "Raw predictions are captured above a fixed floor; lowering the evaluation threshold below it "
                  "requires new inference. Confidence scores are not calibrated probabilities.",
                  "The JSON includes input pixels, seeds, physical parameters, raw predictions, package versions, "
                  "and model identity. Regenerating pixels is deterministic under the recorded dependency versions; "
                  "model aliases, runtime backends, and hosted model revisions may change fresh predictions.", ""])
    return "\n".join(lines)
