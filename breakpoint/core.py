"""Pure experimental logic. All images are uint8 BGR; geometry never changes."""
import cv2
import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment

KINDS = {"blur": "Gaussian blur", "darkness": "Low lighting", "noise": "Gaussian noise"}


def physical_value(kind, severity, shape):
    if kind == "blur":
        return severity * min(shape[:2]) * 0.035, "σ pixels"
    if kind == "darkness":
        return 10 ** (-3 * severity), "brightness multiplier"
    if kind == "noise":
        return severity * 100, "σ intensity / 255"
    raise ValueError(f"Unknown perturbation: {kind}")


def perturb(image, kind, severity, seed):
    if not np.isfinite(severity) or not 0 <= severity <= 1:
        raise ValueError("Severity must be finite and between 0 and 1")
    value, _ = physical_value(kind, severity, image.shape)
    if severity == 0:
        return image.copy()
    if kind == "blur":
        return cv2.GaussianBlur(image, (0, 0), sigmaX=value, sigmaY=value)
    if kind == "darkness":
        return np.rint(image.astype(float) * value).clip(0, 255).astype(np.uint8)
    noise = np.random.default_rng(seed).normal(0, value, image.shape)
    return np.rint(image.astype(float) + noise).clip(0, 255).astype(np.uint8)


def detections(raw, confidence):
    result = sv.Detections.from_inference(raw)
    return result[result.confidence >= confidence]


def compare(baseline, current, iou_threshold=0.5):
    """Maximum cardinality class-aware assignment, then maximum summed IoU.

    Confidence mass includes zero for unmatched reference boxes. It is a score,
    not a calibrated probability. Unmatched current boxes are not false positives.
    """
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    n, m = len(baseline), len(current)
    pairs = []
    if n and m:
        iou = sv.box_iou_batch(baseline.xyxy, current.xyxy)
        same_class = baseline.data["class_name"][:, None] == current.data["class_name"][None, :]
        eligible = same_class & (iou >= iou_threshold)
        # One extra valid match outweighs any possible summed IoU improvement.
        weights = eligible * (1 + iou / (min(n, m) + 1))
        rows, cols = linear_sum_assignment(weights, maximize=True)
        pairs = [(int(a), int(b), float(iou[a, b])) for a, b in zip(rows, cols) if eligible[a, b]]
    matched = len(pairs)
    return {
        "matched": matched, "lost": n - matched, "reference_count": n,
        "current_count": m, "unmatched_current": m - matched,
        "retention": matched / n if n else None,
        "confidence_mass": sum(float(current.confidence[b]) for _, b, _ in pairs) / n if n else None,
        "mean_matched_iou": float(np.mean([p[2] for p in pairs])) if pairs else None,
        "pairs": pairs,
    }


def first_crossing(levels, retentions, cutoff):
    """First sampled point strictly below cutoff; no monotonicity assumption."""
    previous = None
    for level, retention in zip(levels, retentions):
        if retention is None or not np.isfinite(retention):
            continue
        if retention < cutoff:
            return (previous, level)
        previous = level
    return None
