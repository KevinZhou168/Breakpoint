"""Supervision annotations for matched, lost, and unmatched detections."""
import numpy as np
import supervision as sv

from .core import detections, compare

GREEN = "#B7F36B"
RED = "#FF775E"
BLUE = "#72B5FF"


def draw(image, det, labels, color):
    if not len(det):
        return image
    image = sv.BoxAnnotator(color=sv.Color.from_hex(color), thickness=2).annotate(image, det)
    return sv.LabelAnnotator(color=sv.Color.from_hex(color), text_color=sv.Color.BLACK,
                             text_scale=0.4, text_thickness=1, text_padding=4).annotate(image, det, labels)


def label(det, indices, suffix=""):
    return [f"#{i+1} {det.data['class_name'][i]} {det.confidence[i]:.2f}{suffix}" for i in indices]


def annotate_pair(image, transformed, baseline_raw, raw, confidence, iou, show_lost=True):
    baseline, current = detections(baseline_raw, confidence), detections(raw, confidence)
    metrics = compare(baseline, current, iou)
    reference = draw(image.copy(), baseline, label(baseline, range(len(baseline))), BLUE)
    view = transformed.copy()
    matched_a = {a for a, _, _ in metrics["pairs"]}
    matched_b = {b for _, b, _ in metrics["pairs"]}
    if show_lost:
        lost = np.array([i for i in range(len(baseline)) if i not in matched_a], dtype=int)
        view = draw(view, baseline[lost], label(baseline, lost, " LOST"), RED)
    extra = np.array([i for i in range(len(current)) if i not in matched_b], dtype=int)
    view = draw(view, current[extra], [f"{current.data['class_name'][i]} {current.confidence[i]:.2f} NEW/SHIFTED" for i in extra], BLUE)
    indices = np.array([b for _, b, _ in metrics["pairs"]], dtype=int)
    labels = [f"#{a+1} {baseline.data['class_name'][a]} {current.confidence[b]:.2f}" for a, b, _ in metrics["pairs"]]
    view = draw(view, current[indices], labels, GREEN)
    return reference, view, metrics
