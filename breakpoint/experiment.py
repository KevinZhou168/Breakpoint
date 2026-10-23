"""A serial, bounded sweep with a portable record of every real prediction."""
import base64
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import platform
from time import perf_counter

import cv2
import numpy as np

from .core import KINDS, perturb, physical_value, detections, compare

SCHEMA = "breakpoint/v1"


@dataclass(frozen=True)
class ExperimentConfig:
    model_id: str = "rfdetr-small"
    backend: str = "local"
    api_url: str = "https://serverless.roboflow.com"
    levels: tuple = tuple(round(i / 10, 2) for i in range(11))
    noise_trials: int = 3
    seed: int = 42
    capture_confidence: float = 0.05
    nms_iou: float = 0.5

    def __post_init__(self):
        if (len(self.levels) < 2 or self.levels[0] != 0 or self.levels[-1] != 1
                or any(not np.isfinite(x) or not 0 <= x <= 1 for x in self.levels)
                or any(a >= b for a, b in zip(self.levels, self.levels[1:]))):
            raise ValueError("Levels must increase strictly from 0 to 1")
        if not 1 <= self.noise_trials <= 10 or not 0 <= self.seed <= 2**32 - 11:
            raise ValueError("Invalid trial count or random seed")
        if not 0 < self.capture_confidence <= 1 or not 0 < self.nms_iou <= 1:
            raise ValueError("Invalid inference thresholds")
        if self.backend not in ("local", "http"):
            raise ValueError("Backend must be local or http")


def encode_image(image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode image")
    return base64.b64encode(encoded).decode("ascii")


def decode_image(encoded):
    image = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image")
    return image


def image_hash(image):
    return hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()


def package_versions():
    result = {"python": platform.python_version()}
    for name in ("inference", "inference-models", "inference-sdk", "supervision", "numpy", "opencv-python", "scipy"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            pass
    return result


def run_experiment(image, config, infer, progress=None):
    """One reference request; independent corruptions of original, never chained.

    A trial uses the same standard-normal noise field across severity levels
    (common random numbers). Different trials use independent seeds.
    """
    config_dict = asdict(config)
    digest = image_hash(image)
    run_id = hashlib.sha256((digest + json.dumps(config_dict, sort_keys=True)).encode()).hexdigest()[:16]
    total = 1 + (len(config.levels) - 1) * (2 + config.noise_trials)
    start = perf_counter()
    baseline = infer(image)
    # Fail immediately for responses incompatible with object detection.
    detections(baseline, config.capture_confidence)
    baseline_ms = (perf_counter() - start) * 1000
    completed = 1
    if progress:
        progress(completed, total)
    records = []
    for kind in KINDS:
        for severity in config.levels:
            trials = config.noise_trials if kind == "noise" and severity > 0 else 1
            for trial in range(trials):
                seed = config.seed + trial
                if severity == 0:
                    raw, elapsed_ms = baseline, baseline_ms
                else:
                    transformed = perturb(image, kind, severity, seed)
                    tick = perf_counter()
                    raw = infer(transformed)
                    detections(raw, config.capture_confidence)
                    elapsed_ms = (perf_counter() - tick) * 1000
                    completed += 1
                    if progress:
                        progress(completed, total)
                value, unit = physical_value(kind, severity, image.shape)
                records.append({"kind": kind, "severity": severity, "trial": trial,
                                "seed": seed, "physical_value": value, "unit": unit,
                                "elapsed_ms": elapsed_ms, "raw": raw})
    return {"schema": SCHEMA, "run_id": run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(), "config": config_dict,
            "image_sha256": digest, "image_shape": list(image.shape),
            "image_png": encode_image(image), "baseline": baseline,
            "records": records, "versions": package_versions(),
            "inference_calls": completed, "duration_seconds": perf_counter() - start}


def analyze(bundle, confidence=0.25, iou=0.5):
    if not bundle["config"]["capture_confidence"] <= confidence <= 1:
        raise ValueError("Evaluation threshold cannot be below the captured confidence floor")
    baseline = detections(bundle["baseline"], confidence)
    rows = []
    for record in bundle["records"]:
        current = detections(record["raw"], confidence)
        metrics = compare(baseline, current, iou)
        rows.append({**{k: v for k, v in record.items() if k != "raw"}, **metrics})
    return rows
