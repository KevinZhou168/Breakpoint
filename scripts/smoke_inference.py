"""Validate the real native Roboflow path before investing in UI."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("ROBOFLOW_CACHE_DIR", str(ROOT / ".cache" / "roboflow"))
os.environ.setdefault("MODEL_CACHE_DIR", str(ROOT / ".cache" / "models"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))

import cv2
import supervision as sv
from inference import get_model

model = get_model(model_id="rfdetr-small")
image = cv2.imread(str(ROOT / "assets" / "people-walking.jpg"))
raw = model.infer(image, confidence=0.05, iou_threshold=0.5)[0].model_dump(by_alias=True, mode="json")
detections = sv.Detections.from_inference(raw)
print(json.dumps({"model": "rfdetr-small", "detections": len(detections), "classes": list(set(detections.data["class_name"]))}))
(ROOT / "artifacts" / "smoke.json").write_text(json.dumps(raw, indent=2))
