"""Roboflow adapters. Import native inference only when local inference is used."""
from functools import lru_cache
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configure_runtime():
    # Keep all model/font caches inside the project, including in restricted shells.
    for name, directory in {"MPLCONFIGDIR": "matplotlib", "HF_HOME": "huggingface",
                            "TORCH_HOME": "torch", "MODEL_CACHE_DIR": "models",
                            "INFERENCE_HOME": "models"}.items():
        os.environ.setdefault(name, str(ROOT / ".cache" / directory))
    for name in ("CORE_MODEL_GAZE_ENABLED", "CORE_MODEL_SAM_ENABLED", "CORE_MODEL_SAM3_ENABLED",
                 "CORE_MODEL_YOLO_WORLD_ENABLED"):
        os.environ.setdefault(name, "False")


def normalize_model_id(model_id):
    model_id = model_id.strip()
    parts = model_id.split("/")
    # Universe URLs include a workspace; inference IDs use only project/version.
    if len(parts) == 3 and all(re.fullmatch(r"[\w-]+", part) for part in parts[:2]) and parts[-1].isdigit():
        return "/".join(parts[1:])
    if re.fullmatch(r"[\w-]+(?:/[0-9]+)?", model_id):
        return model_id
    raise ValueError("Use a pretrained model name such as rfdetr-small, or a project/version ID such as vehicle-detection-wvcx6/1.")


def inference_error_message(error):
    """Actionable diagnostics without exposing SDK request URLs or credentials."""
    name = type(error).__name__
    status = getattr(error, "status_code", None) or getattr(getattr(error, "response", None), "status_code", None)
    if name == "InvalidModelIdentifier":
        return "The model ID format was rejected. Use project/version, without the workspace prefix or website URL."
    if name == "APIKeyNotProvided" or status in (401, 403):
        return "Roboflow could not authorize this request. Check your API key (ROBOFLOW_API_KEY in .env) and that it can access the model."
    if status == 404:
        return "The model or endpoint was not found. Check the project/version ID and the selected inference server."
    if status in (402, 429):
        return "Roboflow reported an account or usage limit. Check credits and request limits, or retry later."
    if status and status >= 500:
        return "The inference server returned an error. Retry shortly and check that the model supports this server."
    if name in ("ConnectionError", "ConnectTimeout", "ReadTimeout", "Timeout", "HTTPClientError"):
        return "Could not complete the connection to Roboflow. Check the network and inference server URL, then retry."
    return f"Inference failed ({name}). Check model access, network connectivity, and installed model dependencies. No failed request is counted as an empty detection."


@lru_cache(maxsize=2)
def local_model(model_id, api_key=None):
    configure_runtime()
    from inference import get_model
    return get_model(model_id=model_id, api_key=api_key)


def make_detector(config, api_key=None):
    model_id = normalize_model_id(config.model_id)
    api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
    if config.backend == "local":
        model = local_model(model_id, api_key)

        def infer(image):
            result = model.infer(image, confidence=config.capture_confidence, iou_threshold=config.nms_iou)[0]
            return result.model_dump(by_alias=True, mode="json")
        return infer

    from inference_sdk import InferenceHTTPClient, InferenceConfiguration
    client = InferenceHTTPClient(api_url=config.api_url, api_key=api_key).configure(
        InferenceConfiguration(confidence_threshold=config.capture_confidence,
                               iou_threshold=config.nms_iou, api_key_transport="header",
                               disable_active_learning=True, source="breakpoint"))

    def infer(image):
        return client.infer(image, model_id=model_id)
    return infer
