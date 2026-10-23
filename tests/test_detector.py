from types import SimpleNamespace

import pytest

from breakpoint.detector import normalize_model_id, make_detector, inference_error_message
from breakpoint.experiment import ExperimentConfig


def test_workspace_prefix_is_removed_but_aliases_are_preserved():
    assert normalize_model_id(" lennys-workspace-gabu9/vehicle-detection-wvcx6/1 ") == "vehicle-detection-wvcx6/1"
    assert normalize_model_id("vehicle-detection-wvcx6/1") == "vehicle-detection-wvcx6/1"
    assert normalize_model_id("rfdetr-small") == "rfdetr-small"
    for invalid in ("", "https://universe.roboflow.com/workspace/project/1", "project/no-version", "a/b/c/d"):
        with pytest.raises(ValueError):
            normalize_model_id(invalid)


def test_local_adapter_explicitly_passes_key_and_normalized_model(monkeypatch):
    received = []
    model = SimpleNamespace(infer=lambda image, **kwargs: [SimpleNamespace(model_dump=lambda **kwargs: {"predictions": []})])
    monkeypatch.setattr("breakpoint.detector.local_model", lambda model_id, api_key: received.append((model_id, api_key)) or model)
    config = ExperimentConfig(model_id="workspace/project/1")
    infer = make_detector(config, "test-only-key")
    assert received == [("project/1", "test-only-key")]
    assert infer(None) == {"predictions": []}


def test_error_messages_explain_status_without_echoing_request_secrets():
    error = RuntimeError("sensitive-key-and-request-url")
    error.status_code = 401
    message = inference_error_message(error)
    assert "API key" in message
    assert "sensitive" not in message
    error.status_code = 404
    assert "model" in inference_error_message(error).lower()
