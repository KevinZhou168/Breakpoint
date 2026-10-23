import numpy as np
import pytest

from breakpoint.experiment import ExperimentConfig, run_experiment, analyze, decode_image, encode_image


def raw(score=0.9):
    return {"predictions": [{"x": 5, "y": 5, "width": 8, "height": 8,
                             "confidence": score, "class": "person", "class_id": 0}],
            "image": {"width": 20, "height": 20}}


def test_run_reuses_clean_reference_and_exports_reproducible_pixels():
    calls = []
    def infer(image):
        calls.append(image.copy())
        return raw()
    image = np.full((20, 20, 3), 150, dtype=np.uint8)
    config = ExperimentConfig(levels=(0, 0.5, 1), noise_trials=2)
    bundle = run_experiment(image, config, infer)
    # 1 baseline + 2 blur + 2 darkness + 2*2 noise
    assert len(calls) == 9
    assert np.array_equal(decode_image(bundle["image_png"]), image)
    assert len(bundle["records"]) == 11
    rows = analyze(bundle, 0.25, 0.5)
    assert all(r["retention"] == 1 for r in rows)
    assert bundle["config"]["noise_trials"] == 2


def test_threshold_reanalysis_does_not_run_inference_and_handles_no_reference():
    bundle = run_experiment(np.zeros((20, 20, 3), dtype=np.uint8),
                            ExperimentConfig(levels=(0, 1), noise_trials=1), lambda _: raw(0.3))
    assert analyze(bundle, 0.25, 0.5)[0]["reference_count"] == 1
    assert all(r["retention"] is None for r in analyze(bundle, 0.4, 0.5))
    with pytest.raises(ValueError):
        analyze(bundle, 0.01, 0.5)  # Cannot recover predictions below capture floor


def test_errors_are_not_converted_to_empty_predictions():
    def broken(_):
        raise RuntimeError("network down")
    with pytest.raises(RuntimeError):
        run_experiment(np.zeros((20, 20, 3), dtype=np.uint8), ExperimentConfig(), broken)


@pytest.mark.parametrize("kwargs", [{"levels": (0.2, 1)}, {"levels": (0, 0.5, 0.5)},
                                   {"noise_trials": 0}, {"seed": -1}, {"capture_confidence": 2}])
def test_invalid_configs_rejected(kwargs):
    with pytest.raises(ValueError):
        ExperimentConfig(**kwargs)


def test_png_roundtrip_is_lossless():
    image = np.random.default_rng(9).integers(0, 256, (20, 30, 3), dtype=np.uint8)
    assert np.array_equal(decode_image(encode_image(image)), image)
