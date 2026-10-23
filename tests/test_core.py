import numpy as np
import pytest
import supervision as sv

from breakpoint.core import perturb, compare, first_crossing, physical_value


def det(boxes, names, scores=None):
    return sv.Detections(
        xyxy=np.array(boxes, dtype=float).reshape(-1, 4),
        confidence=np.array(scores if scores is not None else [0.9] * len(boxes)),
        class_id=np.zeros(len(boxes), dtype=int),
        data={"class_name": np.array(names)},
    )


@pytest.mark.parametrize("kind", ["blur", "darkness", "noise"])
def test_zero_severity_is_exact_identity_without_aliasing(kind):
    image = np.arange(300, dtype=np.uint8).reshape(10, 10, 3)
    result = perturb(image, kind, 0, 42)
    assert np.array_equal(result, image)
    assert not np.shares_memory(result, image)


def test_noise_is_seeded_and_shape_preserving():
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    a = perturb(image, "noise", 0.4, 42)
    assert np.array_equal(a, perturb(image, "noise", 0.4, 42))
    assert not np.array_equal(a, perturb(image, "noise", 0.4, 43))
    assert a.dtype == image.dtype and a.shape == image.shape


def test_darkness_is_monotonic_and_physical_units_are_explicit():
    image = np.full((16, 16, 3), 200, dtype=np.uint8)
    assert perturb(image, "darkness", 0.8, 1).mean() < perturb(image, "darkness", 0.2, 1).mean()
    assert physical_value("darkness", 1, image.shape)[0] == pytest.approx(0.001)


@pytest.mark.parametrize("severity", [-0.1, 1.1, float("nan")])
def test_invalid_severity_rejected(severity):
    with pytest.raises(ValueError):
        perturb(np.zeros((8, 8, 3), dtype=np.uint8), "noise", severity, 0)


def test_one_detection_cannot_explain_two_references():
    baseline = det([[0, 0, 10, 10], [1, 0, 11, 10]], ["person", "person"])
    result = compare(baseline, det([[0, 0, 10, 10]], ["person"]), 0.5)
    assert result["retention"] == 0.5
    assert result["matched"] == 1 and result["lost"] == 1


def test_wrong_class_does_not_match_even_when_class_ids_equal():
    result = compare(det([[0, 0, 10, 10]], ["person"]), det([[0, 0, 10, 10]], ["dog"]), 0.5)
    assert result["retention"] == 0
    assert result["unmatched_current"] == 1
    assert result["confidence_mass"] == 0


def test_matching_maximizes_cardinality_before_iou():
    baseline = det([[0, 0, 10, 10], [4, 0, 14, 10]], ["p", "p"])
    current = det([[1, 0, 11, 10], [-3, 0, 7, 10]], ["p", "p"])
    assert compare(baseline, current, 0.5)["matched"] == 2


def test_empty_baseline_is_undefined_and_empty_predictions_are_zero():
    empty = det([], [])
    assert compare(empty, empty, 0.5)["retention"] is None
    result = compare(det([[0, 0, 10, 10]], ["p"]), empty, 0.5)
    assert result["retention"] == 0 and result["confidence_mass"] == 0


def test_boundary_reports_first_observed_bracket_without_monotonic_assumption():
    assert first_crossing([0, 0.2, 0.4, 0.6], [1, 0.8, 0.4, 0.7], 0.5) == (0.2, 0.4)
    assert first_crossing([0, 1], [1, 0.5], 0.5) is None
    assert first_crossing([0, 1], [None, None], 0.5) is None
