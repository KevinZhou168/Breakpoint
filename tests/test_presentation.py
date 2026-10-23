from breakpoint.presentation import boundary_display, status_matrix


def test_boundary_explains_real_endpoints_and_no_crossing():
    shape = (540, 960, 3)
    blur = boundary_display("blur", (0.3, 0.4), shape, True)
    assert "5.67" in blur["value"] and "7.56" in blur["value"]
    assert "0.30" in blur["detail"] and "0.40" in blur["detail"]
    dark = boundary_display("darkness", (0.5, 0.6), shape, True)
    assert "3.16%" in dark["value"] and "1.58%" in dark["value"]
    noise = boundary_display("noise", None, shape, True)
    assert noise["value"] == "Not observed"
    assert "100" in noise["detail"]
    assert boundary_display("blur", None, shape, False)["value"] == "Undefined"


def row(level, trial, kept):
    return {"severity": level, "trial": trial, "pairs": [(i, i, 1.0) for i in kept]}


def test_matrix_uses_binary_categories_and_independent_matching_counts():
    rows = [row(0, 0, [0, 1]), row(1, 0, [0])]
    fig = status_matrix(rows, [0, 1], ["A", "B"], "blur", (20, 20, 3))
    assert {trace.name for trace in fig.data} == {"Retained", "Lost"}
    assert all(trace.type == "scatter" for trace in fig.data)
    assert sum(len([x for x in trace.x if x is not None]) for trace in fig.data) == 4


def test_noise_matrix_shows_trial_counts_and_identifies_reused_control():
    rows = [row(0, 0, [0, 1]), row(1, 0, [0, 1]), row(1, 1, [0]), row(1, 2, [])]
    fig = status_matrix(rows, [0, 1], ["A", "B"], "noise", (20, 20, 3))
    assert {t.name for t in fig.data} == {"Original control", "0/3 trials", "1/3 trials", "2/3 trials", "3/3 trials"}
    assert any("2/3" in str(t.customdata) for t in fig.data)
    assert any("1/3" in str(t.customdata) for t in fig.data)
    assert all(t.type == "scatter" for t in fig.data)
