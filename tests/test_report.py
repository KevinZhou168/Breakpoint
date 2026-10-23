import io
import json
from pathlib import Path

import pandas as pd
import pytest

from breakpoint.experiment import analyze
from breakpoint.report import results_table, metrics_csv, RESULT_HELP


def test_simple_results_keep_counts_percentages_and_trial_mean():
    bundle = json.loads((Path(__file__).resolve().parents[1] / "artifacts/demo.json").read_text())
    rows = analyze(bundle)
    table = results_table(rows)
    assert list(table.columns) == list(RESULT_HELP)
    assert len(table) == 33
    blur = table[(table.Change == "Gaussian blur") & (table.Severity == 0.4)].iloc[0]
    assert blur["Original objects"] == 35
    assert blur["Objects kept"] == 16
    assert blur["Objects lost"] == 19
    assert blur["Kept (%)"] == pytest.approx(100 * 16 / 35)
    noise = table[(table.Change == "Gaussian noise") & (table.Severity == 1)].iloc[0]
    assert noise.Trials == 3
    assert noise["Objects kept"] == pytest.approx(27)
    assert list(pd.read_csv(io.StringIO(metrics_csv(rows))).columns) == list(RESULT_HELP)


def test_simple_results_leave_undefined_percentage_blank():
    bundle = json.loads((Path(__file__).resolve().parents[1] / "artifacts/demo.json").read_text())
    table = results_table(analyze(bundle, 0.95))
    assert table["Kept (%)"].isna().all()
