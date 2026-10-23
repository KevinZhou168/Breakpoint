"""Exercise the actual Streamlit script without making inference calls."""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def app():
    result = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not result.exception
    return result


def test_recorded_demo_and_identity_control(app):
    assert app.metric[3].value == "35"
    assert app.metric[4].value == "46%"
    app.select_slider[0].set_value(0.0).run()
    assert not app.exception
    assert app.metric[4].value == "100%"
    assert app.metric[5].value == "0"


def test_noise_trials_and_threshold_reanalysis(app):
    app.selectbox[0].set_value("noise").run()
    app.select_slider[0].set_value(0.8).run()
    assert len(app.selectbox[1].options) == 3
    app.selectbox[1].set_value(2).run()
    app.slider[0].set_value(0.95).run()
    assert not app.exception
    assert app.metric[3].value == "0"
    assert app.metric[4].value == "Undefined"
    assert any("No clean detections" in message.value for message in app.warning)


def test_live_form_does_not_run_inference_until_clicked(app):
    app.radio[0].set_value("Run live inference").run()
    assert not app.exception
    assert app.button[0].label == "Run robustness experiment"
    assert any("Choose an image" in message.value for message in app.info)


def test_results_page_has_readable_columns_and_optional_explanations(app):
    from breakpoint.report import RESULT_HELP
    assert list(app.dataframe[0].value.columns) == list(RESULT_HELP)
    assert len(app.dataframe[0].value) == 33
    assert any(e.label == "What does each column mean?" for e in app.expander)
    assert any(e.label == "Technical settings and Roboflow tools" for e in app.expander)
    assert any("do not tell us whether" in message.value for message in app.info)
