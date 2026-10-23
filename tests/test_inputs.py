import json
from pathlib import Path
import numpy as np

from breakpoint.inputs import load_input
from breakpoint.experiment import decode_image


def test_ui_and_cli_input_preparation_match_recorded_experimental_pixels():
    root = Path(__file__).resolve().parents[1]
    bundle = json.loads((root / "artifacts/demo.json").read_text())
    assert np.array_equal(load_input(root / "assets/people-walking.jpg"), decode_image(bundle["image_png"]))
