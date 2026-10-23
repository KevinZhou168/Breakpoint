"""Run a real experiment, or reanalyze an existing portable JSON bundle."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from breakpoint.detector import configure_runtime
configure_runtime()

from dotenv import load_dotenv
from breakpoint.detector import make_detector, normalize_model_id
from breakpoint.experiment import ExperimentConfig, run_experiment, analyze
from breakpoint.inputs import load_input
from breakpoint.report import metrics_csv, markdown_report, results_guide


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=str(ROOT / "assets/people-walking.jpg"))
    parser.add_argument("--backend", choices=["local", "http"], default="local")
    parser.add_argument("--model", default="rfdetr-small")
    parser.add_argument("--api-url", default="https://serverless.roboflow.com")
    parser.add_argument("--steps", type=int, choices=range(3, 22), default=11)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--cutoff", type=float, default=0.5)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/demo.json")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=True)
    if args.replay:
        bundle = json.loads(args.replay.read_text())
    else:
        try:
            image = load_input(args.image)
        except (ValueError, OSError):
            parser.error("Could not read image; use an image under 20 megapixels")
        config = ExperimentConfig(model_id=normalize_model_id(args.model), backend=args.backend, api_url=args.api_url,
                                  levels=tuple(round(i / (args.steps - 1), 6) for i in range(args.steps)),
                                  noise_trials=args.trials, seed=args.seed)
        infer = make_detector(config, os.getenv("ROBOFLOW_API_KEY"))
        bundle = run_experiment(image, config, infer,
                                lambda n, total: print(f"Inference {n}/{total}", flush=True))
    rows = analyze(bundle, args.confidence, args.iou)
    bundle["evaluation"] = {"confidence": args.confidence, "match_iou": args.iou, "failure_cutoff": args.cutoff}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, indent=2, allow_nan=False))
    args.output.with_suffix(".csv").write_text(metrics_csv(rows))
    args.output.with_suffix(".guide.md").write_text(results_guide())
    report = markdown_report(bundle, rows, args.confidence, args.iou, args.cutoff)
    args.output.with_suffix(".md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
