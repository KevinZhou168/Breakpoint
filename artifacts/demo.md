# Breakpoint · experiment report

Research question: How much controlled corruption makes clean-image detections disappear or move?

Run `8bb6c4e29308cee3` · captured 2026-10-22T09:23:34.525031+00:00
Model `rfdetr-small` · backend `local`
Image pixel SHA-256: `5db49b7f00f3580032588f11ec38a94fb2c015b64e9e4cdf9b7236b45a741d16`
Reference detections: 35 at confidence ≥ 0.25.
Class-aware one-to-one matching at IoU ≥ 0.50; failure = mean retention < 50%.

| Perturbation | First sampled crossing bracket | Retention AUC |
|---|---|---|
| Gaussian blur | (0.3, 0.4] | 0.304 |
| Low lighting | (0.5, 0.6] | 0.487 |
| Gaussian noise | Not observed | 0.824 |

## Interpretation and limits

Retention is consistency against the model's own clean detections, not ground-truth recall or mAP. The baseline can be wrong. Unmatched corrupted-image detections are not necessarily false positives.
The bracket only locates the first observed crossing on a finite grid. Recovery at later severities is possible; no monotonicity or continuous boundary is assumed.
Noise bands show the observed range across seeded trials, not a confidence interval. Trials share a noise pattern across severities; clean references are reused.
Severity axes use different physical units; AUC is schedule-dependent and is not a universal ranking of corruption difficulty. One image cannot support dataset-wide conclusions.
Brightness multiplication operates on encoded pixels and is not a physically accurate camera model. Noise is additive, independent per channel, and clipped. Blur preserves geometry.
Raw predictions are captured above a fixed floor; lowering the evaluation threshold below it requires new inference. Confidence scores are not calibrated probabilities.
The JSON includes input pixels, seeds, physical parameters, raw predictions, package versions, and model identity. Regenerating pixels is deterministic under the recorded dependency versions; model aliases, runtime backends, and hosted model revisions may change fresh predictions.
