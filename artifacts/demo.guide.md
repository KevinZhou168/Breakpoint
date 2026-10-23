# Reading the results CSV

One row per change and severity. Repeated noise trials are averaged.

**Change:** The alteration applied to the original image: blur, low lighting, or noise.

**Severity:** Strength on this change's own scale: 0 is unchanged; 1 is the strongest tested. Equal values across different changes do not mean equal difficulty.

**Original objects:** Detections in the unchanged image that meet the current confidence setting. This is the reference count, not a human-verified count of real objects.

**Objects kept:** Original detections found again with the same label and enough box overlap. For repeated noise trials, this is the average count, so it can be fractional.

**Objects lost:** Original detections without an acceptable match. Includes missing detections, changed labels, low scores, or boxes that moved too far. Averaged across trials.

**Kept (%):** Objects kept divided by original objects, multiplied by 100. Averaged across trials. Blank when there are no original detections. This measures consistency, not accuracy.

**Trials:** Number of noisy versions tested at this strength. Blur and lighting use one version. Severity 0 reuses the original result and is not a separate repeat.

For exact alteration settings, seeds, and individual trial predictions, see experiment.json in the full download.
