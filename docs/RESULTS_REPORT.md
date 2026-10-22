# Results report

## Executive summary

No model-training or downstream evaluation results are available in the current workspace. There is no checkpoint under `checkpoints/`, no `results/evaluation/`, and no `results/trustworthiness/` output.

## Completed artifact: dataset audit

- RFMiD: 3,200/3,200 images readable; 3,200 RGB images.
- IDRiD: 516/516 images readable; 516 RGB images at 4288×2848.
- CSV matching: all observed split-specific rows matched without missing or ambiguous matches.
- Cross-split duplicate audit: one exact duplicate group was detected across IDRiD training/testing; this is flagged in the audit report and has not been silently removed.
- Pretraining manifest: 1,920 RFMiD training rows, with no label fields.

## Seed variability

Not available because no training/evaluation run completed. Once run, report every seed, then mean ± sample standard deviation, plus the number of valid seeds and any absent-class exclusions.

## Submission interpretation

This is a reproducible implementation and audit scaffold, not evidence of clinical utility. The prototype is not clinically validated.

