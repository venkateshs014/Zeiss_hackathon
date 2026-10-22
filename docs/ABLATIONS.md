# Ablation table

This table intentionally contains no invented values. The repository contains no completed pretraining/evaluation experiments or checkpoints.

| Comparison | Dataset/split | Seeds | Metric | Result | Status |
|---|---|---:|---|---|---|
| Frozen ImageNet ResNet-50 vs frozen SimCLR ResNet-50 | RFMiD validation/test; IDRiD test | — | Macro-F1, balanced accuracy, AUROC/mAP, QWK | Not available | Evaluation not run |
| Generic augmentations vs retinal-specific augmentations | RFMiD training/validation | — | Same downstream protocol | Not available | Ablation not implemented as a completed experiment |
| SimCLR label budgets: 1%, 5%, 10%, 100% | RFMiD/IDRiD designated splits | — | Mean ± SD across seeds | Not available | Evaluation not run |

The code and configurations define the intended protocol; they are not evidence that the experiments succeeded.

