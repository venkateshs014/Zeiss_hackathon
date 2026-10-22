# Model card: Trustworthy Contrastive Foundation Models for Retinal Imaging

## Status

Prototype research repository. No trained checkpoint exists in this workspace as of 2026-10-21. No downstream performance, OOD, retrieval, or clinical result is reported.

## Intended use

Research on frozen retinal fundus representations, few-shot transfer, domain shift, OOD scoring, and image retrieval. The exported target is a torchvision ResNet-50 encoder with 2,048-dimensional global-average-pooled features; the SimCLR projection head is used only during pretraining.

## Not intended for

Clinical diagnosis, screening, triage, treatment decisions, regulatory submission, or patient-level risk estimation. The prototype is not clinically validated.

## Data

RFMiD training images are the approved self-supervised pretraining source. Disease labels are excluded from the pretraining loader. RFMiD validation/testing and IDRiD splits are held for evaluation. See [PROVENANCE.md](PROVENANCE.md).

## Limitations and risks

- The current workspace has no trained weights, so model behavior cannot yet be assessed.
- Dataset camera, population, acquisition, and disease-prevalence differences may cause domain shift.
- Filename identity is not treated as patient identity; patient-level grouping rules were not supplied.
- Synthetic brightness/contrast/blur/resolution tests are not clinical reliability evidence.
- UMAP plots are diagnostic visualizations and do not demonstrate biological understanding.
- A low contrastive training loss alone is not evidence of useful or safe representations.

## Reproducibility

Configurations, seeds, checkpoint RNG state, optimizer/scheduler/scaler state, and encoder-only exports are supported by the code. Strict CUDA determinism is not guaranteed unless the user enables and records the relevant PyTorch deterministic settings for their environment.

