# Hackathon submission audit

Date: 2026-10-21 (Asia/Calcutta)

## Split leakage and duplicates

- Pretraining manifest contains only RFMiD training rows.
- RFMiD validation/testing and IDRiD testing are separate manifests.
- IDRiD training is separate from RFMiD pretraining and is used only for designated ordinal adaptation/evaluation protocols.
- CSV matching is split-aware because RFMiD reuses numeric IDs across split folders.
- The audit found one exact duplicate group across IDRiD training/testing. It is reported, not silently resolved.
- Filenames were not interpreted as patient identifiers; patient-level organizer rules remain required.

## Label leakage

- `src/data/contrastive_dataset.py` reads JSONL manifest fields only and rejects keys containing label/disease/grade/risk/diagnosis/patient metadata.
- It enforces `dataset == RFMiD` and `split == training`.
- It never opens ground-truth CSVs.
- The pretraining manifest has only stable ID, dataset, split, relative path, and source archive.

## Checkpoints and reproducibility

- `src/models/checkpoints.py` rejects full SimCLR/projection-head key prefixes when loading encoder exports.
- `scripts/infer_embedding.py` loads the encoder-only export and emits a 2,048-dimensional embedding.
- The training checkpoint stores model, optimizer, scheduler, AMP scaler, seed/RNG state, epoch, step, and config.
- No checkpoint is currently present, so encoder export loading has been tested synthetically but not against a completed experiment.
- CUDA strict determinism remains environment-dependent and must be recorded for real runs.

## Test status

The last local run passed 17 tests. There is one sklearn deprecation warning for `multi_class`; it is non-fatal and should be cleaned up before a final release.

