# Trustworthy Contrastive Foundation Models for Retinal Imaging

This repository is a compute-efficient research scaffold for pretraining a retinal fundus image encoder with self-supervised contrastive learning. Disease labels are intentionally excluded from the pretraining interfaces. Downstream evaluation is designed to cover few-shot transfer, external dataset generalization, domain-shift robustness, OOD detection, and image retrieval.

The repository contains dataset auditing, SimCLR pretraining, frozen-encoder evaluation, trustworthiness checks, encoder-only inference, and CPU-safe tests. No datasets are downloaded automatically and no unrun experimental results are presented as completed.

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .\\.venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA, install the appropriate PyTorch and torchvision wheels for the target CUDA version before installing the remaining requirements. The code is intended to use CUDA mixed precision when available, while CPU smoke tests remain supported.

## Repository layout

```text
configs/          YAML experiment configurations
scripts/          audit, training, evaluation, trustworthiness, inference
src/data/         image datasets and self-supervised transforms
src/models/       encoder and projection-head definitions
src/training/     training abstractions; full loop to be added
src/evaluation/   downstream evaluation abstractions
src/utils/        configuration, reproducibility, and device helpers
tests/            unit and smoke tests
```

## Audit datasets and generate manifests

The supplied RFMiD and IDRiD archives can be audited in place:

```bash
python scripts/audit_datasets.py --config configs/audit.yaml
```

This writes `results/audit/audit_report.json`, `image_audit.jsonl`, and deterministic JSONL manifests. The report records archive schemas, image readability, dimensions, color modes, exact/near duplicates, cross-split duplicates, and CSV matching. The pretraining manifest contains only readable RFMiD training images and no label columns. IDRiD testing is emitted only as an external-testing manifest, never as pretraining data.

The current archives reuse numeric RFMiD IDs across splits, so matching is explicitly constrained to the observed corresponding split directory. Filenames are not treated as patient identifiers. The few-shot manifest is intentionally empty and marked `not_ready` until the organizer supplies an approved ID list or patient-level split rule; no hidden-test split is invented.

If the audit has already completed, manifest generation/report refresh can avoid decoding the archives again:

```bash
python scripts/audit_datasets.py --config configs/audit.yaml --reuse-audit
```

## Run tests

From the repository root:

```bash
python -m pytest -q
```

Tests do not download data and use tiny in-memory images for CPU-safe checks.

## SimCLR pretraining

After generating the approved manifest, launch pretraining with:

```bash
python scripts/pretrain.py --config configs/pretrain.yaml
```

The default is a randomly initialized torchvision ResNet-50 with 2,048-dimensional global-average-pooled features and a 128-dimensional two-layer projection head. The training code reads only the RFMiD training JSONL manifest and rejects label-, disease-, diagnosis-, risk-, grade-, or patient-bearing fields. It does not open any ground-truth CSV.

Each sample produces two independently augmented views using conservative crops, horizontal flips, small rotations, mild color jitter, and occasional mild Gaussian blur. Vertical flips and aggressive transformations are intentionally excluded.

Checkpoints are written to `training.output_dir/latest.pt`, with a separate `latest.encoder.pt` containing encoder weights only. The full checkpoint includes optimizer, cosine scheduler, AMP scaler, epoch/step, configuration, and RNG state for resuming via `training.resume`.

ResNet-50 and large batches require substantial memory. Reduce `training.batch_size`, `data.image_size`, or use gradient accumulation when added; mixed precision is enabled automatically only on CUDA. Larger batches provide more in-batch negatives but increase activation memory quadratically in the NT-Xent similarity matrix, so batch size is a throughput/memory trade-off rather than a direct quality guarantee. Training loss, throughput, and peak GPU memory are logged in `training.jsonl`; loss alone must not be used to infer representation quality without downstream transfer, robustness, OOD, and retrieval evaluation.

## Encoder inference

After a real pretraining run has produced `latest.encoder.pt`, export one embedding with:

```bash
python scripts/infer_embedding.py --image path/to/fundus.png --checkpoint checkpoints/pretrain/latest.encoder.pt --output results/example_embedding.json
```

The loader rejects full SimCLR/projection-head checkpoints. It exports the frozen ResNet-50 encoder representation, not the projection head.

## Data and experiment hygiene

Place datasets outside Git-tracked source paths and point to them from a copied YAML configuration. Dataset archives, checkpoints, generated results, and run directories are ignored by `.gitignore`.

See [docs/PROVENANCE.md](docs/PROVENANCE.md), [docs/MODEL_CARD.md](docs/MODEL_CARD.md), [docs/COMPUTE_EFFICIENCY.md](docs/COMPUTE_EFFICIENCY.md), [docs/ABLATIONS.md](docs/ABLATIONS.md), [docs/RESULTS_REPORT.md](docs/RESULTS_REPORT.md), and [docs/SUBMISSION_AUDIT.md](docs/SUBMISSION_AUDIT.md) before submission.

## Frozen-encoder evaluation

Run the evaluation protocol with:

```bash
python scripts/evaluate.py --config configs/evaluation.yaml
```

The evaluator compares a frozen ImageNet ResNet-50, a frozen SimCLR encoder checkpoint, and an optional fine-tuned ImageNet checkpoint. All models use the same manifest rows, label budgets, seeds, classifier hyperparameters, and feature extraction transform. It reports every seed separately in addition to the raw data needed to aggregate mean and standard deviation without hiding failures.

RFMiD is evaluated as multilabel classification using macro-F1, mean per-label balanced accuracy, macro-AUROC over valid classes, and mean average precision. IDRiD uses the official 0–4 retinopathy grade mapping and reports macro-F1, balanced accuracy, macro-AUROC when defined, and quadratic weighted kappa. Missing/constant training classes are recorded rather than silently dropped.

RFMiD validation and final testing are emitted as separate evaluation splits. Classifier settings are fixed in YAML; final-test labels are never used for model selection. IDRiD testing is external evaluation only, and grading is not treated as lesion segmentation. Metrics, per-image predictions, and the resolved configuration are saved under `results/evaluation/`.

Every experiment should record its resolved configuration, seed, software versions, and dataset split before reporting results.

## Trustworthiness suite

Run the frozen-encoder trustworthiness checks with:

```bash
python scripts/trustworthiness.py --config configs/trustworthiness.yaml
```

The suite reports clean-versus-shifted embedding consistency for mild and moderate brightness, contrast, saturation, blur, and resolution changes. It separately scores genuinely unseen approved-manifest data with a documented distance-to-reference OOD score: `1 - max cosine similarity to the in-domain reference gallery`. AUROC/AUPR are emitted only when both groups are non-empty and finite.

It also writes seeded UMAP diagnostics and deterministic cosine nearest-neighbor retrieval records containing query paths, retrieved paths, stable IDs, and similarity scores. UMAP geometry does not prove biological understanding, and synthetic corruption stress tests do not establish clinical reliability.

Outputs are written to `results/trustworthiness/`: `metrics.json`, `retrieval.jsonl`, `umap_in_domain_vs_ood.png`, and `resolved_config.json`. The report template is the JSON structure in `metrics.json`; retain the config and seed with every run.
