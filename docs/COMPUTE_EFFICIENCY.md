# Compute and efficiency report

## Completed measurements

No pretraining or downstream evaluation run was completed in this workspace. Consequently, there are no measured epoch times, throughput, peak GPU memory, energy, or cost claims.

The environment inspection recorded Python 3.12.7, CPU-only PyTorch 2.8.0+cpu, and `torch.cuda.is_available() == False`. The audit decoded the supplied archives but is not a model-training benchmark.

## Configured design

- Default encoder: ResNet-50, 2,048-dimensional output.
- Default input: 224×224.
- Default batch size: 64.
- CUDA AMP: enabled only when a CUDA device is available.
- Checkpoint: complete resumable state plus separate encoder-only weights.
- NT-Xent similarity matrix: O((2B)^2) similarity memory, so larger batches improve negative coverage but increase memory quadratically.

After a real run, append hardware, software versions, batch size, image size, epochs, wall time, images/s, peak GPU memory, and checkpoint size from `training.jsonl`. Do not substitute estimates for measurements.

