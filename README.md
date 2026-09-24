# SAGER

This repository contains code for the paper **Selective Text-Anchor Revision for
Multimodal Emotion Recognition in Conversation**.

SAGER selectively revises frozen text predictions using text and audio evidence
through utility-guided routing, consensus verification, and an adoption gate.

## Install

Use Python 3.10+ on Linux, macOS, or WSL. From the repository root, install the
dependencies:

```bash
pip install -r requirements.txt
```

The default training commands use CUDA and require a compatible PyTorch installation.
Use `--device cpu` for CPU execution.

## Quick Start

1. Prepare the frozen feature bundles for IEMOCAP and MELD following
   [Data preparation](docs/data.md). Place `train.rows.npz`, `dev.rows.npz`, and
   `test.rows.npz` in each dataset directory:

   ```text
   features/
   ├── iemocap_legacy7433/
   └── meld_official/
   ```

2. Run the main experiments on both datasets:

   ```bash
   bash run.sh
   ```

   This runs each dataset sequentially with seeds `42`, `19222`, and `831962`.
   To run a single dataset and seed:

   ```bash
   bash run.sh --dataset iemocap_legacy7433 --seeds 42
   ```

   Use `bash run.sh --dry-run` to inspect the commands without starting training.

Training saves `checkpoint.pt` and training records to
`runs/main/<dataset>/seed_<seed>/`. Evaluation verifies matching source code,
configuration, and experiment metadata.
After development-set checkpoint selection, test metrics and predictions are
saved to `test/eval_receipt.json` and `test/predictions.npz` within that directory.
Dataset configurations are in [configs/](configs/).

## Documentation

- [Data preparation and utility targets](docs/data.md)
- [Method and implementation](docs/method.md)
- [Citation metadata](CITATION.cff)

Raw datasets, frozen features, upstream feature extraction, pretrained checkpoints,
and ablation/control pipelines are not included. This revision has been checked
with synthetic inputs; reproduction of the paper's reported benchmark results
has not yet been established for this revision.
