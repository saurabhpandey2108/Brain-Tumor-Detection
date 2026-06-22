# brain-tumor-ssl

Label-efficient, leakage-free, and explainable **4-class brain tumor MRI
classification** (glioma / meningioma / notumor / pituitary).

This is research toward a journal paper. The focus is three properties that are
often missing from MRI-classification papers:

1. **Label efficiency** — how well the model does with only 1 / 5 / 10 / 25 %
   labelled data, using SimCLR self-supervised pretraining and FixMatch
   semi-supervised fine-tuning.
2. **Leakage-free evaluation** — the public dataset has no patient IDs, so
   near-duplicate slices are grouped with perceptual hashing and kept on one side
   of every split (an approximation of patient-level grouping).
3. **Explainability** — ViT attention-rollout heatmaps over each prediction.

> ⚠️ **Decision-support research only.** This is a research artifact, **not** a
> medical device. It must **not** be used for diagnosis or any clinical
> decision-making. Predictions are unverified and may be wrong.

---

## Build status

The project is being built and verified in stages.

| Stage | Area | State |
|------:|------|:-----:|
| 1 | Repo skeleton, `uv` packaging, tooling | ✅ done |
| 2 | Config layer (pydantic-validated YAML) | ✅ done |
| 3 | Data layer (indexing, splits, transforms, datasets, synthetic) | ✅ done |
| 4 | Models layer (timm ViT backbone, heads, classifier) | ✅ done |
| 5 | Training layer (NT-Xent, SimCLR, supervised + FixMatch, callbacks) | ✅ done |
| 6 | Evaluation — metrics ✅ · attention-rollout explainability ✅ | ✅ done |
| 7 | CLI (`pretrain / finetune / evaluate / run-grid / smoke`) | ✅ done |
| 8 | Streamlit app | ⏳ pending |
| 9 | README / SLURM template / EDA notebook | 🚧 in progress |
| 10 | End-to-end smoke test on synthetic data | ✅ done |

Current verification (everything that exists passes):

```bash
uv run pytest        # 43 passed
uv run ruff check .  # All checks passed!
```

> Note: the `btssl` subcommands now run end to end on CPU. Try
> `uv run btssl smoke` for a full pretrain → finetune → evaluate pass on a
> generated synthetic dataset. HPC/SLURM training templates land in Stage 9.

---

## The method

- **Backbone:** Vision Transformer via [`timm`](https://github.com/huggingface/pytorch-image-models)
  (default `vit_base_patch16_224`).
- **Stage 1 — SimCLR** self-supervised pretraining on the *unlabelled* training
  images (two augmented views, projection head, NT-Xent loss).
- **Stage 2 — fine-tuning**, two selectable modes:
  - `supervised` — labelled fraction only (baseline).
  - `fixmatch` — labelled cross-entropy + consistency loss (confident pseudo-label
    on a weak augmentation supervises the strong augmentation of unlabelled data).

### Experiment matrix

Baselines exist as evidence, not products:

| ID | Method |
|----|--------|
| B1 | Supervised + ImageNet init |
| B2 | Supervised from scratch (no ImageNet) |
| B3 | SSL pretrain + supervised fine-tune |
| B4 | SSL pretrain + FixMatch (**proposed**) |

Each is run across **split strategies** × **label fractions** × **seeds**:

- Splits: `phash` (leakage-free grouped), `naive` (random — used only to *measure*
  leakage inflation), `source` (the dataset's own `Tr-`/`Te-` partition).
- Label fractions: `1%, 5%, 10%, 25%, 100%` (class-balanced selection).
- Seeds: `42, 123, 456`.

Results accumulate to `results/results.csv`, one row per
`(seed, label_fraction, split, method)` with `accuracy`, `macro_f1`, and the run
`config_hash`, so the paper's tables/plots come straight from the CSV.

### MRI-specific rules

Images are grayscale, replicated to 3 channels. **No colour jitter is ever used**
— augmentation is geometry/intensity only (flip, affine, resized-crop, blur,
erasing).

---

## Explainability (attention rollout)

Every prediction can be explained with an **attention-rollout** heatmap
(Abnar & Zuidema, 2020) that shows which image regions drove the ViT's decision.
Lives in `brain_tumor_ssl.evaluation.explain`.

**How it works**

1. timm's fused attention does not expose the attention probabilities, so a
   forward hook on each transformer block recomputes them from that block's own
   `qkv` projection during a **single** forward pass (q/k-norm aware, so it works
   across timm ViT variants).
2. Per block the attention is reduced over heads (`mean` / `max` / `min`), a
   residual **identity** term is added (tokens attend to themselves through the
   skip connection), and rows are renormalised to sum to 1.
3. These per-block matrices are **multiplied across all blocks**; the `[CLS]`
   row of the product (offset by `num_prefix_tokens`) is the saliency of each
   patch.
4. That row is reshaped to the patch grid (`patch_embed.grid_size`), normalised
   to `[0, 1]`, and upsampled over the image as a colour overlay.

An optional `discard_ratio` zeroes the lowest-weight attention entries before
rollout to suppress background noise.

**API**

```python
import torch
from brain_tumor_ssl.evaluation.explain import (
    attention_rollout,   # saliency maps only
    explain,             # predictions + saliency maps (one forward pass)
    overlay_heatmap,     # blend a heatmap onto an image -> RGB uint8 array
    save_explanation,    # render the overlay and write it to disk (PNG)
)

# images: (B, 3, H, W) tensor; clf: a trained Classifier with a ViT backbone
preds, heatmaps = explain(clf, images)            # heatmaps: (B, grid_h, grid_w) in [0, 1]
save_explanation(images[0], heatmaps[0], Path("out/overlay.png"))

# lower-level: heatmaps for an arbitrary ViT backbone, with options
maps = attention_rollout(clf.backbone, images, head_fusion="mean", discard_ratio=0.1)
```

`overlay_heatmap` accepts CHW / HWC / HW images (tensor or array, any scale),
bilinearly upsamples the heatmap to the image resolution, and blends it with a
matplotlib colormap (`alpha`, `colormap` configurable). A non-ViT backbone
raises a clear `ValueError`.

---

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) **exclusively** (never `pip`).
Python 3.11+.

```bash
uv sync                 # resolve + install all dependencies into .venv
uv run btssl version    # confirm the package is installed
uv run pytest           # run the test suite
uv run ruff check .     # lint
```

### Dataset layout

Place the data under `Dataset/` (gitignored), one folder per class. Indexing
recurses, so the nested real-world layout works as-is:

```
Dataset/
├── glioma/      (… nested … )/*.jpg
├── meningioma/  (… nested … )/*.jpg
├── notumor/     (… nested … )/*.jpg
└── pituitary/   (… nested … )/*.jpg
```

Everything is also runnable on CPU against a tiny **synthetic** dataset for smoke
testing (`brain_tumor_ssl.data.synthetic.generate_synthetic_dataset`), so
correctness can be checked before moving to the HPC GPU.

---

## Configuration

All hyperparameters and paths come from YAML — no magic numbers in code. The base
file composes per-concern files; every value is validated by pydantic and fails
fast with a clear message on a bad config.

```
configs/
├── config.yaml       # composition root (includes the files below)
├── data.yaml         # root, classes, image_size, split strategy, phash distance, fractions
├── model.yaml        # backbone, pretrained flag, num_classes
├── ssl.yaml          # SimCLR epochs/lr/batch/temperature/proj dims
├── finetune.yaml     # method, epochs/lr/batch, dropout, early stop, FixMatch mu/threshold/lambda
└── experiment.yaml   # label_fractions, seeds, device, output_dir, workers
```

```python
from brain_tumor_ssl.config import load_config
cfg = load_config()                # validated Config
cfg = load_config(overrides={"finetune": {"method": "supervised"}})  # per-run overrides
```

---

## Running the pipeline

All stages are driven by the `btssl` CLI (thin wrappers over
`brain_tumor_ssl.runner`, which is import-and-call usable from notebooks/the app too):

```bash
# 0. Verify wiring end-to-end on a generated synthetic dataset (CPU, seconds)
uv run btssl smoke

# 1. SimCLR self-supervised pretraining -> SSL checkpoint
uv run btssl pretrain --checkpoint results/checkpoints/simclr.pt

# 2. Fine-tune at one (label_fraction, seed) point and evaluate on the test split
uv run btssl finetune -f 0.1 --seed 42 \
    --ssl-checkpoint results/checkpoints/simclr.pt \
    --output results/checkpoints/clf.pt
# (--method supervised|fixmatch overrides the config; omit --ssl-checkpoint for B1/B2)

# 3. Score a saved checkpoint, optionally writing attention-rollout overlays
uv run btssl evaluate results/checkpoints/clf.pt --explain 8

# 4. Sweep the full seeds x label_fractions grid -> results/results.csv
uv run btssl run-grid --ssl-checkpoint results/checkpoints/simclr.pt
```

Each `finetune`/`run-grid` run appends one row to `results/results.csv`
(`seed, label_fraction, split, method, ssl_init, accuracy, macro_f1, config_hash`).

---

## Project layout

```
src/brain_tumor_ssl/
├── config.py            # pydantic models + load_config()
├── data/                # indexing, splits, transforms, datasets, synthetic
├── models/              # backbone (timm ViT), heads, classifier + SSL transfer
├── training/            # losses (NT-Xent), ssl_simclr, finetune, callbacks
├── evaluation/          # metrics + explain (ViT attention-rollout)
├── utils/               # seed, logging (loguru), device, io
├── runner.py            # end-to-end orchestration (pretrain/finetune/evaluate/grid)
└── cli.py               # typer app (pretrain / finetune / evaluate / run-grid / smoke)
tests/                   # config, splits, transforms, models, metrics, explain, runner
```

---

## Reproducibility

`brain_tumor_ssl.utils.seed.seed_everything(seed)` seeds Python, NumPy and PyTorch.
All experiments are parameterised by an explicit seed, and the config hash is
recorded alongside every result.

---

## Development

```bash
uv run ruff format .          # format
uv run ruff check . --fix     # lint + autofix
uv run pytest                 # tests
uv run pre-commit install     # optional: ruff hooks on commit
```

(See the `Makefile` for the same targets: `sync`, `lint`, `format`, `test`,
`smoke`, `check`.)
