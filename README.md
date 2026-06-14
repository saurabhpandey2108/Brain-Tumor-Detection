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
| 6 | Evaluation — metrics ✅ · attention-rollout explainability ⏳ | 🚧 partial |
| 7 | CLI (`pretrain / finetune / evaluate / run-grid / smoke`) | ⏳ pending |
| 8 | Streamlit app | ⏳ pending |
| 9 | README / SLURM template / EDA notebook | 🚧 in progress |
| 10 | End-to-end smoke test on synthetic data | ⏳ pending |

Current verification (everything that exists passes):

```bash
uv run pytest        # 34 passed
uv run ruff check .  # All checks passed!
```

> Note: `uv run btssl <pretrain|finetune|evaluate|run-grid|smoke>` and HPC training
> are **not available yet** — they land in Stage 7+. Only `uv run btssl version`
> works today.

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

## Project layout

```
src/brain_tumor_ssl/
├── config.py            # pydantic models + load_config()
├── data/                # indexing, splits, transforms, datasets, synthetic
├── models/              # backbone (timm ViT), heads, classifier + SSL transfer
├── training/            # losses (NT-Xent), ssl_simclr, finetune, callbacks
├── evaluation/          # metrics  (+ explain — pending)
├── utils/               # seed, logging (loguru), device, io
└── cli.py               # typer app (subcommands pending)
tests/                   # config, splits, transforms, models, metrics
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
