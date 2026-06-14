# brain-tumor-ssl

Label-efficient, leakage-free, and explainable **4-class brain tumor MRI
classification** (glioma / meningioma / notumor / pituitary).

> ⚠️ **Decision-support research only.** This project is a research artifact
> toward a journal paper. It is **not** a medical device and **must not** be used
> for diagnosis or any clinical decision-making.

## Status

🚧 Under construction — being built in stages. Stage 1 (repo skeleton) is complete.

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) exclusively (never `pip`).

```bash
uv sync          # install all dependencies into .venv
uv run btssl --help
```

More documentation (overview, usage, experiment matrix, disclaimer) is added in
Stage 9.
