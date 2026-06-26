# Experiment Log — Brain Tumor SSL (SimCLR + FixMatch + ViT)

Running record of the label-efficiency study on a **CPU-only remote desktop**
(`vit_base_patch16_224`, leakage-free `phash` split, 4 classes:
glioma / meningioma / notumor / pituitary). Split sizes: **train 843 / val 120 / test 242**.

> The goal: show that **SimCLR pretraining + FixMatch** keeps accuracy high when
> labels are scarce, beating plain supervised training at the same label budget,
> on an honest (leakage-free) split.

---

## CPU memory fixes (important — needed on this machine)

`vit_base@224` activations are large; the defaults assume a GPU. On CPU these two
changes were required to avoid silent out-of-memory (OOM) kills. Both are
**memory knobs only** — model, resolution, epochs and the split stay full quality.

| Config | Default (GPU) | CPU value | Why |
|---|---|---|---|
| `configs/ssl.yaml` → `batch_size` | 256 | **32** | 256-image forward OOMs `vit_base` on CPU |
| `configs/finetune.yaml` → `fixmatch.mu` | 7 | **2** | unlabelled forward = `batch_size*mu`; 32*7=224 OOMs; 32*2=64 fits |

Set both back to GPU values when a GPU is available.

### Early stopping + crash-safety added to SSL pretrain
`configs/ssl.yaml` now has (code in `src/brain_tumor_ssl/training/ssl_simclr.py`):
- `early_stop_patience: 10` — stop if NT-Xent loss plateaus for 10 epochs (restores best weights)
- `checkpoint_every: 5` — write a crash-safe checkpoint every 5 epochs

---

## Pipeline & commands

PowerShell on the remote. `*> file.log` captures all output (the `uv : ... NativeCommandError`
banner at the top of each log is a harmless PowerShell-stderr quirk, not a crash).
Monitor live with: `Get-Content <log> -Wait -Tail 20`.

**Stage 1 — SimCLR self-supervised pretraining** (uses all 843 train images, no labels):
```powershell
uv run btssl pretrain --checkpoint results/checkpoints/simclr.pt *> pretrain.log
```

**Stage 2 — fine-tune at one label fraction** (FixMatch when fraction < 1.0):
```powershell
# SSL + FixMatch (the method)
uv run btssl finetune --ssl-checkpoint results/checkpoints/simclr.pt --label-fraction 0.10 --output results/checkpoints/clf_f10.pt *> finetune_f10.log
# Supervised-only baseline (no SSL, ImageNet init)
uv run btssl finetune --label-fraction 0.10 --method supervised --output results/checkpoints/clf_sup_f10.pt *> sup_f10.log
```

**Evaluate a saved checkpoint** (per-class report):
```powershell
uv run btssl evaluate results/checkpoints/clf_f10.pt
```

---

## Results so far (seed 42)

| Method | Labels | Test accuracy | Test macro-F1 | Checkpoint |
|---|---|---|---|---|
| Supervised (ImageNet) | 100% (843) | 0.851 | 0.848 | `clf.pt` |
| Supervised (ImageNet) | 10% (84) | 0.612 | 0.599 | `clf_sup_f10.pt` |
| **SSL + FixMatch** | **10% (84)** | **0.698** | **0.685** | `clf_f10.pt` |

**Key finding:** at 10% labels, SSL+FixMatch beats supervised-only by **+8.6 macro-F1
points** (0.685 vs 0.599) — and this is with the handicapped `mu=2`, so it's a
conservative floor. This is the core label-efficiency result.

SSL pretraining (Stage 1): 100 epochs, ~15 h on CPU, best NT-Xent loss 2.7608 @ epoch 93.

---

## TODO — to complete the paper

1. **Fill in the label-efficiency curve** — run both methods at `0.01, 0.05, 0.25`
   (0.10 done, 1.0 supervised done). Command queued below.
2. **Add seeds 123 & 456** at key fractions for mean ± std error bars.
3. Plot accuracy/macro-F1 vs label fraction (two curves) from `results/results.csv`.
4. (Optional) naive-split run to quantify leakage inflation vs `phash`.

### Next command (queues all 6 remaining single-seed runs, ~15–25 h on CPU)
```powershell
foreach ($f in 0.01, 0.05, 0.25) {
  $tag = ($f -replace '\.','')
  uv run btssl finetune --ssl-checkpoint results/checkpoints/simclr.pt --label-fraction $f --output "results/checkpoints/clf_f$tag.pt" *> "finetune_f$tag.log"
  uv run btssl finetune --label-fraction $f --method supervised --output "results/checkpoints/clf_sup_f$tag.pt" *> "sup_f$tag.log"
}
```

All runs append to `results/results.csv` automatically.

---
_Notes: device CPU; class-weighted CE on; val macro-F1 early stopping (patience 15) on finetune._
