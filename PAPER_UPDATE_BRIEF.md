# Paper Update Brief — for the editing assistant

**Target file:** `Brain_Tumor_SSL_IEEE_paper.pdf` (and its LaTeX source, wherever it
lives — ask the author for the `.tex`; the PDF is compiled output).

**Read this whole brief before editing.** The paper is a well-structured *scaffold*:
the methodology, the pHGS protocol, and the RQ framing are solid and should be
**preserved**. The job is to (a) replace placeholders with **real** numbers, (b) fix
places where the paper **describes a setup that was not the one actually run**, and
(c) flag/soften claims that the current experiments do not yet support.

> ⛔ **Do not invent numbers.** Every numeric cell must come from `results/results.csv`
> (the authoritative one on the training machine) or from the run logs quoted below.
> Where no run exists yet, **leave the placeholder and mark it `TODO (run pending)`** —
> do not guess.

---

## 0. Source of truth

1. `results/results.csv` on the **training (remote) machine** — authoritative for all
   metrics. (The copy in this local checkout is stale: it holds only an old
   `naive/100%/supervised` row. Get the remote one.)
2. `EXPERIMENTS.md` (repo root) — run log, CPU fixes, commands.
3. The run logs (`pretrain.log`, `finetune_f10.log`, `sup_f10.log`, etc.).
4. The actual config files (`configs/*.yaml`) — authoritative for hyperparameters.

---

## 1. Real results available so far (seed 42, pHGS/leakage-free)

Compiled from the run logs. **macro-F1 and accuracy are fractions ×100 for the paper.**

| Method | Label budget | Test accuracy | Test macro-F1 | Source |
|---|---|---|---|---|
| B1 — Supervised + ImageNet | 100% (843 lbl) | 85.12 | 84.82 | first finetune run |
| B1 — Supervised + ImageNet | 10% (84 lbl) | 61.16 | 59.92 | `sup_f10.log` |
| B4 — SF-ViT (SSL+FixMatch) | 10% (84 lbl) | 69.83 | 68.47 | `finetune_f10.log` |

**Split sizes (pHGS, this dataset):** train 843 / val 120 / test 242 (total 1,205).
**SSL pretraining:** 100 epochs, best NT-Xent loss 2.7608 @ epoch 93 (~15 h on CPU).

**Headline finding (already supportable):** at 10% labels, SF-ViT beats the
same-budget supervised baseline by **+8.55 macro-F1 points** (68.47 vs 59.92). This is
the strongest, cleanest claim currently backed by data — use it.

**A 6-run batch is in progress** that will add, for seed 42, pHGS: B1 and B4 at
fractions 1%, 5%, 25%. Pull those from `results.csv` when done.

---

## 2. CRITICAL discrepancies to fix (integrity issues — do these first)

### 2.1 Dataset size (Section III, Abstract)
- Paper says **7,023 images**. Actual experiments use **1,205** images
  (glioma 240, meningioma 306, notumor 405, pituitary 254).
- **Action (pick one, ask author):**
  - (a) If the subset is intentional: change all dataset-size statements to the real
    counts, and add one sentence in Section III + Limitations: *"For compute reasons we
    use a class-balanced subset of N=1,205 images from the benchmark."* Update Table I
    note accordingly.
  - (b) If the full set should be used: author must download the full Kaggle dataset and
    re-run; keep 7,023 only after that.
- ⚠️ This number propagates: Section III paragraph 1, Table I, and any "7,023" mention.

### 2.2 Hyperparameter table (Table III) does not match the configs actually run
Correct Table III to the **real CPU values** (from `configs/ssl.yaml`,
`configs/finetune.yaml`):

| Stage | Param | Paper says | **Actual (correct to this)** |
|---|---|---|---|
| SimCLR | batch size | 256 | **32** |
| SimCLR | (add) early stop | — | patience 10 on NT-Xent loss; checkpoint every 5 ep |
| Fine-tune | epochs | 50 | **80** (cap; early stop) |
| Fine-tune | labelled batch size | 64 | **32** |
| Fine-tune | (add) dropout | — | **0.1** (classifier head) |
| Fine-tune | (add) class weighting | — | **inverse-frequency class-weighted CE** |
| Fine-tune | early-stop patience | 10 (also in §V-C) | **15** |
| FixMatch | ratio µ | 7 | **2** |

- Also fix **Section V-C** ("early stopping with patience 10" → **15**).
- Also fix **Section IV-D**: the supervised/FixMatch losses are described as plain CE,
  but training uses **inverse-frequency class-weighted** cross-entropy on the labelled
  term. Add one clause.

### 2.3 The µ=2 disclosure (Method + Discussion + Limitations) — important for honesty
Standard FixMatch uses µ=7; this work used **µ=2** because µ=7 (unlabelled batch
32×7=224) exceeds CPU RAM for ViT-B/16. **Do not hide this.** Add to Limitations:
*"Due to CPU memory limits we set the FixMatch unlabelled:labelled ratio to µ=2 rather
than the customary 7; the reported SF-ViT numbers are therefore a conservative lower
bound on what the method achieves with a larger ratio on GPU hardware."*

### 2.4 Hardware / compute (Section V-D) is blank and assumes GPU
- Current text: *"Hardware: – (e.g. single NVIDIA – GPU); … Mixed precision is enabled
  on GPU."*
- Reality: **runs were CPU-only.** Fill honestly, e.g.: *"All reported runs were executed
  on CPU (no GPU available); mixed precision and the GPU batch sizes in the original
  recipe were therefore not used, and SimCLR/FixMatch batch sizes and µ were reduced to
  fit RAM (Table III). SimCLR pretraining took ≈15 h; each fine-tuning run ≈3–5 h."*
- Keep the AMP sentence only as a conditional ("AMP is used when a GPU is present").

---

## 3. Experimental-scope gap (decide before claiming the full design)

The paper (Contributions, §V-B, Tables IV/V, Figs 2/3) promises a grid of
**4 methods × 3 splits × 5 label budgets × 3 seeds**. What actually exists right now:

| Dimension | Promised | Done | Pending |
|---|---|---|---|
| Methods | B1, B2, B3, B4 | **B1, B4** | **B2 (scratch), B3 (SSL+sup)** |
| Splits | pHGS, naive, source | **pHGS only** | **naive, source** |
| Label budgets | 1/5/10/25/100% | 10% (+100% for B1); 1/5/25% running | finish remaining |
| Seeds | 42, 123, 456 | **42 only** | **123, 456** |

**Consequences the editor must respect:**
- **RQ2 & RQ3 (leakage inflation, Tables V & Fig 3) have NO data** — they need
  `naive`-split runs, which have not been done. **Leave those tables/figures as
  `TODO (run pending)`; do not fabricate ∆ values.** These are billed as the *novel*
  contributions, so flag prominently to the author that naive-split runs are required.
- **Table IV** can only be partially filled (B1, B4 rows at the fractions that exist;
  B2, B3 rows pending).
- **Soften the seed claim:** until 3 seeds are run, change "mean±std over seeds
  {42,123,456}" to "seed 42 (multi-seed runs in progress)" wherever results are
  reported, OR report single-seed numbers without ± and note it in Limitations.

**Recommendation to author (state this in a note, don't decide unilaterally):** given
CPU compute, either (i) run the missing cells (B2, B3, naive split, seeds) — large time
cost, or (ii) reduce the paper's stated scope to what is feasible (e.g. B1 vs B4, pHGS
vs naive, 1–100% labels, seed 42 + one more). The paper's narrative survives either way,
but the claimed design must match what was actually executed.

---

## 4. Placeholders to replace (mechanical, once numbers exist)

1. **Abstract** — fill the dashes: *"SF-ViT attains **68.5%** macro-F1 using only 10% of
   labels — within **~16** points of the fully-supervised upper bound (**84.8%**) — while
   the naive split inflates accuracy by **[TODO]** points, an inflation that **[TODO]** as
   labels become scarcer."* (Leave the naive-split parts as TODO until those runs exist.)
2. **Author block (page 1)** — fill Department/Institution, City, Country, Email.
3. **Section VI red "Draft placeholder" banner** — remove once any real cell is added.
4. **Table IV** — populate from `results.csv`; keep `–` only where runs are pending.
5. **Fig. 2 (label-efficiency curves)** — regenerate from `results.csv`; remove the
   "illustrative placeholder" watermark.
6. **Table V & Fig. 3 (leakage inflation)** — `TODO (naive runs pending)`; keep watermark.
7. **Fig. 4 confusion matrix** — current numbers are fake (they sum to >1,300 but the test
   set is 242). Regenerate from a trained checkpoint via
   `uv run btssl evaluate results/checkpoints/clf_f10.pt` (or the best B4 checkpoint).
   Per-class attention-rollout overlays: render real ones (`--explain N`).
8. **Discussion (§VII)** is written conditionally ("If SF-ViT retains…"). Once Table IV
   is populated, rewrite RQ1 with the actual gap (e.g. "+8.6 macro-F1 over B1 at 10%
   labels"). Keep RQ2/RQ3 conditional until naive data exists.

---

## 5. Expert framing advice (quality, not just correctness)

- **Lead with the right number.** "Within ~16 points of the 100% upper bound" is modest;
  the compelling, defensible claim is the **+8.6 macro-F1 gain over the same-budget
  supervised baseline (B1) at 10% labels.** Make that the headline of §VI-A and the
  abstract.
- **B3 vs B4 is the key ablation** (does FixMatch add value on top of SSL features?). It
  cannot be made until B3 is run — note it as the most scientifically valuable missing
  run, ahead of extra seeds.
- **At 100% labels B4 ≡ B3** (no unlabelled data; FixMatch reduces to supervised). State
  this so readers don't expect a B3/B4 gap at 100%.
- **Verify the augmentation list** in §IV-F against `src/brain_tumor_ssl/data/transforms.py`
  before claiming the exact set (flip/affine/resized-crop/blur/erasing, no colour jitter).
- **Leakage-direction sanity check:** RQ2 predicts naive > pHGS accuracy. When the naive
  runs come in, verify the sign. If on this small subset the inflation is small or
  reversed, report it honestly — that is still a finding, but the abstract/claims must
  match it. Do not assume the hypothesised direction.
- **Reproducibility:** every `results.csv` row carries a `config_hash`; mention that the
  reported cells map to specific hashes (good for the reproducibility statement).
- **References [22], [23]** are incomplete (no authors). Complete them.

---

## 6. Suggested edit order

1. Fix integrity issues (§2 here): dataset N, Table III hyperparameters, µ=2 disclosure,
   hardware/CPU. These don't need new runs and must be right.
2. Reconcile scope (§3): adjust seed/method/split claims to reality; mark RQ2/RQ3 pending.
3. Fill the cells that have data (§1, §4): abstract 10% numbers, Table IV partial, Fig. 2,
   confusion matrix from a real checkpoint.
4. Rewrite §VI-A / §VII-RQ1 around the +8.6-point result.
5. Leave clearly-marked TODOs for everything still running (naive split, B2/B3, seeds).

---

## 7. One-line summary for the editor
Preserve the strong scaffold; correct the four setup mismatches (dataset N=1,205, batch
sizes, µ=2, CPU); fill only the cells backed by `results.csv`; mark leakage-inflation
(RQ2/RQ3), B2/B3, and multi-seed results as run-pending; and lead with the real,
defensible headline: **+8.6 macro-F1 over the supervised baseline at 10% labels under a
leakage-free split.**
