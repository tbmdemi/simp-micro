# Results — Draft for Paper #1

> **Working title:** Inverse design of continuous auxetic metamaterials via WIRE INR and KAN.
> **Dependencies:** `docs/paper1_methods_draft.md`, `docs/paper1_discussion_draft.md`,
> `task_progress.md` (all numbers verified, 2026-08-23).
> **Status:** draft for review. All R² figures are re-measured with the current evaluation
> pipeline (`property_accuracy`, frozen surrogate v2, same test set) unless stated otherwise.

---

## Results

### Reproducing the historical baseline changed the comparison frame

We first re-measured the previously reported linear-baseline performance with the current,
version-pinned evaluation pipeline. The July evaluation report claimed R² = 0.8575/0.8492
(ν₁₂/ν₂₁) for the fine-tuned linear checkpoint `cvae_v2_finetuned.pt`. Re-measuring the
*identical checkpoint* on the *identical test set* yields **R² = 0.35** (macro) with the current
surrogate v2, and −1.81 with the surrogate version used by the original report; we could not
trace the discrepancy to a functional change in either the evaluation routine or the forward
pass. The two linear baselines used throughout this paper are therefore the re-measured values:
**0.29** (base) and **0.35** (fine-tuned) macro R². This establishes a strict, reproducible
reference frame for all architecture comparisons below.

### Reconstruction quality does not imply property fidelity

We trained three KAN-head variants and selected one of them by validation loss, the default
practice for VAEs. The run selected by `val_loss` (lr 5e-4, patience 25) achieved the **best
reconstruction of all three runs** (`val ≈ 774`) yet delivered a macroscopic property
R² of **−1.34** — worse than predicting the constant mean (Table 1). A second KAN run selected
by the same criterion reached macro R² = 0.19. Under a likelihood-based selection oracle, KAN
therefore *underperforms even the linear baseline*, while exhibiting excellent reconstruction.

### Real-physics supervision unlocks KAN: two-stage fine-tuning

Fine-tuning the KAN checkpoint with the differentiable real-physics loss restored and then
surpassed the linear baseline (Table 1). Round 1 (λ = 1.0, 50 epochs) reached macro R² = 0.58
(ν₁₂ 0.76, ν₂₁ 0.41; best in-training true-FE R² = 0.7865). Round 2 (λ = 2.0, 60 epochs, resumed
from round 1) reached **macro R² = 0.64** (ν₁₂ **0.82**, ν₂₁ 0.45) with a peak in-training
true-FE R² of **0.8889** — a near-2× improvement over the re-measured linear fine-tuned baseline
(0.35) and a clear monotone progression (0.19 → 0.58 → 0.64). The gain is concentrated in ν₁₂,
the quantity most strongly controlled by `volfrac` in the dataset; ν₂₁ (R² = 0.45) remains the
weak point, consistent with its narrower and more clustered label distribution.

**Table 1.** Architecture comparison on the v2 test set (macro R² = mean of per-target R²).

| Model | ν₁₂ R² | ν₂₁ R² | Macro R² | Best true-FE R² (in-train) |
|---|---|---|---|---|
| Linear base (re-measured) | 0.27 | 0.30 | 0.29 | — |
| Linear fine-tuned (re-measured) | 0.24 | 0.47 | 0.35 | — |
| KAN base (selected by `val_loss`) | 0.35 | 0.02 | 0.19 | — |
| KAN + Real-Physics v1 | 0.76 | 0.41 | 0.58 | 0.7865 |
| **KAN + Real-Physics v2 (recommended)** | **0.82** | **0.45** | **0.64** | **0.8889** |

*Note on metric scope:* the "best true-FE R²" column is the peak of periodic true-FE evaluations
during training (every 10 epochs). It is not directly comparable to best-of-N oracle R² reported
for earlier linear real-physics checkpoints (e.g., 0.9953 for `cvae_v2_finetuned.pt` under a
30-candidate best-of-N protocol), which additionally benefit from candidate selection at
inference time. The two protocols answer different questions; we report both where available.

### Checkpoint-selection oracle matters more than architecture

Across all configurations tested, the choice of model-selection oracle dominated the difference
between architectures: the best-reconstruction KAN checkpoint (R² = −1.34) performed 1.98 R²
*below* the same architecture selected by true-FE R² (0.64). We treat this as a headline result:
for mechanics-sensitive generative models, the selection criterion is part of the method, not an
implementation detail.

### Status of the continuous WIRE decoder

The WIRE INR decoder is implemented and resolution-agnostic (generates at 128²/256²/512² from a
single checkpoint via `generate(resolution=...)`), but has not yet been trained; single-shot
manufacturability (`passes_all ≥ 75%`) and boundary-sharpness targets are therefore
**not yet evaluated** and are deferred to a follow-up study.

---

## Placeholders for the final manuscript

- [ ] Figure: macro R² bar chart (Linear base / finetuned / KAN val-loss / KAN+RP v1 / v2).
- [ ] Figure: per-property (ν₁₂ vs ν₂₁) scatter of predicted vs. target for the recommended
      checkpoint.
- [ ] Figure: reconstruction vs. property-fidelity scatter showing the `val_loss` paradox
      (best recon → worst property).
- [ ] Table: exact training configs (lr, epochs, λ, subsample, `fe-eval-every`) per run, to be
      exported from `outputs/phase5/*_history.json`.
- [ ] 95% bootstrap CIs for macro R² (protocol exists: `bootstrap_ci.py`), pending the
      reproducibility run.
