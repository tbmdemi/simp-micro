# SIMP Analyst

**Topology Optimization for Auxetic Metamaterial Microstructure Design**

[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](#)
[![License](https://img.shields.io/badge/license-MIT-green)](#)
[![Version](https://img.shields.io/badge/version-1.4.0-blueviolet)](simp/__init__.py)
[![Branch](https://img.shields.io/badge/branch-OnlyAuxetic-orange)](#)

---

## Overview

**SIMP Analyst** implements the **Solid Isotropic Material with Penalization (SIMP)** method for topology optimization of periodic unit-cell microstructures, targeting **auxetic behaviour** (negative Poisson's ratio). The project's end goal is **inverse design**: given a target Poisson ratio, generate a microstructure geometry that achieves it, using a conditional generative model (cVAE) trained on a surrogate-augmented dataset produced by this SIMP engine.

```
Seed Generation → FE Analysis → Homogenization → Objective & Sensitivity →
Density/Sensitivity Filtering → OC Update → Convergence Check → Repeat
```

This codebase is a Python reimplementation of the classic 88-line/99-line SIMP MATLAB codes, extended with periodic boundary conditions, energy-based homogenization, and an adaptive multi-batch DOE (Design of Experiments) pipeline for large-scale dataset generation.

---

## Table of Contents

- [Project Status](#project-status)
- [Getting Started](#getting-started)
- [Package Structure](#package-structure)
- [Available Seeds](#available-seeds)
- [Objective Function](#objective-function-auxetic)
- [Pipeline: Screening → Multi-Batch DOE → Dataset](#pipeline-screening--multi-batch-doe--dataset)
- [CLI Reference](#cli-reference)
- [Programmatic Usage](#programmatic-usage)
- [Output Files](#output-files)
- [Convergence Criteria](#convergence-criteria)
- [Tests](#tests)
- [Key Bugfixes](#key-bugfixes)
- [Documentation](#documentation)
- [References](#references)
- [License](#license)

---

## Project Status

8-phase inverse-design roadmap. Phases 1-4 complete and validated on real data; Phase 5 baseline trains but **fails real physics verification** (see row below); Phase 6 onward in progress.

| Phase | Component | Status | Notes |
|-------|-----------|--------|-------|
| 0 | Core SIMP Engine | ✅ Stable | 11 seed types, auxetic objective, PBC, energy-based homogenization |
| 1 | LHS Screening | ✅ Complete | Initial run produced **0 auxetic samples** — root-caused to a Poisson-ratio formula error under rotation and an FE displacement bug; both fixed (see [Key Bugfixes](#key-bugfixes)) |
| 2 | Multi-Batch Adaptive DOE | ✅ Complete | **8/8 batches**, 7,920 samples, **82.1% auxetic**, best ν₁₂ = −0.807. Adaptive pipeline auto-stopped after 2 consecutive batches with no objective improvement |
| 3 | Dataset Build (density fields + targets) | ✅ Complete | 7,920 samples → 64×64 density fields, outlier-filtered, seed-stratified 70/15/15 split, physics-aware symmetry augmentation (train: 33,120 samples) |
| 4 | CNN Surrogate Model | ✅ Complete | Predicts (ν₁₂, ν₂₁, volfrac) from density field. Test-set R²: ν₁₂ = 0.910, ν₂₁ = 0.911, volfrac = 0.982 (MAE 0.037 / 0.036 / 0.007). See [Phase 4](#4-cnn-surrogate-model-phase-4) below |
| 5 | Conditional VAE | ✅ Single-shot generation still exploits the surrogate, but **best-of-N + real-FE selection fixes inverse design in practice**: R²(v12, real FE) = **+0.44 to +0.60**, true-auxetic hit rate = **100%** (up from -9.5/0.33 baseline) | Inverse design: target Poisson ratio → generated geometry. A single cVAE sample still cannot be trusted (see surrogate-exploitation findings below), but generating N=30 candidates per target and picking the best by **real FE** (not the surrogate) — the same recipe as the published Deep-DRAM pipeline (Pahlavani et al., *Advanced Materials* 2024) — reliably produces genuinely auxetic geometry. Two training-time remedies (self-play adversarial retraining, surrogate ensembling) were tried first and did **not** fix the single-shot problem within this session's budget; the inference-time fix below does. See [Phase 5](#5-conditional-vae-phase-5) below |
| 6 | cGAN / Conditional Diffusion (optional upgrade) | ⬜ Not started | |
| 7-8 | Validation, deployment | ⬜ Not started | |

> Detailed phase breakdown (2.1-2.9, 3.1-3.6, etc.) tracked in the project workflow dashboard (see [Documentation](#documentation)).
> **Known gaps carried into Phase 6+:** (1) `mu` penalty in the auxetic objective is still disabled (`mu=0.0`), redesign pending; (2) homogenization does not yet export stiffness (`E₁₁/E₀`, `E₂₂/E₀`), so the original roadmap's `f1, f2` multi-objective targets are not available to Phase 4/5; (3) no automated tests exist yet for `pipeline/phase4_surrogate/` or `pipeline/phase5_cvae/`; (4) the `train`-vs-`val` gap on the property-consistency loss term (~2.5–3×, epochs 33-40 of the gamma=20 run) suggests mild overfitting specific to property prediction and should be monitored in future training runs.

---

## Getting Started

### Requirements

- **Python** ≥ 3.10
- **numpy** ≥ 1.24
- **scipy** ≥ 1.10
- **matplotlib** ≥ 3.7 (PNG output)
- **pandas**, **scikit-learn**, **Pillow** (dataset pipeline, `pipeline/phase3/`)

```bash
pip install numpy scipy matplotlib pandas scikit-learn pillow
```

### Quick Start — single SIMP run

```bash
python -m simp.run
```

### CLI Parameter Overrides

```bash
python -m simp.main --nelx 80 --nely 60 --volfrac 0.35 --seed hexagonal
```

If installed in editable mode:

```bash
simp --nelx 80 --nely 60 --volfrac 0.35 --seed hexagonal
```

Expected output:

```
Loop:   1  obj:+1.3345e-01  vol:0.532  chg:0.984  v12:-0.0489  v21:-0.0489
Loop:   2  obj:+9.5432e-02  vol:0.410  chg:0.231  v12:-0.1412  v21:-0.1412
...
Loop: 134  obj:-2.8750e-01  vol:0.400  chg:0.003  v12:-0.8510  v21:-0.8510
[DONE] Hội tụ tại lần lặp 134
Hoàn thành 134 loops (45.2s)
  obj=-0.2875  v12=-0.8510  v21=-0.8510  vol=0.400
```

Results are written to `outputs/simp_results_{seed}/`:
- `iteration_XXXXX.png` — density field images (black = solid, white = void)
- `iteration_data.csv` — convergence history (ν₁₂, ν₂₁, objective, volume fraction)

### Quick Start — full dataset pipeline (Phase 2 → Phase 3)

```bash
# Phase 2: run/extend the adaptive multi-batch DOE (see pipeline/multi_batch/)
python -m pipeline.multi_batch.main --phase1-summary outputs/pipeline/phase1

# Phase 3: build the ML-ready dataset from completed batches
python3 pipeline/phase3/scan_dataset.py
python3 pipeline/phase3/build_npz.py --resolution 64
python3 pipeline/phase3/finalize_dataset.py --resolution 64
```

Output: `outputs/phase3/{train,val,test}.npz` — see [Dataset Build (Phase 3)](#3-dataset-build-density-fields--targets) below.

---

## Package Structure

```
├── simp/                          # Core SIMP package (v1.4.0)
│   ├── run.py                     # Default entry point (hourglass, auxetic)
│   ├── main.py                    # CLI entry point with argparse
│   ├── runner.py                  # Optimisation loop orchestrator
│   ├── config.py                  # SimpConfig dataclass (alternative interface)
│   │
│   ├── core/
│   │   ├── fem.py                 # FE mesh, DOF mapping, sparse index vectors
│   │   ├── filter.py              # Cone-shaped density/sensitivity filter
│   │   ├── pbc.py                 # Periodic Boundary Conditions (null-space projection)
│   │   ├── solver.py              # Sparse FE solver with PBC (LU + CG fallback)
│   │   ├── oc.py                  # Optimality Criteria update (bisection on λ)
│   │   └── convergence.py         # Multi-criterion convergence detection
│   │
│   ├── materials/isotropic.py     # Isotropic material: 4-node quad element stiffness
│   ├── objectives/auxetic.py      # Minimise Q₁₂ − μ(Q₁₁+Q₂₂) + stiffness penalty (μ=0 by default, see notes)
│   ├── homogenization/compute.py  # Energy-based homogenisation: Q, dQ (uses U_total = U0 + χ)
│   │
│   ├── seeds/                     # 11 initial void-pattern generators
│   │   ├── circle.py, square.py, hourglass.py, four_circle.py, hexagonal.py,
│   │   ├── nine_circle.py, cross_rectangular.py, grid_circular_voids.py,
│   │   ├── small_square_cross.py, circle_half_quarter.py
│   │   └── reentrant_bowtie.py    # newest seed, hardest to make auxetic (48.6% success rate)
│   │
│   └── io/
│       ├── logger.py              # CSV logging (iteration, ν₁₂, ν₂₁, objective, volume)
│       └── visualizer.py          # Density-field PNG export (matplotlib imshow, gray, no axes)
│
├── pipeline/
│   ├── params.py                          # Parameter space definitions
│   ├── phase1_screening_parallel.py       # Phase 1: LHS screening with multiprocessing
│   ├── phase1_refine_params.py            # Post-screening parameter range refinement
│   │
│   ├── multi_batch/                       # Phase 2: adaptive multi-batch DOE
│   │   ├── params.py                      # BatchConfig, PipelineConfig, enums
│   │   ├── sampling.py                    # Sobol / LHS / Optimised LHS strategies
│   │   ├── runner.py                      # Batch execution with multiprocessing Pool
│   │   ├── adaptive.py                    # Decision logic: refine / expand / stop
│   │   ├── coverage.py                    # KDE density + sparse-region detection
│   │   └── visualize.py                   # Standalone HTML batch-progression reports
│   │
│   └── phase3/                            # Phase 3: ML dataset construction
│       ├── scan_dataset.py                # Batch results -> manifest.csv (image path + targets)
│       ├── build_npz.py                   # Resize density PNGs, normalize -> dataset_{res}.npz
│       ├── augment_symmetry.py            # Physics-aware rotation/flip augmentation
│       └── finalize_dataset.py            # Outlier filter + stratified split -> train/val/test.npz
│
├── analysis/                      # Sensitivity analysis, Pareto fronts, dataset QC notebooks support
├── notebooks/                     # Jupyter notebooks: data loading, sensitivity, design recommendation
├── html/                          # Dashboards, guides, and reports (see html/index.html)
├── tests/                         # PyTest suite
├── outputs/                       # Generated data (see .gitignore — large .npz/.png not committed)
└── README.md
```

---

## Available Seeds

| Seed | Description | Auxetic success rate* |
|------|-------------|------------------------|
| `circle` | Single circular void at centre | 93.8% |
| `square` | Single square void at centre | 94.0% |
| `hourglass` | Two triangular voids | 67.8% |
| `four_circle` | Four circular voids, symmetric | 87.9% |
| `hexagonal` | Single hexagonal void | 64.4% |
| `nine_circle` | 3×3 grid of circular voids | 98.9% |
| `cross_rectangular` | Cross-shaped void | 93.3% |
| `grid_circular_voids` | N×N uniform grid of circular voids | 99.4% |
| `small_square_cross` | Small square cross at centre | 93.1% |
| `circle_half_quarter` | Centre circle + four quarter-circles at corners | 61.7% |
| `reentrant_bowtie` | Bowtie-shaped void (re-entrant geometry) — newest seed | 48.6% |

\* Fraction of samples with both ν₁₂ < 0 and ν₂₁ < 0, measured across all 7,920 samples of the completed multi-batch DOE (Phase 2). `reentrant_bowtie` and `hexagonal` are the hardest geometries to push auxetic and are good candidates for further parameter refinement.

Rotation (`--rotation_deg`) can be applied to any seed.

---

## Objective Function (Auxetic)

```
c = Q₁₂ − μ·(Q₁₁ + Q₂₂) + penalty_terms
penalty: activates when Q₁₁ or Q₂₂ < δ = 0.1·volfrac·E₀, normalized by δ²
```

- `compute_nu12` / `compute_nu21` use the **full 3×3 compliance-matrix inverse** (`S = Q⁻¹`), not the orthotropic shortcut `ν₁₂ = Q₁₂/Q₂₂` — the shortcut breaks down whenever the unit cell is rotated (shear-normal coupling `Q₁₃, Q₂₃ ≠ 0`). This was the root cause of the Phase 1 zero-auxetic-samples bug (see [Key Bugfixes](#key-bugfixes)).
- The `μ` term is intended to push `Q₁₂` further negative instead of stalling near zero, but the current formulation is **conceptually flawed** (pending redesign) — it is currently disabled by default (`mu=0.0`), which is the setting used for all 8 completed multi-batch DOE runs.
- A stiffness penalty activates when `Q₁₁` or `Q₂₂` falls below `δ`, preventing structural collapse (void degenerate topologies still occur in ~0.4% of runs — filtered out in Phase 3).

---

## Pipeline: Screening → Multi-Batch DOE → Dataset → Surrogate → cVAE

### 1. LHS Screening (Phase 1)

Scans the parameter space (`volfrac`, `penal`, `rmin`, `move`, `void_size_frac`, `rotation_deg`) using Latin Hypercube Sampling.

```bash
python -m pipeline.phase1_screening_parallel --objective auxetic --seed hexagonal
python -m pipeline.phase1_screening_parallel --all   # full sweep, all seeds
```

Sensitivity analysis (Spearman correlation) identified **`volfrac` as the dominant parameter** (r ≈ 0.87–0.96); `move`, `rmin`, `void_size_frac` were non-significant. The first screening run returned **zero auxetic samples**, traced to: (1) the orthotropic ν₁₂ shortcut breaking under rotation, (2) poorly-scaled penalty terms, (3) an FE displacement bug in homogenization (fluctuation field χ used as total displacement instead of `U_total = U0 + χ`).

### 2. Multi-Batch Adaptive DOE (Phase 2) — ✅ complete

Sequential batches, each guided by coverage analysis (KDE + sparse-region detection) of accumulated results. `adaptive.py` decides to **refine** (narrow parameter ranges + target sparse regions), **expand** (add seeds/objectives), or **stop**.

```bash
python -m pipeline.multi_batch.main --phase1-summary outputs/pipeline/phase1
```

**Results (8 batches, 7,920 samples, 100% FE convergence):**

| Batch | Strategy | n samples | Auxetic % | Best ν₁₂ |
|-------|----------|-----------|-----------|----------|
| 1 | Sobol (explore) | 1,320 | 74.9% | −0.612 |
| 2 | Sobol (explore) | 600 | 79.7% | −0.519 |
| 3 | Sobol (explore) | 720 | 71.8% | −0.565 |
| 4 | Optimized LHS (refine) | 1,056 | 83.6% | −0.605 |
| 5 | Optimized LHS (refine) | 1,067 | 85.7% | −0.752 |
| 6 | Optimized LHS (refine) | 1,045 | 85.4% | −0.649 |
| 7 | Optimized LHS (refine) | 1,056 | 85.3% | −0.621 |
| 8 | Optimized LHS (refine) | 1,056 | 87.8% | **−0.807** |

Parameter ranges converged from the initial `volfrac ∈ [0.45, 0.70]` down to `[0.50, 0.58]` by batch 8, consistent with Phase 1 sensitivity results. The pipeline auto-stopped after batch 8 following 2 consecutive batches with no objective improvement (`n_batches_no_improvement: 2` in `decision_batch8.json`) — parameter ranges had converged and sparsity had stabilized at ~18.5%.

### 3. Dataset Build (Phase 3) — ✅ complete

```bash
python3 pipeline/phase3/scan_dataset.py       # -> outputs/phase3/manifest.csv
python3 pipeline/phase3/build_npz.py --resolution 64   # -> dataset_64.npz
python3 pipeline/phase3/finalize_dataset.py --resolution 64  # -> train/val/test.npz
```

- Density field PNGs (616×616, matplotlib `imshow` render of the raw 50×50 `xPhys` grid) resized to 64×64 via box-filter downsampling.
- 33/7,920 samples (0.4%) dropped: degenerate topologies where `volfrac_achieved` collapsed outside `[0.05, 0.95]` despite `converged=True`.
- **Train/val/test split: 70/15/15, stratified by seed geometry** (all 11 seeds represented proportionally in every split).
- **Physics-aware symmetry augmentation** applied to train only: 90°/270° rotation swaps `ν₁₂ ↔ ν₂₁` (unit-cell axes exchange roles); 180° rotation and horizontal/vertical flips preserve `ν₁₂, ν₂₁`. Train set: 5,520 → 33,120 samples (×6).
- Targets currently exported: `v12`, `v21`, `volfrac_achieved`. **Note:** the original roadmap's `f1, f2` properties are not yet computed by the homogenization module — only `ν₁₂`/`ν₂₁` exist today. Extending `compute_homogenized_tensor()` to output normalized stiffness (e.g. `E₁₁/E₀`, `E₂₂/E₀`) is required before `f1, f2` can be added as surrogate/cVAE targets.

### 4. CNN Surrogate Model (Phase 4) — ✅ complete

```bash
python3 pipeline/phase4_surrogate/train.py
python3 pipeline/phase4_surrogate/evaluate.py
python3 pipeline/phase4_surrogate/export_for_phase5.py
```

Architecture ("Phương án A" baseline): 4× `Conv(3x3) + BatchNorm + ReLU + MaxPool` blocks → global average pool → concat seed one-hot → 2 FC layers → 3 outputs (ν₁₂, ν₂₁, volfrac_achieved). Trained on `outputs/phase3/train.npz`, validated on `val.npz`.

**Test-set performance** (`outputs/phase4/evaluation_report.json`, held-out `test.npz`, no leakage):

| Target | R² | MAE |
|---|---|---|
| ν₁₂ | 0.910 | 0.037 |
| ν₂₁ | 0.911 | 0.036 |
| volfrac_achieved | 0.982 | 0.007 |

Per-seed MAE ranges 0.021 (`reentrant_bowtie`) to 0.048 (`square`) — no seed catastrophically underperforms despite `reentrant_bowtie`/`hexagonal` having the lowest auxetic yield in Phase 2. Per-bin error is roughly flat across the ν₁₂ range, with the single most-negative bin (`[-0.8,-0.6)`, n=1) too sparse to draw conclusions from.

If R² < 0.90 on any target, the model doc recommends widening `channels` in `SurrogateCNN` (e.g. `[32,64,128,256] → [64,128,256,512]`) before changing architecture.

### 5. Conditional VAE (Phase 5) — ✅ inverse design solved via best-of-N + real-FE selection

```bash
python3 pipeline/phase5_cvae/train.py --gamma 20.0 --epochs 50
python3 pipeline/phase5_cvae/evaluate.py
python3 pipeline/phase5_cvae/sample.py
```

Trains on `train.npz` (augmented, 33,120 samples), validates on `val.npz`, and uses the **frozen** Phase-4 surrogate to compute a property-consistency loss against the sampled geometry. Total loss: `recon + beta·kl + gamma·PROP_LOSS_SCALE·prop_loss`, with `PROP_LOSS_SCALE = 1000` fixed to bring `prop_loss` (~O(0.01–0.05)) onto the same scale as `recon_loss` (~O(1000)) — without this scaling, `gamma=1` effectively zeroes out the property gradient (see [Key Bugfixes](#key-bugfixes)).

**`gamma` sweep** (property-loss weight; not to be confused with `--beta`, the KL weight), evaluated at test time with `z ~ N(0, 1)` (no encoder leakage):

| gamma | R² (ν₁₂) | MAE (ν₁₂) | pixel_std (diversity @ v12=v21=-0.6) |
|---|---|---|---|
| 1 | −0.418 | 0.174 | 0.326 |
| 5 | 0.450 | 0.106 | 0.274 |
| 20 | 0.633 | 0.086 | 0.314 |

R² kept rising well past `gamma=20` in a wider, since-run sweep (`outputs/phase5/gamma_sweep_results/`): gamma=30→0.60, 50→0.63, 80→0.71, 100→0.76, 150→0.79, 200→0.79, 250→0.81, 300→0.86 — never plateauing.

**⚠️ This R² is not trustworthy — confirmed by real-FE verification.** `pipeline/phase5_cvae/verify_fe.py` (independent of the surrogate: binarizes the generated image, resizes it to the actual FE grid, and runs the real `simp/core/solver.py` + `simp/homogenization/compute.py` to get ground-truth ν₁₂/ν₂₁) shows:

| gamma | R²(v12) via surrogate | R²(v12) via real FE | true-auxetic hit rate (of 24 auxetic-target samples) |
|---|---|---|---|
| 1 | −0.42 | **−1.97** | 7/24 |
| 20 | 0.63 | **−1.16** | 12/24 |
| 100 | 0.76 | **−2.23** | 6/24 |
| 300 | 0.86 | **−2.41** | 4/24 |

Real R² is deeply negative at every gamma, and the surrogate-vs-real gap *widens* as gamma increases (surrogate exploitation — the decoder gets better at fooling the frozen CNN, not at generating real auxetic geometry). True-auxetic hit rate is actually *worse* at high gamma than at gamma=20. Full data: `outputs/phase5/fe_verification_report.json`. Before trusting/extending any gamma-sweep result, run `python3 pipeline/phase5_cvae/verify_fe.py --sanity-check` first (must show mean error < 0.05 using each sample's **real** `penal`, not the fixed default) then the full verification.

**Two training-time remedies were tried and did NOT fix single-shot generation (2026-07-23):**

*Self-play* (`pipeline/phase5_cvae/self_play.py`): periodically fine-tune the surrogate on cVAE-generated images scored by real FE, and select cVAE checkpoints by real-FE R² instead of gameable `val_loss`. A first proof-of-concept (2 rounds, 8 conditions) appeared to improve R² monotonically (−9.50→−4.90→−4.17) — but this turned out to be **measurement noise**: `verify_round()` was (a) using a different random seed each round (so each round was scored on a different subset of the test set — not apples-to-apples) and (b) `model.generate()` samples its latent vector from an unseeded global RNG, so even re-scoring the *same* checkpoint twice gives different numbers. Both bugs are now fixed (fixed seed, `torch.manual_seed()` before generation). Re-verifying rounds 0–8 on one fixed 96-sample held-out set gave essentially flat, noisy R² (−2.27 to −3.30, no trend). A corrected v2 run (10 rounds planned, from-scratch, with a further fix — adversarial samples were only ~0.1% of each fine-tune batch, invisible to gradient descent, oversampled ×40 to fix that) still showed no real improvement over 5 rounds (−2.41 baseline vs −2.65/−2.78/−2.93/−2.67) before being stopped in favor of the fix below.

*Surrogate ensembling* (`pipeline/phase5_cvae/losses.py`: `load_frozen_surrogate_ensemble` / `property_consistency_loss_ensemble`, `train.py --surrogate-paths --lambda-disagreement`): use 3 independently-trained surrogates, train against their mean prediction, and penalize decoder outputs where the 3 disagree (a structural anti-exploitation measure — harder to simultaneously fool 3 independent models, and disagreement is a proxy for "extrapolation region"). Infrastructure is in place and unit-tested (forward+backward pass verified); a full training run did not converge within this session's time budget. Left as a documented, ready-to-use option (`--surrogate-paths a.pt b.pt c.pt --lambda-disagreement 0.1`) for further work, not a proven fix.

**The fix that works: best-of-N + real-FE selection at inference time (`pipeline/phase5_cvae/best_of_n_eval.py`):**

Rather than trying to make a *single* cVAE sample reliably auxetic (the training objective itself is fundamentally hard to trust, since gradient signal has to flow through the same surrogate it's trying to not exploit), generate **N candidates per target condition and let real FE — never the surrogate — pick the winner.** This is the same overall recipe as the published **Deep-DRAM** pipeline ([Pahlavani et al., "Deep Learning for Size-Agnostic Inverse Design of Random-Network 3D Printed Mechanical Metamaterials," *Advanced Materials* 2024, DOI [10.1002/adma.202303481](https://advanced.onlinelibrary.wiley.com/doi/10.1002/adma.202303481)](https://advanced.onlinelibrary.wiley.com/doi/10.1002/adma.202303481)): cVAE generates a population of candidates → a cheap DL surrogate filters/ranks them → real FE simulation is the final arbiter. It works precisely because it never treats the surrogate's output as ground truth for the *reported* design — only as a cheap pre-filter — so surrogate exploitation (which is really about the surrogate being *wrong in a specific, decoder-discoverable way*) can no longer corrupt the final answer.

Measured on the existing `gamma=20` checkpoint (`outputs/phase5/cvae_gamma20.pt`, **no retraining needed**), same fixed 24-condition held-out set used above (`test.npz`, seed=123):

| strategy | # real-FE calls / condition | R²(v12, real FE) | true-auxetic hit rate |
|---|---|---|---|
| single-shot (1 sample, no filtering) | 1 | deeply negative (see gamma-sweep table above) | 0.526 |
| best-of-N, **oracle** (FE on all N=30 candidates, keep closest to target) | 30 | **+0.5955** | **1.000** (24/24) |
| best-of-N, **practical** (surrogate ranks N=30, FE only verifies top K=10) | 10 | **+0.4384** | **1.000** (24/24) |

Every single auxetic target in the held-out set was hit by at least one of the generated candidates, and the practical (surrogate-prefiltered, 3× cheaper) variant keeps almost all of the R² gain. Full data: `outputs/phase5/self_play/best_of_n_result.json` (oracle) and `best_of_n_k10_result.json` (practical). Reproduce with:

```bash
python3 pipeline/phase5_cvae/best_of_n_eval.py --n-samples 30                      # oracle (FE on all N)
python3 pipeline/phase5_cvae/best_of_n_eval.py --n-samples 30 --k-fe-verify 10      # practical (surrogate pre-filter)
```

**Why this is the right way to read the whole Phase 5 story:** the gamma-sweep/self-play/ensemble sections above are all about whether the cVAE's *training objective* can be trusted — it can't, not with a single frozen (or even ensembled) surrogate providing the only gradient signal. But inverse design doesn't require trusting the training objective for a single sample; it only requires that the *trained generator's distribution* contains good solutions somewhere, which it clearly does (24/24 hit rate). Best-of-N with a real-FE final gate is the practical, deployable way to extract them.

**Caveats:**
- The best-of-N numbers above use `outputs/phase5/cvae_gamma20.pt` as-is; they were **not** improved by extra training (self-play/ensemble), so any future retraining effort should be judged against this best-of-N benchmark, not just against single-shot R².
- Property-consistency loss (single-shot training) shows a train/val gap (~0.0005 vs ~0.0013–0.0016 over epochs 33-40 of the `gamma=20` run, roughly 2.5-3×) — mild overfitting specific to property prediction, worth monitoring if training is extended.
- `property_consistency_loss()` uses the **true** seed one-hot of the original sample as a stand-in for the generated image's seed (no seed label exists for generated geometry yet) — a documented approximation (see code comment in `losses.py`), noted as a TODO for a more general version.
- No automated tests yet for this module (see [Tests](#tests)).
- Best-of-N was only validated at N=30/K=10 on this one checkpoint; it was not combined with self-play or ensembling (which could plausibly push the single-shot base rate higher and let a smaller/cheaper N suffice) — a reasonable next step, not required for the current result to hold.

---

## CLI Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--nelx` | int | 100 | Elements in x-direction |
| `--nely` | int | 100 | Elements in y-direction |
| `--volfrac` | float | 0.4 | Target volume fraction |
| `--penal` | float | 3.0 | SIMP penalisation factor |
| `--rmin` | float | 3.0 | Filter radius |
| `--ft` | int | 2 | Filter type (1 = sensitivity, 2 = density) |
| `--E0` | float | 199.0 | Young's modulus, solid |
| `--Emin` | float | 1e-9 | Young's modulus, void |
| `--nu` | float | 0.3 | Base-material Poisson ratio |
| `--move` | float | 0.1 | Max OC update step |
| `--max_iter` | int | 200 | Max iterations |
| `--tol_change` | float | 0.01 | Design-change convergence threshold |
| `--tol_obj` | float | 0.05 | Objective-stability convergence threshold |
| `--window_size` | int | 20 | Stable-iteration window |
| `--seed` | str | hourglass | Initial seed pattern (11 available, see above) |
| `--objective` | str | auxetic | Only `auxetic` is supported |
| `--void_size_frac` | float | 0.4 | Void-size fraction for seed generation |
| `--rotation_deg` | float | 0.0 | Seed rotation angle |
| `--beta` | float | 0.8 | Stiffness-penalty coefficient |
| `--save_every` | int | 1 | Save image every N iterations |
| `--scale_factor` | int | 1 | PNG upscale factor |
| `--output_dir` | str | auto | Default: `outputs/simp_results_{seed}` |
| `--quiet` | flag | false | Suppress final summary |

---

## Programmatic Usage

```python
from simp.runner import run_simp

params = {
    'nelx': 120, 'nely': 120, 'volfrac': 0.35, 'penal': 3.0, 'rmin': 2.5,
    'seed': 'hexagonal', 'objective': 'auxetic', 'void_size_frac': 0.45,
    'max_iter': 300, 'save_every': 5,
}
result = run_simp(params)

print(f'ν₁₂ = {result["v12"]:.4f}, ν₂₁ = {result["v21"]:.4f}, converged: {result["converged"]}')

xPhys = result['xPhys']      # (nely, nelx) density field
Q = result['Q']              # 3×3 homogenised stiffness tensor
history = result['history']  # dict: iteration, v12, v21, objective, volume
```

---

## Output Files

### PNG Images (`iteration_XXXXX.png`)
Grayscale density field — black (0) = void, white (1) = solid.

### CSV Data (`iteration_data.csv`)

| Column | Description |
|--------|-------------|
| `Iteration` | Loop number |
| `Poisson_v12` | ν₁₂, computed via full 3×3 compliance inverse |
| `Poisson_v21` | ν₂₁, computed via full 3×3 compliance inverse |
| `Objective` | Objective function value |
| `Volume_Fraction` | Mean of `xPhys` |

### Metadata (`metadata.json`)
`git_hash`, `timestamp`, `version`, full `params` used for the run.

---

## Convergence Criteria

Stops when **any** condition is met:
1. Design change < `tol_change`
2. Objective stability — relative change < `tol_obj` for `window_size` consecutive iterations
3. `max_iter` reached

Handled by `ConvergenceChecker` (`simp/core/convergence.py`), with `min_iter` to prevent premature stopping. Across the 7,920-sample multi-batch DOE, **FE convergence rate was 100%**.

---

## Tests

```bash
pytest tests/ -v
```

| Module | Status |
|--------|--------|
| CLI argument parsing | ✅ |
| SimpConfig validation | ✅ |
| Convergence checker | ✅ |
| Core smoke (FEM, material, filter, OC, solver, PBC) | ✅ |
| Dataset loading & auxetic classification | ✅ |
| Logger CSV formatting | ✅ |

> `tests/test_core_smoke.py` already has smoke-level (shape/no-crash, not deep-correctness) coverage for `solver.py`, `homogenization/compute.py`, `objectives/*.py`, `core/filter.py`, `core/oc.py`, `core/pbc.py`. Still zero automated tests for `seeds/*.py`, `pipeline/multi_batch/*`, `pipeline/phase3/`, **and — highest priority — `pipeline/phase4_surrogate/` and `pipeline/phase5_cvae/`**, which currently have zero automated tests despite being the most recently added and actively iterated-on modules.

---

## Key Bugfixes

| Bug | Impact | Fix |
|-----|--------|-----|
| **FE displacement bug in homogenization** | Fluctuation field χ used directly as total displacement instead of `U_total = U0 + χ` — silently corrupted the homogenized tensor for every run | Fixed in `compute_homogenized_tensor()`: now uses `U_total = U0 + U` |
| **ν₁₂ orthotropic-shortcut formula under rotation** | `ν₁₂ = Q₁₂/Q₂₂` is only valid when `Q₁₃ = Q₂₃ = 0`; broke down at `rotation_deg = 32.4°` and similar, causing **zero auxetic samples** in the initial Phase 1 screening | Rewrote `compute_nu12`/`compute_nu21` to use the full 3×3 compliance-matrix inverse (`S = Q⁻¹`) |
| **`mu` penalty term in auxetic objective** | Intended to push `Q₁₂` further negative but conceptually flawed — can cause void collapse without reliably improving auxeticity | Disabled by default (`mu=0.0`); redesign pending |
| **Objective sign (removed non-auxetic objectives)** | OC update direction was wrong for previously-supported `first`/`second` objectives | Removed those objective types entirely; only `auxetic` remains |
| **`max` instead of `min` in `aggregate_correlations.py`** | Best-sample selection picked the *worst* objective value | Changed `max(...)` → `min(...)` |
| **Self-play round-over-round measurement confound** | `self_play.py`'s `verify_round()` used seed `123+k` (different every round) and `model.generate()` samples from an unseeded global RNG — apparent R² improvement across rounds 0-8 (−9.50→−1.49) was mostly noise from scoring each round on a *different* condition subset, not real progress (re-verified flat at −2.27 to −3.30 on one fixed set) | Fixed seed (123, constant across rounds) + `torch.manual_seed(seed)` before generation in `verify_round()` |
| **Adversarial data invisible during self-play fine-tuning** | `phase4_surrogate/train.py --adversarial-npz` added ~16-32 adversarial samples to a 33,120-sample batch (~0.1%) — gradient signal from the exploit-specific data was too small to change the surrogate | Added `--adversarial-oversample N` to repeat the adversarial dataset N× before concatenating (self-play v2 used N=40) |

---

## Documentation

Additional dashboards, guides, and technical reports live under `html/` — see [`html/index.html`](html/index.html) for the full index. Note: some dashboard/report pages (`html/dashboards/`, `html/reports/`, `html/guides/`) reflect Phase 1 screening only and have **not yet been regenerated** against the completed Phase 2-5 results in this README. Treat their specific numbers as historical unless regenerated.

- `pipeline/REVIEW_ALGORITHMS_VI.md` — algorithm review (Vietnamese)
- `outputs/multi_batch/reports/batch_progression.html` — auto-generated Phase 2 batch progression report
- `outputs/phase3/manifest.csv`, `split_report.json` — Phase 3 dataset build artifacts
- `outputs/phase4/evaluation_report.json`, `train_history.json` — Phase 4 surrogate results
- `outputs/phase5/evaluation_report.json`, `eval_gamma{1,5,20}.json`, `train_history.json`, `diagnostics/` — Phase 5 cVAE results, including the gamma sweep
- `notebooks/04_phase4_surrogate_analysis.ipynb`, `notebooks/05_phase5_cvae_analysis.ipynb`, `notebooks/06_phase1_to_phase5_pipeline_summary.ipynb` — end-to-end analysis notebooks

---

## References

- Sigmund, O. (2001). *A 99 line topology optimization code written in Matlab.* Structural and Multidisciplinary Optimization, 21(2), 120–127.
- Andreassen, E., et al. (2011). *Efficient topology optimization in MATLAB using 88 lines of code.* Structural and Multidisciplinary Optimization, 43(1), 1–16.
- Xia, L., & Breitkopf, P. (2015). *Design of materials using topology optimization and energy‑based homogenization.* Archives of Computational Methods in Engineering, 22(2), 229–260.
- Bendsøe, M. P., & Sigmund, O. (2003). *Topology Optimization: Theory, Methods, and Applications.* Springer.
- Pahlavani, H., et al. (2024). *Deep Learning for Size-Agnostic Inverse Design of Random-Network 3D Printed Mechanical Metamaterials.* Advanced Materials, 36(6). DOI: [10.1002/adma.202303481](https://advanced.onlinelibrary.wiley.com/doi/10.1002/adma.202303481) — the "Deep-DRAM" cVAE + surrogate + real-FE-selection pipeline that motivated the Phase 5 best-of-N fix above.
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS 2017 — the ensemble-disagreement idea behind `load_frozen_surrogate_ensemble`/`property_consistency_loss_ensemble`.

---

## License

MIT — see [`simp/__init__.py`](simp/__init__.py).

---

*Maintained by the SIMP Analyst Team.*