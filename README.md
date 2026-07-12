# SIMP Analyst

**Topology Optimization for Periodic Material Microstructure Design**

[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](#)
[![License](https://img.shields.io/badge/license-MIT-green)](#)
[![Version](https://img.shields.io/badge/version-1.4.0-blueviolet)](simp/__init__.py)

---

## Overview

**SIMP Analyst** implements the **Solid Isotropic Material with Penalization (SIMP)** method for topology optimization of periodic unit-cell microstructures with targeted mechanical properties - primarily **auxetic behaviour** (negative Poisson's ratio).

The optimization pipeline follows a standard SIMP loop:

```
Seed Generation → FE Analysis → Homogenization → Objective & Sensitivity →
Sensitivity/Density Filtering → OC Update → Convergence Check → Repeat
```

This codebase is a Python reimplementation of original MATLAB reference code and provides a complete, modular, and extensible framework for computational material design.

---

## Table of Contents

- [Project Status](#project-status)
- [Getting Started](#getting-started)
- [Package Structure](#package-structure)
- [Available Seeds](#available-seeds)
- [Objective Functions](#objective-functions)
- [Pipeline: Screening & Adaptive Sampling](#pipeline-screening--adaptive-sampling)
- [CLI Reference](#cli-reference)
- [Programmatic Usage](#programmatic-usage)
- [Output Files](#output-files)
- [Convergence Criteria](#convergence-criteria)
- [Output Data (Google Drive)](#output-data-google-drive)
- [Tests](#tests)
- [Key Bugfixes](#key-bugfixes)
- [References](#references)
- [License](#license)

---

## Project Status

| Component | Status | Notes |
|-----------|--------|-------|
| Core SIMP Engine | ✅ Stable | 10 seed types, 3 objectives, PBC, homogenization |
| Phase 1 Screening (LHS) | ✅ Complete | 30 combos (10 seeds × 3 objectives) × 50 samples run |
| Phase 2 Parameter Tuning | 🟡 Implemented | `differential_evolution`, SHGO, basinhopping, L-BFGS-B |
| Multi-Batch Adaptive Pipeline | 🟡 Implemented | Sobol/LHS + coverage analysis + adaptive decision logic, not yet run in production |
| Unit Tests | 🟡 Partial | Coverage: convergence, logger, config, CLI, dataset smoke tests |
| Performance Optimisation | 🟡 Pending | Seed generators and density filter still use nested loops; vectorisation planned |

---

## Getting Started

### Requirements

- **Python** ≥ 3.10
- **numpy** ≥ 1.24
- **scipy** ≥ 1.10
- **matplotlib** ≥ 3.7 (PNG output; optional but recommended)

Install dependencies:

```bash
pip install numpy scipy matplotlib
```

### Quick Start

```bash
# Run with default parameters (100×100 mesh, hourglass seed, auxetic objective)
python -m simp.run
```

### CLI Parameter Overrides

```bash
python -m simp.main --nelx 80 --nely 60 --volfrac 0.35 --seed hexagonal --objective second
```

If the package is installed in editable mode, the same options are available via the console script:

```bash
simp --nelx 80 --nely 60 --volfrac 0.35 --seed hexagonal --objective second
```

### Full CLI Example

```bash
python -m simp.main \
  --nelx 80 --nely 60 \
  --volfrac 0.35 \
  --penal 4.0 \
  --seed hexagonal \
  --objective second \
  --void_size_frac 0.5 \
  --save_every 5 \
  --output_dir outputs/simp_hex_second
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
- `iteration_XXXXX.png` - density field images (grayscale: black = solid, white = void)
- `iteration_data.csv` - convergence history (Poisson ratio, objective, volume fraction)

---

## Package Structure

```
├── simp/                          # Core SIMP package (v1.4.0)
│   ├── __init__.py
│   ├── run.py                     # Default entry point (hourglass, auxetic)
│   ├── main.py                    # CLI entry point with argparse
│   ├── runner.py                  # Optimisation loop orchestrator
│   ├── config.py                  # SimpConfig dataclass (alternative interface)
│   │
│   ├── core/                      # Core SIMP algorithms
│   │   ├── fem.py                 # FE mesh: node numbering, DOF mapping, sparse index vectors
│   │   ├── filter.py              # Cone-shaped density/sensitivity filter
│   │   ├── pbc.py                 # Periodic Boundary Conditions (null-space projection)
│   │   ├── solver.py              # Sparse FE solver with PBC (LU + CG fallback)
│   │   ├── oc.py                  # Optimality Criteria update (bisection on λ)
│   │   └── convergence.py         # Multi-criterion convergence detection
│   │
│   ├── materials/
│   │   └── isotropic.py           # Isotropic material: 4-node quad element stiffness (plane stress)
│   │
│   ├── objectives/
│   │   ├── auxetic.py             # Minimise Q₁₂ (auxetic target)
│   │   ├── first_obj.py           # Maximise Q₁₂ − β^loop · (Q₁₁ + Q₂₂)
│   │   └── second_obj.py          # Maximise Q₁₂ + stiffness penalty
│   │
│   ├── homogenization/
│   │   └── compute.py             # Energy-based homogenisation: stiffness tensor Q + sensitivity dQ
│   │
│   ├── seeds/                     # 10 initial void-pattern generators
│   │   ├── circle.py
│   │   ├── square.py
│   │   ├── hourglass.py
│   │   ├── four_circle.py
│   │   ├── hexagonal.py
│   │   ├── nine_circle.py
│   │   ├── cross_rectangular.py
│   │   ├── grid_circular_voids.py
│   │   ├── small_square_cross.py
│   │   └── circle_half_quarter.py
│   │
│   └── io/
│       ├── logger.py              # CSV logging (iteration, ν₁₂, ν₂₁, objective, volume)
│       └── visualizer.py          # Density-field PNG export
│
├── pipeline/                      # Screening & tuning pipelines
│   ├── params.py                  # Parameter space definitions (LHS ranges, fixed params)
│   ├── phase1_screening_parallel.py  # Phase 1: LHS screening with multiprocessing
│   ├── phase2_tuning.py           # Phase 2: derivative-free optimisation tuning
│   ├── REVIEW_ALGORITHMS_VI.md    # Algorithm review report (in Vietnamese)
│   │
│   └── multi_batch/               # Adaptive multi-batch pipeline
│       ├── params.py              # BatchConfig, PipelineConfig, enums
│       ├── sampling.py            # Sobol, LHS, Optimised LHS strategies
│       ├── runner.py              # Batch execution with multiprocessing Pool
│       ├── adaptive.py            # Decision logic: stop / refine / expand
│       ├── coverage.py            # KDE density, sparse-region detection
│       └── visualize.py           # Standalone HTML scatter + contour reports
│
├── tests/                         # PyTest test suite
│   ├── test_cli.py
│   ├── test_config.py
│   ├── test_convergence.py
│   ├── test_core_smoke.py
│   ├── test_dataset.py
│   └── test_logger.py
│
├── outputs/                       # ⚠️ Not tracked in git - see section below
│
└── README.md                      # This file
```

---

## Available Seeds

| Seed | Description | Preview |
|------|-------------|---------|
| `circle` | Single circular void at centre | ⬤ |
| `square` | Single square void at centre | ◼ |
| `hourglass` | Two triangular voids (hourglass shape) | ⌛ |
| `four_circle` | Four circular voids, symmetric | ◉ ◉ |
| `hexagonal` | Single hexagonal void | ⬡ |
| `nine_circle` | 3×3 grid of circular voids | 9× ◯ |
| `cross_rectangular` | Cross-shaped void | ✚ |
| `grid_circular_voids` | N×N uniform grid of circular voids | ◯◯◯ |
| `small_square_cross` | Small square cross at centre | ┼ |
| `circle_half_quarter` | Centre circle + four quarter-circles at corners | ⊙ |

Rotation (`--rotation_deg`) can be applied to any seed.

---

## Objective Functions

### 1. Auxetic (default)

```
c = Q₁₂    (+ penalty if Q₁₁ < δ or Q₂₂ < δ)
δ = 0.1 · volfrac · E₀
```

- Directly minimises the shear-coupling term `Q₁₂` (negative → auxetic)
- Stiffness penalty prevents structural collapse
- **Use this for achieving ν₁₂ < 0**

### 2. First Objective

```
c = Q₁₂ − β^loop · (Q₁₁ + Q₂₂)
```

- Maximises shear coupling while suppressing axial stiffness
- `β^loop` decays over iterations, gradually relaxing the axial penalty
- Smooth, stable convergence; useful for exploring the design space

### 3. Second Objective

```
c = Q₁₂    (+ quadratic penalty if Q₁₁ < δ or Q₂₂ < δ)
δ = 0.1 · volfrac · E₀
```

- Aggressively maximises `Q₁₂`
- Penalty activates only when axial stiffness drops below threshold
- May produce more extreme topologies

---

## Pipeline: Screening & Adaptive Sampling

The project provides a phased approach to parameter exploration:

### Phase 1 - LHS Screening

Scans the 7-dimensional parameter space (`volfrac`, `penal`, `rmin`, `move`, `void_size_frac`, `rotation_deg`, beta) using **Latin Hypercube Sampling** (50 samples per combination). Spearman rank correlation identifies the most influential parameters.

```bash
# Single combination
python -m pipeline.phase1_screening_parallel --objective auxetic --seed hexagonal

# Full sweep (30 combos)
python -m pipeline.phase1_screening_parallel --all
```

The screening narrows the active parameter set from 7 to the 2–3 most influential dimensions (currently `volfrac` and `void_size_frac`).

### Phase 2 - Derivative-Free Tuning

Uses `scipy.optimize` solvers (`differential_evolution`, SHGO, basinhopping, L-BFGS-B) to locate the global optimum within the reduced parameter space identified in Phase 1.

### Multi-Batch Adaptive Pipeline

An advanced adaptive-sampling pipeline that runs sequential batches, each guided by coverage analysis of the accumulated results:

- **Batch 1:** Space-filling (Sobol sequence)
- **Batch 2+:** Decision-driven - `adaptive.py` chooses to refine promising regions, expand to new seeds/objectives, or stop
- **Coverage:** KDE density estimation + sparse-region detection
- **Reports:** Standalone HTML pages with scatter plots, density contours, and per-batch progression

```
python -m pipeline.multi_batch.main --phase1-summary outputs/pipeline/phase1
```

> **Note:** The multi-batch pipeline is implemented but has not yet been run in production.

---

## CLI Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--nelx` | int | 100 | Number of elements in x-direction |
| `--nely` | int | 100 | Number of elements in y-direction |
| `--volfrac` | float | 0.4 | Target volume fraction |
| `--penal` | float | 3.0 | SIMP penalisation factor |
| `--rmin` | float | 3.0 | Filter radius (density/sensitivity) |
| `--ft` | int | 2 | Filter type (1 = sensitivity, 2 = density) |
| `--E0` | float | 199.0 | Young's modulus of solid material (GPa) |
| `--Emin` | float | 1e-9 | Young's modulus of void material |
| `--nu` | float | 0.3 | Base-material Poisson ratio |
| `--move` | float | 0.1 | Maximum change per OC update |
| `--max_iter` | int | 200 | Maximum optimisation iterations |
| `--tol_change` | float | 0.01 | Convergence threshold for design change |
| `--tol_obj` | float | 0.05 | Convergence threshold for objective stability |
| `--window_size` | int | 20 | Stable-iteration window for objective convergence |
| `--seed` | str | hourglass | Initial seed pattern name |
| `--objective` | str | auxetic | Objective: `first`, `second`, or `auxetic` |
| `--void_size_frac` | float | 0.4 | Void-size fraction for seed generation |
| `--rotation_deg` | float | 0.0 | Seed rotation angle (degrees) |
| `--beta` | float | 0.8 | Beta decay coefficient (first objective) |
| `--beta_second` | float | 100.0 | Penalty weight (second objective) |
| `--save_every` | int | 1 | Save image every N iterations |
| `--scale_factor` | int | 1 | PNG upscale factor |
| `--output_dir` | str | auto | Output directory (default: `outputs/simp_results_{seed}`) |
| `--quiet` | flag | false | Suppress final summary |

---

## Programmatic Usage

```python
from simp.runner import run_simp

params = {
    'nelx': 120,
    'nely': 120,
    'volfrac': 0.35,
    'penal': 3.0,
    'rmin': 2.5,
    'seed': 'hexagonal',
    'objective': 'auxetic',
    'void_size_frac': 0.45,
    'max_iter': 300,
    'save_every': 5,
}

result = run_simp(params)

print(f'Final Poisson ratio: ν₁₂ = {result["v12"]:.4f}, ν₂₁ = {result["v21"]:.4f}')
print(f'Converged: {result["converged"]}')
print(f'Iterations: {result["n_iters"]}')
print(f'Output directory: {result["output_dir"]}')

# Access results
xPhys = result['xPhys']          # (nely, nelx) density field
Q     = result['Q']              # 3×3 homogenised stiffness tensor
history = result['history']      # dict with iteration, v12, v21, objective, volume
```

---

## Output Files

### PNG Images (`iteration_XXXXX.png`)

Grayscale density field at saved iterations:
- **Black (0)** = void
- **White (1)** = solid material
- Zero-padded iteration numbers for animation assembly

### CSV Data (`iteration_data.csv`)

| Column | Description |
|--------|-------------|
| `Iteration` | Loop number |
| `Poisson_v12` | ν₁₂ = Q₁₂ / Q₂₂ |
| `Poisson_v21` | ν₂₁ = Q₁₂ / Q₁₁ |
| `Objective` | Objective function value |
| `Volume_Fraction` | Mean of `xPhys` |

### Metadata (`metadata.json`)

Each run produces a JSON file containing:
- `git_hash` - commit hash of the code that produced the result
- `timestamp` - run start time
- `version` - SIMP package version
- `params` - full parameter set used

---

## Convergence Criteria

Optimisation stops when **any** of the following conditions is met:

1. **Design change** < `tol_change` - maximum absolute density change between consecutive iterations
2. **Objective stability** - relative objective change < `tol_obj` for `window_size` consecutive iterations
3. **Maximum iterations** - `max_iter` reached

A `ConvergenceChecker` class (in `simp/core/convergence.py`) handles all three criteria, with `min_iter` to prevent premature stopping.

---

## Output Data (Google Drive)

The `outputs/` directory contains all experimental results - raw iteration data, convergence logs, density-field images, screening reports, and slide images - and is **not tracked in this repository**.

All outputs are available on Google Drive:

📁 **[SIMP Analyst - Output Data](https://drive.google.com/drive/folders/1dZHKKWdp4mDRVEXAiOVGuguAr5QYCMBD?usp=sharing)**

The directory includes:

| Path | Contents |
|------|----------|
| `outputs/pipeline/phase1/` | Phase 1 LHS screening: 30 combos × 50 samples, Spearman correlation analysis, refined parameters |
| `outputs/simp_results_circle/` | Single-run example (circle/auxetic) |
| `outputs/slide_images/` | 18 representative topology images for reports and presentations |
| `outputs/figures/` | Spearman correlation heatmaps, bar plots, analysis charts |

To use these results locally, download the desired folder and place it under `outputs/` in the project root.

---

## Tests

```bash
pytest tests/ -v
```

Current test coverage:

| Module | Status |
|--------|--------|
| CLI argument parsing | ✅ |
| SimpConfig validation | ✅ |
| Convergence checker | ✅ |
| Core smoke (FEM, material, filter, OC, solver, PBC) | ✅ |
| Dataset loading & auxetic classification | ✅ |
| Logger CSV formatting | ✅ |

> **Note:** Coverage for `solver.py`, `homogenization/compute.py`, individual `objectives/*.py`, `seeds/*.py`, `core/filter.py`, `core/oc.py`, and `core/pbc.py` is still pending.

---

## Key Bugfixes

The following critical issues were identified during an algorithm review (June 2026) and have been fixed:

| Bug | Impact | Fix |
|-----|--------|-----|
| **Objective sign for `first`/`second` objectives** | OC update moved in wrong direction for maximise-type objectives | Negate `c` and `dc` for `first`/`second` before OC update in `runner.py` |
| **`max` instead of `min` in `aggregate_correlations.py`** | Best-sample selection picked worst objective value | Changed `max(...)` → `min(...)` |
| **Poisson-ratio formula sign** | `ν₁₂ = -Q₁₂/Q₂₂` produced positive ν₁₂ for auxetic designs | Fixed to `ν₁₂ = Q₁₂/Q₂₂` in commit `07914ea` |

---

## References

- Sigmund, O. (2001). *A 99 line topology optimization code written in Matlab.* Structural and Multidisciplinary Optimization, 21(2), 120–127.
- Andreassen, E., et al. (2011). *Efficient topology optimization in MATLAB using 88 lines of code.* Structural and Multidisciplinary Optimization, 43(1), 1–16.
- Xia, L., & Breitkopf, P. (2015). *Design of materials using topology optimization and energy‑based homogenization.* Archives of Computational Methods in Engineering, 22(2), 229–260.
- Bendsøe, M. P., & Sigmund, O. (2003). *Topology Optimization: Theory, Methods, and Applications.* Springer.

---

## License

MIT - see [`simp/__init__.py`](simp/__init__.py).

---

*Maintained by the SIMP Analyst Team.*
