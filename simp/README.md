# SIMP — Topology Optimization for Periodic Material Microstructure Design

**S**olid **I**sotropic **M**aterial with **P**enalization — a Python implementation of topology optimization for designing periodic unit cells (microstructures) with targeted mechanical properties, especially **auxetic** behavior (negative Poisson's ratio).

This package is a Python port of the MATLAB reference code originally developed by the mechanics team. It provides a complete SIMP optimization loop: FE analysis → homogenization → objective computation → sensitivity filtering → OC update → convergence check.

---

## Quick Start

```bash
# Run with default parameters (100×100 mesh, circle seed, auxetic objective)
python -m simp.run
```

### CLI parameter overrides

You can override the default parameters directly from the terminal by using the CLI entry point:

```bash
python -m simp.main --nelx 80 --nely 60 --volfrac 0.35 --seed hexagonal --objective second --output_dir outputs/simp_hex
```

If the package is installed in editable or normal mode, the same options can also be used via the console script:

```bash
simp --nelx 80 --nely 60 --volfrac 0.35 --seed hexagonal --objective second
```

### Available CLI options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--nelx` | int | 100 | Number of elements in x direction |
| `--nely` | int | 100 | Number of elements in y direction |
| `--volfrac` | float | 0.4 | Target volume fraction |
| `--penal` | float | 3.0 | SIMP penalization factor |
| `--rmin` | float | 3.0 | Filter radius for density/sensitivity filter |
| `--ft` | int | 2 | Filter type (1 = sensitivity, 2 = density) |
| `--E0` | float | 199.0 | Young's modulus of solid material |
| `--Emin` | float | 1e-9 | Young's modulus of void material |
| `--nu` | float | 0.3 | Base material Poisson ratio |
| `--move` | float | 0.1 | Maximum change per OC update |
| `--max_iter` | int | 200 | Maximum optimization iterations |
| `--tol_change` | float | 0.01 | Convergence threshold for design change |
| `--tol_obj` | float | 0.05 | Convergence threshold for objective stability |
| `--window_size` | int | 20 | Number of stable iterations required for objective convergence |
| `--seed` | str | circle | Initial seed pattern name |
| `--objective` | str | auxetic | Objective type; allowed: `first`, `second`, `auxetic` |
| `--void_size_frac` | float | 0.4 | Void size fraction used by the seed generator |
| `--rotation_deg` | float | 0.0 | Initial seed rotation angle in degrees |
| `--beta` | float | 0.85 | Beta decay coefficient for first objective |
| `--beta_second` | float | 1.0 | Penalty weight for second objective |
| `--save_every` | int | 1 | Save image every N iterations |
| `--scale_factor` | int | 1 | Image upscale factor for saved PNGs |
| `--output_dir` | str | outputs/simp_results_[seed] | Output directory for results |
| `--quiet` | flag | false | Suppress final summary output |

Use only the options you want to change; any unspecified option remains at its default value.

### Example: detailed override

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

Results are saved to `outputs/simp_results_[seed]/`:
- `iteration_XXXXX.png` — density field images (grayscale, black = solid, white = void)
- `iteration_data.csv` — convergence history (Poisson ratio, objective, volume fraction)

---

## Package Structure

```
simp/
├── __init__.py               # Package metadata (version 1.1.0, MIT license)
├── run.py                    # Entry point — run optimization with default params
├── runner.py                 # Main optimization loop orchestrator
│
├── core/                     # Core SIMP algorithms
│   ├── fem.py                # FE mesh: node numbering, DOF mapping, sparse index vectors
│   ├── filter.py             # Cone-shaped density filter (prevents checkerboard)
│   ├── pbc.py                # Periodic Boundary Conditions (null‑space projection)
│   ├── solver.py             # Sparse FE solver with PBC (direct + CG fallback)
│   ├── oc.py                 # Optimality Criteria update (bisection on Lagrange multiplier)
│   └── convergence.py        # Convergence detection (design change + objective stability)
│
├── materials/                # Material definition
│   └── isotropic.py          # Isotropic material: 4-node quad element stiffness matrix (plane stress)
│
├── objectives/               # Objective functions
│   ├── first_obj.py          # Type 1: c = Q₁₂ − β^loop · (Q₁₁ + Q₂₂)
│   ├── second_obj.py         # Type 2: c = Q₁₂ + penalty for low axial stiffness
│   └── auxetic.py            # Auxetic: c = ν₁₂ = −Q₁₂ / Q₂₂
│
├── homogenization/           # Homogenization
│   └── compute.py            # Energy‑based homogenization: stiffness tensor Q + sensitivity dQ
│
├── seeds/                    # Initial void pattern generators
│   ├── circle.py             # Single circular void at center
│   ├── square.py             # Single square void at center
│   ├── hourglass.py          # Two triangular voids (hourglass shape)
│   ├── four_circle.py        # Four circular voids (symmetric)
│   ├── hexagonal.py          # Single hexagonal void
│   ├── nine_circle.py        # 3×3 grid of circular voids
│   ├── cross_rectangular.py  # Cross‑shaped void
│   ├── grid_circular_voids.py# Regular grid of circular voids
│   ├── small_square_cross.py # Small square cross at center
│   └── circle_half_quarter.py# Circle + four quarter‑circles at corners
│
└── io/                       # Input / Output
    ├── logger.py             # CSV logging (iteration, v12, v21, objective, volume)
    └── visualizer.py         # Density field PNG export (grayscale, optional scale factor)
```

---

## Available Seeds

| Seed name | Description | Visual |
|-----------|-------------|--------|
| `circle` | Single circular void at center | ⬤ |
| `square` | Single square void at center | ◼ |
| `hourglass` | Two triangular voids forming an hourglass | ⌛ |
| `four_circle` | Four circular voids in symmetric arrangement | ◉ ◉ |
| `hexagonal` | Single hexagonal void | ⬡ |
| `nine_circle` | 3×3 grid of circular voids | 9× ◯ |
| `cross_rectangular` | Cross‑shaped void | ✚ |
| `grid_circular_voids` | Uniform grid of circular voids | ◯◯◯ |
| `small_square_cross` | Small square cross | ┼ |
| `circle_half_quarter` | Center circle + quarter‑circles at corners | ⊙ |

To use a specific seed, set `params['seed'] = 'hexagonal'` (or any name from the list above).

---

## Objective Functions

### 1. First Objective (`first_obj`)

```
c = Q₁₂ − β^loop · (Q₁₁ + Q₂₂)
```

- Maximizes shear coupling `Q₁₂` while suppressing axial stiffness `Q₁₁`, `Q₂₂`
- The `β` decay term (`β^loop`) gradually reduces the axial penalty over iterations
- **Stable convergence**, good for exploring the design space

### 2. Second Objective (`second_obj`)

```
c = Q₁₂  (+ penalty if Q₁₁ < δ or Q₂₂ < δ,  δ = 0.1 · volfrac · E₀)
```

- Directly maximizes `Q₁₂`
- Penalty activates only when axial stiffness falls below threshold `δ`
- **Focused on shear**, may be more aggressive

### 3. Auxetic Objective (`auxetic`)

```
c = ν₁₂ = −Q₁₂ / Q₂₂
```

- Directly minimizes the Poisson ratio (makes it more negative)
- Sensitivity computed via quotient rule
- **Best for achieving auxetic designs** (ν₁₂ < 0)

---

## Programmatic Usage

```python
from simp.runner import run_simp

# Custom parameters
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

# Access the density field
xPhys = result['xPhys']          # (nely, nelx) numpy array
Q     = result['Q']              # 3×3 homogenized stiffness tensor
history = result['history']      # dict with iteration, v12, v21, objective, volume
```

---

## Requirements

- **Python** ≥ 3.10
- **numpy** ≥ 1.24
- **scipy** ≥ 1.10
- **matplotlib** ≥ 3.7 (for PNG output; optional but recommended)

All dependencies are listed in `pyproject.toml` and `requirements.txt` at the project root.

---

## Output Files

### PNG Images (`iteration_XXXXX.png`)
- Grayscale images of the density field at each saved iteration
- Black (0) = void, White (1) = solid material
- Naming follows zero‑padded iteration numbers for easy video/animation assembly

### CSV Data (`iteration_data.csv`)
| Column | Description |
|--------|-------------|
| `Iteration` | Loop number |
| `Poisson_v12` | Poisson ratio ν₁₂ (`= −Q₁₂/Q₂₂`) |
| `Poisson_v21` | Poisson ratio ν₂₁ (`= −Q₁₂/Q₁₁`) |
| `Objective` | Objective function value |
| `Volume_Fraction` | Actual volume fraction (mean of xPhys) |

---

## Convergence Criteria

The optimization stops when **any** of the following conditions is met:

1. **Design change** < `tol_change`: maximum absolute density change between consecutive iterations
2. **Objective stability**: relative objective change < `tol_obj` for `window_size` consecutive iterations
3. **Maximum iterations** `max_iter` reached

A `ConvergenceChecker` class handles all three criteria — see `simp/core/convergence.py`.

---

## Related Resources

| Resource | Description |
|----------|-------------|
| [`../docs/PROJECT_ONBOARDING.md`](../docs/PROJECT_ONBOARDING.md) | Comprehensive project introduction for new collaborators |
| [`../notebooks/`](../notebooks/) | Jupyter notebooks for convergence and image quality analysis |
| [`../analysis/`](../analysis/) | CLI and library for analyzing SIMP outputs (dataset, image metrics, reports) |
| [`simp_workflow_guide.html`](simp_workflow_guide.html) | Interactive HTML guide to the SIMP workflow (in Vietnamese) |
| [`../html/simp_unified_guide.html`](../html/simp_unified_guide.html) | Unified SIMP reference guide |

---

## License

MIT — see [`../LICENSE`](../LICENSE) (or refer to `simp/__init__.py`).

---

## References

- Sigmund, O. (2001). *A 99 line topology optimization code written in Matlab*. Structural and Multidisciplinary Optimization, 21(2), 120–127.
- Andreassen, E., et al. (2011). *Efficient topology optimization in MATLAB using 88 lines of code*. Structural and Multidisciplinary Optimization, 43(1), 1–16.
- Xia, L., & Breitkopf, P. (2015). *Design of materials using topology optimization and energy‑based homogenization*. Archives of Computational Methods in Engineering, 22(2), 229–260.