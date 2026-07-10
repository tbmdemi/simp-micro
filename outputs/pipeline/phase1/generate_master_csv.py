"""
Generate master CSV for Phase 1 data.

Each row corresponds to one sample, with columns grouped into:

  - Metadata:    sample_id, objective, seed, batch
  - Input:       volfrac, move, rmin, penal, void_size_frac, rotation_deg,
                 beta, beta_second
  - Output:      best_obj_value, poisson_ratio, eigenfreq_1, eigenfreq_2,
                 compliance, volume_fraction_final
  - Status:      n_iter, converged, elapsed_time, topology_path

Sources:
  - Summary CSV:        phase1_{seed}_{objective}.csv  (per seed/objective)
  - Iteration data:     sample_{idx:04d}/iteration_data.csv  (per sample)
  - Topology image:     sample_{idx:04d}/iteration_{last:05d}.png (final iteration)

NOTE on eigenfreq_1: For 'first' and 'second' objectives, the value stored
is best_obj_value (generalized objective), NOT the actual physical frequency.
For 'second' specifically, obj = -(f2 - f1), so gating best_obj_value to
eigenfreq_1 is physically inaccurate. Phase 2+ should log true eigenfrequencies
as separate outputs.
NOTE on compliance: Not computed in Phase 1; always NaN.
NOTE on eigenfreq_2: Not computed in Phase 1; always NaN.
"""

import csv
import os

import numpy as np

OUTPUT_DIR = "outputs/pipeline/phase1"

ALL_SEEDS = [
    'circle',
    'circle_half_quarter',
    'cross_rectangular',
    'four_circle',
    'grid_circular_voids',
    'hexagonal',
    'hourglass',
    'nine_circle',
    'small_square_cross',
    'square',
]
OBJECTIVES = ['auxetic', 'first', 'second']

MASTER_HEADER = [
    # ── Metadata ──
    'sample_id',
    'objective',
    'seed',
    'batch',
    # ── Input / Design parameters ──
    'volfrac',
    'move',
    'rmin',
    'penal',
    'void_size_frac',
    'rotation_deg',
    'beta',
    'beta_second',
    # ── Output / Material properties ──
    'best_obj_value',
    'poisson_ratio',
    'eigenfreq_1',
    'eigenfreq_2',
    'compliance',
    'volume_fraction_final',
    # ── Status & performance ──
    'n_iter',
    'converged',
    'elapsed_time',
    'topology_path',
]


def _safe_float(val: str | None) -> float | None:
    """Convert a CSV cell to float, returning None on failure."""
    if val is None:
        return None
    v = val.strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _safe_int(val: str | None) -> int | None:
    """Convert a CSV cell to int, returning None on failure."""
    if val is None:
        return None
    v = val.strip()
    if not v:
        return None
    try:
        return int(float(v))  # tolerate "36.0" style numbers
    except ValueError:
        return None


def find_last_iteration(sample_dir: str) -> int | None:
    """Return the highest iteration number from PNG files in *sample_dir*."""
    if not os.path.isdir(sample_dir):
        return None
    max_iter = None
    for fname in os.listdir(sample_dir):
        if fname.startswith('iteration_') and fname.endswith('.png'):
            stem = fname[len('iteration_'):-len('.png')]
            try:
                num = int(stem)
                if max_iter is None or num > max_iter:
                    max_iter = num
            except ValueError:
                continue
    return max_iter


def read_volume_fraction_final(sample_dir: str) -> float | None:
    """
    Read the final Volume_Fraction from iteration_data.csv.

    Returns None if the file is missing or unparseable.
    """
    csv_path = os.path.join(sample_dir, 'iteration_data.csv')
    if not os.path.isfile(csv_path):
        return None
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                return None
            last = rows[-1].get('Volume_Fraction', '').strip()
            return float(last) if last else None
    except Exception:
        return None


def generate_master_csv() -> None:
    """Assemble the Phase-1 master CSV from per-seed per-objective files."""
    master_path = os.path.join(OUTPUT_DIR, 'master_phase1.csv')
    rows: list[list] = [MASTER_HEADER]
    total_rows = 0

    for seed in ALL_SEEDS:
        for obj in OBJECTIVES:
            summary_csv = os.path.join(
                OUTPUT_DIR, seed, obj, f'phase1_{seed}_{obj}.csv'
            )
            if not os.path.isfile(summary_csv):
                print(f'  [SKIP] missing: {summary_csv}')
                continue

            with open(summary_csv, 'r') as f:
                reader = csv.DictReader(f)
                summary_cols = reader.fieldnames or []

                for row_dict in reader:
                    idx = _safe_int(row_dict.get('sample_id')) or 0

                    # ── Metadata ──────────────────────────────────────
                    sample_id = f'{obj}_{seed}_{idx:03d}'

                    # ── Input parameters ──────────────────────────────
                    volfrac       = _safe_float(row_dict.get('volfrac'))
                    move          = _safe_float(row_dict.get('move'))
                    rmin          = _safe_float(row_dict.get('rmin'))
                    penal         = _safe_float(row_dict.get('penal'))
                    void_size_frac = _safe_float(row_dict.get('void_size_frac'))
                    rotation_deg  = _safe_float(row_dict.get('rotation_deg'))

                    # beta / beta_second — only the matching objective has a value
                    beta = None
                    beta_second = None
                    if obj == 'first' and 'beta' in summary_cols:
                        beta = _safe_float(row_dict.get('beta'))
                    elif obj == 'second' and 'beta_second' in summary_cols:
                        beta_second = _safe_float(row_dict.get('beta_second'))

                    # ── Output properties ─────────────────────────────
                    best_obj_value = _safe_float(row_dict.get('obj_value'))

                    # Poisson ratio: only physically meaningful for auxetic
                    poisson_ratio = None
                    if obj == 'auxetic':
                        poisson_ratio = _safe_float(row_dict.get('v12'))

                    # Eigenfrequency: proxy value (see docstring)
                    eigenfreq_1 = best_obj_value if obj in ('first', 'second') else None
                    eigenfreq_2 = None       # not available in Phase 1
                    compliance  = None       # not available in Phase 1

                    # Volume fraction after optimisation
                    sample_dir = os.path.join(
                        OUTPUT_DIR, seed, obj, f'sample_{idx:04d}'
                    )
                    vol_frac_final = read_volume_fraction_final(sample_dir)

                    # ── Status & performance ──────────────────────────
                    n_iter      = _safe_int(row_dict.get('n_iters'))
                    converged   = row_dict.get('converged', '').strip().lower() == 'true'
                    elapsed_time = _safe_float(row_dict.get('elapsed_time'))

                    # Topology path: final-iteration PNG relative to OUTPUT_DIR
                    last_iter = find_last_iteration(sample_dir)
                    if last_iter is not None:
                        topology_path = (
                            f'./{seed}/{obj}/sample_{idx:04d}'
                            f'/iteration_{last_iter:05d}.png'
                        )
                    else:
                        topology_path = None

                    # ── Assemble row ──────────────────────────────────
                    rows.append([
                        sample_id,
                        obj,
                        seed,
                        'phase1',
                        volfrac,
                        move,
                        rmin,
                        penal,
                        void_size_frac,
                        rotation_deg,
                        beta,
                        beta_second,
                        best_obj_value,
                        poisson_ratio,
                        eigenfreq_1,
                        eigenfreq_2,
                        compliance,
                        vol_frac_final,
                        n_iter,
                        converged,
                        elapsed_time,
                        topology_path,
                    ])
                    total_rows += 1

    # ── Write ─────────────────────────────────────────────────────────
    with open(master_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f'Master CSV generated: {master_path}')
    print(f'Total data rows (excl. header): {total_rows}')
    print(f'Columns ({len(MASTER_HEADER)}):')
    for i, col in enumerate(MASTER_HEADER, 1):
        print(f'  {i:2d}. {col}')


if __name__ == '__main__':
    generate_master_csv()