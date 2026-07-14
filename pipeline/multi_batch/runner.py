"""
Batch runner: orchestrate SIMP evaluation across samples in a batch.

Wraps simp.runner.run_simp with:
  - Parameter injection (fixed + active per sample)
  - Multiprocessing Pool execution
  - CSV/JSON result persistence
  - Progress tracking per batch
"""

import json
import os
import sys
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from simp.runner import run_simp

# Default fixed parameters (override from config)
DEFAULT_FIXED: Dict[str, float] = {
    'nelx': 50,
    'nely': 50,
    'ft': 2,
    'E0': 199.0,
    'Emin': 1e-9,
    'nu': 0.3,
    'max_iter': 150,
    'tol_change': 0.01,
    'tol_obj': 0.05,
    'window_size': 20,
    'save_every': 9999,
    'scale_factor': 1,
}


def build_params_dict(
    sample_values: Dict[str, float],
    fixed: Dict[str, float],
    seed_name: str,
    objective: str,
    output_dir: str,
) -> Dict:
    """Build a complete parameter dict for run_simp.

    Args:
        sample_values: Dict of active parameter name -> value.
        fixed: Dict of fixed parameters.
        seed_name: Seed shape name.
        objective: Objective function name.
        output_dir: Per-sample output directory.

    Returns:
        Complete params dict for run_simp().
    """
    params = dict(fixed)
    params.update(sample_values)
    params['seed'] = seed_name
    params['objective'] = objective
    params['output_dir'] = output_dir

    # Ensure nelx/nely if not in fixed
    if 'nelx' not in params:
        params['nelx'] = 50
    if 'nely' not in params:
        params['nely'] = 50

    return params


def evaluate_single(task: Tuple[Dict, int]) -> Dict:
    """Worker function: run SIMP for one sample.

    Must be top-level for multiprocessing pickle.

    Args:
        task: (params_dict, sample_id)

    Returns:
        Result dict with sample_id, v12, v21, obj_value, success, etc.
    """
    params, sample_id = task
    t0 = time.time()
    result = {
        'sample_id': sample_id,
        'seed': params.get('seed', ''),
        'objective': params.get('objective', ''),
        'params': {k: params[k] for k in sorted(params.keys())
                   if k not in ('output_dir', 'seed', 'objective')},
        'v12': None,
        'v21': None,
        'obj_value': None,
        'n_iters': None,
        'converged': False,
        'elapsed_time': None,
        'success': False,
        'error': None,
    }

    try:
        output = run_simp(params)
        elapsed = time.time() - t0
        result['v12'] = float(output['v12'])
        result['v21'] = float(output['v21'])
        result['obj_value'] = float(output['objective'])
        result['n_iters'] = int(output['n_iters'])
        result['converged'] = bool(output['converged'])
        result['elapsed_time'] = round(elapsed, 2)
        result['success'] = True
    except Exception as e:
        elapsed = time.time() - t0
        result['elapsed_time'] = round(elapsed, 2)
        result['error'] = str(e)

    return result


def run_batch_from_design(
    design: "pd.DataFrame",
    output_dir: str,
    batch_id: int,
    fixed: Optional[Dict[str, float]] = None,
    n_workers: Optional[int] = None,
) -> Dict:
    """Run SIMP evaluations from a pre-generated design DataFrame.

    The design DataFrame must have columns: seed, objective, and the parameter
    names (e.g. E0, Emin, nu, penal, rmin, p_norm).

    Args:
        design: DataFrame with one row per (seed, objective, params) combination.
        output_dir: Directory to write results (per-sample subdirs created).
        batch_id: Batch identifier.
        fixed: Override for default fixed parameters.
        n_workers: Number of parallel workers (None = cpu_count - 1).

    Returns:
        Summary dict with path, n_success, best_sample, top_results.
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    fixed_params = dict(DEFAULT_FIXED)
    if fixed:
        fixed_params.update(fixed)

    os.makedirs(output_dir, exist_ok=True)

    n_total = len(design)
    print(f'\n{"="*60}')
    print(f'Batch {batch_id}: Running {n_total} evaluations ({n_workers} workers)')
    print(f'  Output: {output_dir}')
    print(f'{"="*60}')

    # ── Build tasks ──
    tasks: List[Tuple[Dict, int]] = []
    for sample_id, (_, row) in enumerate(design.iterrows()):
        seed_name = row['seed']
        objective = row['objective']
        sample_values = {k: float(row[k]) for k in design.columns
                         if k not in ('seed', 'objective', 'batch_id')}
        sample_dir = os.path.join(
            output_dir, seed_name, f'sample_{sample_id:04d}'
        )
        params = build_params_dict(
            sample_values=sample_values,
            fixed=fixed_params,
            seed_name=seed_name,
            objective=objective,
            output_dir=sample_dir,
        )
        tasks.append((params, sample_id))

    # ── Run evaluations ──
    t_start = time.time()
    results: List[Optional[Dict]] = [None] * n_total
    completed = 0

    try:
        with Pool(processes=n_workers) as pool:
            for result in pool.imap_unordered(evaluate_single, tasks, chunksize=2):
                completed += 1
                sid = result['sample_id']
                results[sid] = result
                if completed % 10 == 0 or completed == n_total:
                    n_ok = sum(1 for r in results if r is not None and r['success'])
                    pct = 100.0 * completed / n_total
                    elapsed = time.time() - t_start
                    print(f'  [{completed}/{n_total} ({pct:.0f}%)] '
                          f'success={n_ok} elapsed={elapsed:.0f}s')
    except Exception as e:
        print(f'\n  ⚠️  Pool failed: {e}')
        print('  Switching to sequential fallback...')
        results = [evaluate_single(t) for t in tasks]

    elapsed_total = time.time() - t_start

    results = [r if r is not None else {'sample_id': i, 'success': False,
                                        'error': 'no_result'}
               for i, r in enumerate(results)]

    n_success = sum(1 for r in results if r['success'])
    n_converged = sum(1 for r in results if r['success'] and r.get('converged'))

    best_per_combo: Dict[str, Dict] = {}
    for r in results:
        if r['success'] and r['obj_value'] is not None:
            key = f"{r['seed']}/{r['objective']}"
            if key not in best_per_combo or r['obj_value'] < best_per_combo[key]['obj_value']:
                best_per_combo[key] = {
                    'sample_id': r['sample_id'],
                    'seed': r['seed'],
                    'objective': r['objective'],
                    'obj_value': float(r['obj_value']),
                    'v12': float(r['v12']) if r['v12'] is not None else None,
                    'v21': float(r['v21']) if r['v21'] is not None else None,
                }

    summary = {
        'batch_id': batch_id,
        'strategy': 'design_df',
        'n_samples': n_total,
        'n_total_tasks': n_total,
        'n_success': n_success,
        'n_converged': n_converged,
        'n_workers': n_workers,
        'elapsed_time': round(elapsed_total, 1),
        'output_dir': output_dir,
        'best_per_combo': best_per_combo,
        'timestamp': datetime.now().isoformat(),
    }

    _save_all_results(results, summary, output_dir)

    success_rate = 100.0 * n_success / n_total if n_total > 0 else 0
    print(f'\n{"─"*60}')
    print(f'Batch {batch_id} Complete')
    print(f'  Success:   {n_success}/{n_total} ({success_rate:.0f}%)')
    print(f'  Converged: {n_converged}/{n_success}')
    print(f'  Time:      {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)')
    print(f'  Output:    {output_dir}')
    print(f'{"─"*60}')

    return summary


def _save_all_results(
    results: List[Dict],
    summary: Dict,
    output_dir: str,
) -> None:
    """Save batch results to JSON and CSV.

    Args:
        results: List of result dicts.
        summary: Batch summary dict.
        output_dir: Output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Full results JSON
    json_path = os.path.join(output_dir, f"batch_{summary['batch_id']}_results.json")
    serializable = []
    for r in results:
        sr = dict(r)
        if isinstance(sr.get('params'), dict):
            sr['params'] = {k: float(v) if isinstance(v, (np.floating,)) else v
                            for k, v in sr['params'].items()}
        serializable.append(sr)

    payload = {
        'metadata': {
            'batch_id': summary['batch_id'],
            'strategy': summary['strategy'],
            'n_samples': summary['n_samples'],
            'n_total_tasks': summary['n_total_tasks'],
            'n_success': summary['n_success'],
            'n_converged': summary['n_converged'],
            'timestamp': summary['timestamp'],
        },
        'summary': summary,
        'results': serializable,
    }
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'  → Results: {json_path}')

    # Summary-only JSON
    sum_path = os.path.join(output_dir, f"batch_{summary['batch_id']}_summary.json")
    with open(sum_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  → Summary: {sum_path}')

    # Aggregate CSV (one row per sample)
    import pandas as pd
    rows = []
    for r in results:
        row = {
            'sample_id': r['sample_id'],
            'seed': r['seed'],
            'objective': r['objective'],
            'success': r['success'],
            'v12': r['v12'],
            'v21': r['v21'],
            'obj_value': r['obj_value'],
            'n_iters': r['n_iters'],
            'converged': r['converged'],
            'elapsed_time': r['elapsed_time'],
            'error': r.get('error', ''),
        }
        if isinstance(r.get('params'), dict):
            row.update(r['params'])
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, f"batch_{summary['batch_id']}_results.csv")
    df.to_csv(csv_path, index=False)
    print(f'  → CSV:     {csv_path}')


