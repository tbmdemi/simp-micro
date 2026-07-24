"""
Batch runner: orchestrate SIMP evaluation across samples in a batch.

Wraps simp.runner.run_simp with:
  - Parameter injection (fixed + active per sample)
  - Multiprocessing Pool execution
  - CSV/JSON result persistence
  - Progress tracking per batch
  - Manufacturability đo TẠI THỜI ĐIỂM SINH (roadmap 6.2/6.3 - xem
    pipeline/phase5_cvae/manufacturability.py) - miễn phí, không tốn FE
    thêm, vì đọc lại đúng file PNG cuối cùng mà run_simp() đã lưu (cùng
    file Phase 3's build_npz.py sẽ đọc để xây dataset ML) - đảm bảo con số
    đo được ở đây khớp CHÍNH XÁC với những gì downstream (surrogate/cVAE)
    sẽ thấy, thay vì tính lại trên xPhys thô (sẽ lệch 1 chút so với ảnh đã
    qua matplotlib render + BOX resize mà Phase 3 thực sự dùng). Phát hiện
    dẫn tới thay đổi này: phân tích ngược 7.920 mẫu Phase 2 đã có cho thấy
    tham số DOE liên tục (volfrac/penal/rmin/move/void_size_frac) hầu như
    không tương quan với manufacturability (|r|<0.12 mọi trường hợp), trong
    khi SEED giải thích chênh lệch tới 8 lần (7,9%-62,8%) - xem
    EXPERIMENT_LOG.md mục "Phase 2 — Manufacturability".
"""

import json
import os
import sys
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from simp.runner import run_simp
from pipeline.phase5_cvae.manufacturability import check_manufacturability

MANUF_RESOLUTION = 64  # khớp pipeline/phase3_dataset/build_npz.py mặc định


def _compute_manufacturability_from_saved_png(sample_dir: str, n_iters: int) -> Optional[Dict]:
    """Đọc lại iteration_{n_iters:05d}.png (đã lưu bởi run_simp() qua
    simp/io/visualizer.py::save_density_image - LUÔN lưu ảnh cuối cùng bất
    kể save_every, xem simp/runner.py) rồi resize 64x64 bằng PIL BOX -
    ĐÚNG HỆT logic pipeline/phase3_dataset/build_npz.py::load_and_resize() -
    để manufacturability đo ở đây khớp chính xác với ảnh Phase 3 sẽ dùng.
    Trả về None nếu không tìm thấy/đọc được file (không nên chặn cả batch
    vì 1 mẫu lỗi I/O hiếm gặp)."""
    png_path = os.path.join(sample_dir, f'iteration_{n_iters:05d}.png')
    if not os.path.exists(png_path):
        return None
    try:
        im = Image.open(png_path).convert('L')
        im = im.resize((MANUF_RESOLUTION, MANUF_RESOLUTION), Image.BOX)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        return check_manufacturability((arr > 0.5).astype(np.float32))
    except Exception:
        return None

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
    'verbose': False,
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
        # roadmap 6.2/6.3 - None nếu FE thất bại hoặc không đọc được PNG
        # (xem _compute_manufacturability_from_saved_png).
        'is_connected': None,
        'min_feature_ok': None,
        'manufacturable': None,
        'periodic_ok': None,
        'passes_all': None,
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

        manuf = _compute_manufacturability_from_saved_png(
            params['output_dir'], result['n_iters'],
        )
        if manuf is not None:
            result['is_connected'] = manuf['is_connected']
            result['min_feature_ok'] = manuf['min_feature_ok']
            result['manufacturable'] = manuf['manufacturable']
            result['periodic_ok'] = manuf['periodic_ok']
            result['passes_all'] = manuf['passes_all']
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
        if r['success'] and r['obj_value'] is not None and r['v12'] is not None:
            key = f"{r['seed']}/{r['objective']}"
            if key not in best_per_combo or r['v12'] < best_per_combo[key]['v12']:
                best_per_combo[key] = {
                    'sample_id': r['sample_id'],
                    'seed': r['seed'],
                    'objective': r['objective'],
                    'obj_value': float(r['obj_value']),
                    'v12': float(r['v12']),
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


