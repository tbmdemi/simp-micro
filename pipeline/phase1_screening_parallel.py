"""
Phase 1: LHS Screening - PARALLEL VERSION với multiprocessing.Pool

Cải tiến:
  1. Dùng multiprocessing.Pool để chạy evaluate_sample song song
  2. 2 chiến lược: 
     - map(): chặn tới khi tất cả xong (đơn giản, progress không rõ)
     - imap_unordered(): bất đồng bộ, thấy progress ngay
  3. Tự động phát hiện số CPU
  4. Fallback sequential nếu có lỗi

Usage:
    python phase1_screening_parallel.py --objective auxetic --seed circle --workers 4
    python phase1_screening_parallel.py --all --workers auto
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats.qmc import LatinHypercube
from scipy.stats import spearmanr

# Thêm thư mục gốc vào PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline.params import (
    PARAM_SPACE, FIXED_PARAMS, get_active_params,
    get_param_bounds, SEEDS,
)
from simp.runner import run_simp


# ──────────────────────────────────────────────
#  Helper: xây dựng dict params từ mẫu LHS
# ──────────────────────────────────────────────

def sample_to_params(
    sample: np.ndarray,
    objective: str,
    seed_name: str,
    fixed: Optional[Dict] = None,
    output_dir: str = '',
) -> Dict:
    """Chuyển vector mẫu LHS (chuẩn hóa [0,1]) thành dict params."""
    if fixed is None:
        fixed = FIXED_PARAMS
    param_names = get_active_params(objective)
    bounds = get_param_bounds(objective)

    params = {}
    for i, name in enumerate(param_names):
        lo, hi = bounds[i]
        params[name] = float(sample[i] * (hi - lo) + lo)

    params.update(fixed)
    params['seed'] = seed_name
    params['objective'] = objective
    if output_dir:
        params['output_dir'] = output_dir
    return params


# ──────────────────────────────────────────────
#  Chạy một mẫu (function toàn cầu cho pickle)
# ──────────────────────────────────────────────

def evaluate_sample(task: Tuple[Dict, int]) -> Dict:
    """Worker function - chạy SIMP cho một mẫu tham số.
    
    Args:
        task: Tuple (params_dict, sample_id).
              Cấu trúc này để tránh pickle nested function.
    
    Returns:
        Dict kết quả với keys:
            sample_id, seed, objective, params, v12, v21,
            obj_value, n_iters, converged, elapsed_time, success, error.
    """
    params, sample_id = task
    t0 = time.time()
    result = {
        'sample_id': sample_id,
        'seed': params['seed'],
        'objective': params['objective'],
        'params': {k: params[k] for k in get_active_params(params['objective'])},
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
        result['elapsed_time'] = elapsed
        result['success'] = True

    except Exception as e:
        elapsed = time.time() - t0
        result['elapsed_time'] = elapsed
        result['error'] = str(e)

    return result


# ──────────────────────────────────────────────
#  Phân tích tương quan Spearman
# ──────────────────────────────────────────────

def analyze_correlation(results: List[Dict], objective: str) -> Dict:
    """Tính tương quan Spearman giữa tham số và objective."""
    param_names = get_active_params(objective)
    n_params = len(param_names)

    # Lọc các mẫu thành công
    valid = [r for r in results if r['success'] and r['obj_value'] is not None]
    if len(valid) < 5:
        return {
            'param_names': param_names,
            'correlations': [None] * n_params,
            'p_values': [None] * n_params,
            'top_3': [],
            'n_valid': len(valid),
        }

    correlations = []
    p_values = []

    for i, pname in enumerate(param_names):
        x_vals = np.array([r['params'].get(pname, np.nan) for r in valid])
        y_vals = np.array([r['obj_value'] for r in valid])

        # Lọc NaN
        mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
        if mask.sum() < 5:
            correlations.append(None)
            p_values.append(None)
            continue

        r, p = spearmanr(x_vals[mask], y_vals[mask])
        correlations.append(r)
        p_values.append(p)

    # Top 3 theo |r|
    abs_r = [abs(r) if r is not None else 0 for r in correlations]
    top_indices = np.argsort(abs_r)[::-1][:3]
    top_3 = [(param_names[i], correlations[i], p_values[i])
             for i in top_indices if correlations[i] is not None]

    return {
        'param_names': param_names,
        'correlations': correlations,
        'p_values': p_values,
        'top_3': top_3,
        'n_valid': len(valid),
    }


# ──────────────────────────────────────────────
#  Ghi log
# ──────────────────────────────────────────────

def save_results(results: List[Dict], analysis: Dict, output_dir: str,
                 objective: str, seed_name: str) -> None:
    """Lưu kết quả chi tiết ra CSV và JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(output_dir, f'phase1_{seed_name}_{objective}.csv')
    param_names = get_active_params(objective)
    header = ['sample_id', 'success', 'v12', 'v21', 'obj_value',
              'n_iters', 'converged', 'elapsed_time'] + param_names + ['error']
    lines = [','.join(header)]
    for r in results:
        row = [
            str(r['sample_id']),
            str(r['success']),
            str(r['v12'] or ''),
            str(r['v21'] or ''),
            str(r['obj_value'] or ''),
            str(r['n_iters'] or ''),
            str(r['converged']),
            f'{r["elapsed_time"]:.2f}' if r['elapsed_time'] else '',
        ]
        for p in param_names:
            val = r['params'].get(p, '')
            if val != '':
                row.append(f'{val:.6f}')
            else:
                row.append('')
        row.append(str(r.get('error', '') or ''))
        lines.append(','.join(row))
    with open(csv_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  → CSV: {csv_path}')

    # JSON
    json_path = os.path.join(output_dir, f'phase1_{seed_name}_{objective}.json')
    serializable_results = []
    for r in results:
        sr = dict(r)
        sr['params'] = {k: float(v) if isinstance(v, (np.floating,)) else v
                        for k, v in sr['params'].items()}
        serializable_results.append(sr)
    payload = {
        'metadata': {
            'objective': objective,
            'seed': seed_name,
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(results),
            'n_success': sum(1 for r in results if r['success']),
        },
        'analysis': {
            'param_names': analysis['param_names'],
            'correlations': [float(c) if c is not None else None for c in analysis['correlations']],
            'p_values': [float(p) if p is not None else None for p in analysis['p_values']],
            'top_3': [(name, float(r), float(p)) if r is not None else (name, None, None)
                      for name, r, p in analysis['top_3']],
            'n_valid': analysis['n_valid'],
        },
        'results': serializable_results,
    }
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'  → JSON: {json_path}')


# ──────────────────────────────────────────────
#  PARALLEL VERSIONS
# ──────────────────────────────────────────────

def run_phase1_parallel_map(
    objective: str,
    seed_name: str,
    n_samples: int = 50,
    n_workers: int = 4,
    output_base: str = 'outputs/pipeline/phase1',
) -> Dict:
    """Chạy Phase 1 với Pool.map (đơn giản, nhưng chặn tới khi xong)."""
    run_dir = os.path.join(output_base, seed_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'Phase 1: LHS Screening (PARALLEL) - {seed_name} / {objective}')
    print(f'  Samples: {n_samples}')
    print(f'  Workers: {n_workers}')
    print(f'  Strategy: map (sync)')
    print(f'  Output:  {run_dir}')
    print(f'='*60)

    # 1. Sinh mẫu LHS
    param_names = get_active_params(objective)
    n_dims = len(param_names)
    sampler = LatinHypercube(d=n_dims, seed=42)
    samples = sampler.random(n=n_samples)
    print(f'  Generated {n_samples} LHS samples ({n_dims} params)')

    # 2. Chuẩn bị tasks
    tasks = []
    for i in range(n_samples):
        out_dir = os.path.join(run_dir, f'sample_{i:04d}')
        params = sample_to_params(
            samples[i], objective, seed_name,
            fixed=FIXED_PARAMS, output_dir=out_dir,
        )
        tasks.append((params, i))

    # 3. Chạy song song
    t_start = time.time()
    print(f'\n  Starting evaluation with {n_workers} workers...')
    
    try:
        with Pool(processes=n_workers) as pool:
            results = pool.map(evaluate_sample, tasks, chunksize=1)
    except Exception as e:
        print(f'  ⚠️  Pool.map failed: {e}')
        print(f'  Fallback to sequential evaluation...')
        results = [evaluate_sample(task) for task in tasks]

    elapsed_total = time.time() - t_start

    # 4. In progress
    n_success = sum(1 for r in results if r['success'])
    print(f'\n  Completed in {elapsed_total:.1f}s')
    print(f'  Success: {n_success}/{n_samples}')

    # 5. Phân tích
    analysis = analyze_correlation(results, objective)

    n_converged = sum(1 for r in results if r['success'] and r['converged'])
    best_idx = None
    best_obj = float('inf')
    for i, r in enumerate(results):
        if r['success'] and r['obj_value'] is not None:
            if r['obj_value'] < best_obj:
                best_obj = r['obj_value']
                best_idx = i

    summary = {
        'objective': objective,
        'seed': seed_name,
        'n_samples': n_samples,
        'n_success': n_success,
        'n_converged': n_converged,
        'n_valid_analysis': analysis['n_valid'],
        'top_3_params': [
            {'name': t[0], 'r': float(t[1]) if t[1] is not None else None,
             'p': float(t[2]) if t[2] is not None else None}
            for t in analysis['top_3']
        ],
        'best_sample': best_idx,
        'best_obj_value': float(best_obj) if best_obj != float('inf') else None,
        'best_params': (
            {k: float(results[best_idx]['params'][k])
             for k in param_names}
            if best_idx is not None else None
        ),
        'output_dir': run_dir,
        'elapsed_time': elapsed_total,
        'n_workers': n_workers,
    }

    # 6. Lưu kết quả
    save_results(results, analysis, run_dir, objective, seed_name)

    # In summary
    print(f'\n{"─"*60}')
    print(f'Summary: {seed_name} / {objective}')
    print(f'  Success:   {n_success}/{n_samples}')
    print(f'  Converged: {n_converged}/{n_success}')
    if summary['best_params']:
        print(f'  Best sample #{best_idx}: obj={best_obj:.4e}')
        for k, v in summary['best_params'].items():
            print(f'    {k} = {v:.4f}')
    print(f'  Top-3 influential params:')
    for t in analysis['top_3']:
        print(f'    {t[0]}: r={t[1]:+.3f}  p={t[2]:.3e}')
    print(f'  Total time: {elapsed_total:.1f}s')
    print(f'{"─"*60}')

    return summary


def run_phase1_parallel_async(
    objective: str,
    seed_name: str,
    n_samples: int = 50,
    n_workers: int = 4,
    output_base: str = 'outputs/pipeline/phase1',
) -> Dict:
    """Chạy Phase 1 với Pool.imap_unordered (bất đồng bộ, thấy progress)."""
    run_dir = os.path.join(output_base, seed_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'Phase 1: LHS Screening (PARALLEL) - {seed_name} / {objective}')
    print(f'  Samples: {n_samples}')
    print(f'  Workers: {n_workers}')
    print(f'  Strategy: imap_unordered (async + progress)')
    print(f'  Output:  {run_dir}')
    print(f'='*60)

    # 1. Sinh mẫu LHS
    param_names = get_active_params(objective)
    n_dims = len(param_names)
    sampler = LatinHypercube(d=n_dims, seed=42)
    samples = sampler.random(n=n_samples)
    print(f'  Generated {n_samples} LHS samples ({n_dims} params)')

    # 2. Chuẩn bị tasks
    tasks = []
    for i in range(n_samples):
        out_dir = os.path.join(run_dir, f'sample_{i:04d}')
        params = sample_to_params(
            samples[i], objective, seed_name,
            fixed=FIXED_PARAMS, output_dir=out_dir,
        )
        tasks.append((params, i))

    # 3. Chạy song song với progress
    t_start = time.time()
    print(f'\n  Starting evaluation with {n_workers} workers...\n')
    
    results = [None] * n_samples
    completed = 0
    
    try:
        with Pool(processes=n_workers) as pool:
            for result in pool.imap_unordered(evaluate_sample, tasks, chunksize=2):
                completed += 1
                sample_id = result['sample_id']
                results[sample_id] = result
                
                # Print progress
                status = '✓' if result['success'] else '✗'
                if result['elapsed_time']:
                    elapsed_str = f"{result['elapsed_time']:.1f}s"
                else:
                    elapsed_str = "N/A"
                # Format obj_value safely (not nested in f-string)
                if result['obj_value'] is not None:
                    obj_str = f"{result['obj_value']:+.4e}"
                else:
                    obj_str = "N/A"
                print(f"  [{completed:2d}/{n_samples}] Sample {sample_id:3d} {status} "
                      f"obj={obj_str} ({elapsed_str})")
    except Exception as e:
        print(f'\n  ⚠️  Pool.imap_unordered failed: {e}')
        print(f'  Fallback to sequential evaluation...')
        results = [evaluate_sample(task) for task in tasks]

    elapsed_total = time.time() - t_start

    # 4. Phân tích
    analysis = analyze_correlation(results, objective)

    n_success = sum(1 for r in results if r['success'])
    n_converged = sum(1 for r in results if r['success'] and r['converged'])
    best_idx = None
    best_obj = float('inf')
    for i, r in enumerate(results):
        if r['success'] and r['obj_value'] is not None:
            if r['obj_value'] < best_obj:
                best_obj = r['obj_value']
                best_idx = i

    summary = {
        'objective': objective,
        'seed': seed_name,
        'n_samples': n_samples,
        'n_success': n_success,
        'n_converged': n_converged,
        'n_valid_analysis': analysis['n_valid'],
        'top_3_params': [
            {'name': t[0], 'r': float(t[1]) if t[1] is not None else None,
             'p': float(t[2]) if t[2] is not None else None}
            for t in analysis['top_3']
        ],
        'best_sample': best_idx,
        'best_obj_value': float(best_obj) if best_obj != float('inf') else None,
        'best_params': (
            {k: float(results[best_idx]['params'][k])
             for k in param_names}
            if best_idx is not None else None
        ),
        'output_dir': run_dir,
        'elapsed_time': elapsed_total,
        'n_workers': n_workers,
    }

    # 5. Lưu kết quả
    save_results(results, analysis, run_dir, objective, seed_name)

    # In summary
    print(f'\n{"─"*60}')
    print(f'Summary: {seed_name} / {objective}')
    print(f'  Success:   {n_success}/{n_samples}')
    print(f'  Converged: {n_converged}/{n_success}')
    if summary['best_params']:
        print(f'  Best sample #{best_idx}: obj={best_obj:.4e}')
        for k, v in summary['best_params'].items():
            print(f'    {k} = {v:.4f}')
    print(f'  Top-3 influential params:')
    for t in analysis['top_3']:
        print(f'    {t[0]}: r={t[1]:+.3f}  p={t[2]:.3e}')
    print(f'  Total time: {elapsed_total:.1f}s')
    print(f'{"─"*60}')

    return summary


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Phase 1: LHS Screening (PARALLEL) - tìm tham số ảnh hưởng nhất',
    )
    # The --objective parameter has been removed; it is fixed as 'auxetic' throughout
    parser.add_argument(
        '--seed', type=str, default='circle',
        choices=SEEDS,
        help='Tên seed (mặc định: circle)',
    )
    parser.add_argument(
        '--n_samples', type=int, default=50,
        help='Số mẫu LHS (mặc định: 50)',
    )
    parser.add_argument(
        '--workers', type=str, default='auto',
        help='Số workers: "auto" (dùng CPU count), hoặc số nguyên (mặc định: auto)',
    )
    parser.add_argument(
        '--strategy', type=str, default='async',
        choices=['map', 'async'],
        help='Chiến lược Pool: "map" (sync) hay "async" (progress) (mặc định: async)',
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Chạy tất cả các seeds (ghi đè --seed nếu được chỉ định)',
    )
    parser.add_argument(
        '--output', type=str, default='outputs/pipeline/phase1',
        help='Thư mục đầu ra gốc',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Xác định số workers
    if args.workers.lower() == 'auto':
        n_workers = max(1, cpu_count() - 1)
    else:
        try:
            n_workers = int(args.workers)
        except ValueError:
            print(f'Error: --workers must be "auto" or integer, got {args.workers}')
            sys.exit(1)
    
    print(f'\n[System Info]')
    print(f'  CPU count: {cpu_count()}')
    print(f'  Workers: {n_workers}')
    print(f'  Strategy: {args.strategy}')

    run_phase1_fn = (
        run_phase1_parallel_async if args.strategy == 'async'
        else run_phase1_parallel_map
    )

    if args.all:
        print(f'[INFO] Running all seeds: {SEEDS}')
        for seed in SEEDS:
            run_phase1_fn(
                objective='auxetic',
                seed_name=seed,
                n_samples=args.n_samples,
                n_workers=n_workers,
                output_base=args.output,
            )
    else:
        run_phase1_fn(
            objective='auxetic',
            seed_name=args.seed,
            n_samples=args.n_samples,
            n_workers=n_workers,
            output_base=args.output,
        )


if __name__ == '__main__':
    main()
