"""
Phase 1: LHS Screening — Tìm tham số ảnh hưởng nhất.

Quy trình:
  1. Sinh N mẫu từ không gian tham số bằng LHS.
  2. Chạy SIMP (nelx=nely=50, max_iter=150) cho từng mẫu.
  3. Lưu log chi tiết (CSV + JSON).
  4. Tính tương quan Spearman → xác định top-3 tham số quan trọng.
  5. Xuất summary HTML.

Usage:
    python -m pipeline.phase1_screening --objective auxetic --seed circle
    python -m pipeline.phase1_screening --objective first   --n_samples 60
    python -m pipeline.phase1_screening --all  # quét toàn bộ 30 combo
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from scipy.stats.qmc import LatinHypercube
from scipy.stats import spearmanr

# Thêm thư mục gốc vào PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline.params import (
    PARAM_SPACE, FIXED_PARAMS, get_active_params,
    get_param_bounds, SEEDS, OBJECTIVES,
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
    """Chuyển vector mẫu LHS (chuẩn hóa [0,1]) thành dict params.

    Args:
        sample: Vector giá trị trong [0, 1].
        objective: 'auxetic' | 'first' | 'second'.
        seed_name: Tên seed.
        fixed: Dict tham số cố định.
        output_dir: Thư mục đầu ra.

    Returns:
        Dict params cho run_simp.
    """
    if fixed is None:
        fixed = FIXED_PARAMS
    param_names = get_active_params(objective)
    bounds = get_param_bounds(objective)

    params = {}
    for i, name in enumerate(param_names):
        lo, hi = bounds[i]
        params[name] = float(sample[i] * (hi - lo) + lo)

    # Ghi đè tham số cố định
    params.update(fixed)
    params['seed'] = seed_name
    params['objective'] = objective
    if output_dir:
        params['output_dir'] = output_dir
    return params


# ──────────────────────────────────────────────
#  Chạy một mẫu và parse kết quả
# ──────────────────────────────────────────────

def evaluate_sample(params: Dict, sample_id: int, verbose: bool = True) -> Dict:
    """Chạy SIMP cho một mẫu tham số.

    Args:
        params: Từ điển tham số đầy đủ.
        sample_id: ID của mẫu (dùng cho log).
        verbose: In log ra console.

    Returns:
        Dict kết quả với các key:
            sample_id, seed, objective, params, v12, v21,
            obj_value, n_iters, converged, elapsed_time, success.
    """
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

    if verbose:
        print(f'\n[{sample_id}] Running {params["seed"]}/{params["objective"]} ...')

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

        if verbose:
            print(f'[{sample_id}] obj={output["objective"]:+.4e}  '
                  f'v12={output["v12"]:.4f}  n_iter={output["n_iters"]}  '
                  f'converged={output["converged"]}  ({elapsed:.1f}s)')

    except Exception as e:
        elapsed = time.time() - t0
        result['elapsed_time'] = elapsed
        result['error'] = str(e)
        if verbose:
            print(f'[{sample_id}] ERROR: {e}')

    return result


# ──────────────────────────────────────────────
#  Phân tích tương quan Spearman
# ──────────────────────────────────────────────

def analyze_correlation(results: List[Dict], objective: str) -> Dict:
    """Tính tương quan Spearman giữa tham số và objective.

    Args:
        results: Danh sách kết quả từ evaluate_sample.
        objective: Tên mục tiêu.

    Returns:
        Dict với:
            param_names: tên tham số.
            correlations: hệ số Spearman r.
            p_values: p-value.
            top_3: 3 tham số có |correlation| lớn nhất.
    """
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
    """Lưu kết quả chi tiết ra CSV và JSON.

    Args:
        results: Danh sách kết quả mẫu.
        analysis: Dict phân tích tương quan.
        output_dir: Thư mục đầu ra.
        objective: Tên mục tiêu.
        seed_name: Tên seed.
    """
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
#  Main
# ──────────────────────────────────────────────

def run_phase1(
    objective: str,
    seed_name: str,
    n_samples: int = 50,
    output_base: str = 'outputs/pipeline/phase1',
    verbose: bool = True,
) -> Dict:
    """Chạy Phase 1 LHS Screening cho một (seed, objective).

    Args:
        objective: 'auxetic' | 'first' | 'second'.
        seed_name: Tên seed.
        n_samples: Số mẫu LHS.
        output_base: Thư mục gốc đầu ra.
        verbose: In log.

    Returns:
        Dict kết quả tổng hợp.
    """
    seed_dir = os.path.join(output_base, seed_name)
    run_dir = os.path.join(seed_dir, objective)
    os.makedirs(run_dir, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'Phase 1: LHS Screening — {seed_name} / {objective}')
    print(f'  Samples: {n_samples}')
    print(f'  Output:  {run_dir}')
    print(f'='*60)

    # 1. Sinh mẫu LHS
    param_names = get_active_params(objective)
    n_dims = len(param_names)
    sampler = LatinHypercube(d=n_dims, seed=42)
    samples = sampler.random(n=n_samples)
    print(f'  Generated {n_samples} LHS samples ({n_dims} params)')

    # 2. Đánh giá từng mẫu
    results = []
    for i in range(n_samples):
        out_dir = os.path.join(run_dir, f'sample_{i:04d}')
        params = sample_to_params(
            samples[i], objective, seed_name,
            fixed=FIXED_PARAMS, output_dir=out_dir,
        )
        result = evaluate_sample(params, i, verbose=verbose)
        results.append(result)

    # 3. Phân tích
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
    }

    # 4. Lưu kết quả
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
    print(f'{"─"*60}')

    return summary


def run_all_combinations(n_samples: int = 30) -> None:
    """Chạy Phase 1 cho tất cả 30 combo (10 seed × 3 objective).

    Args:
        n_samples: Số mẫu LHS cho mỗi combo.
    """
    all_summaries = []
    for obj in OBJECTIVES:
        for seed in SEEDS:
            summary = run_phase1(obj, seed, n_samples=n_samples, verbose=True)
            all_summaries.append(summary)

    # Ghi tổng hợp
    output_dir = f'outputs/pipeline/phase1'
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, '_all_summaries.json')
    # Chuyển đổi dữ liệu cho JSON
    serializable = []
    for s in all_summaries:
        serializable.append({
            'objective': s['objective'],
            'seed': s['seed'],
            'n_samples': s['n_samples'],
            'n_success': s['n_success'],
            'n_converged': s['n_converged'],
            'n_valid_analysis': s['n_valid_analysis'],
            'top_3_params': s['top_3_params'],
            'best_obj_value': s['best_obj_value'],
        })
    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f'\nTotal summary: {path}')


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Phase 1: LHS Screening — tìm tham số ảnh hưởng nhất',
    )
    parser.add_argument(
        '--objective', type=str, default='auxetic',
        choices=OBJECTIVES,
        help='Mục tiêu tối ưu (mặc định: auxetic)',
    )
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
        '--all', action='store_true',
        help='Quét toàn bộ 30 combo (seed × objective)',
    )
    parser.add_argument(
        '--output', type=str, default='outputs/pipeline/phase1',
        help='Thư mục đầu ra gốc',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.all:
        run_all_combinations(n_samples=args.n_samples)
    else:
        run_phase1(
            objective=args.objective,
            seed_name=args.seed,
            n_samples=args.n_samples,
            output_base=args.output,
        )


if __name__ == '__main__':
    main()