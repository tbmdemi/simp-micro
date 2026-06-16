"""
Phase 2: Tuning tham số tối ưu - sử dụng các thuật toán tối ưu hóa toàn cục.

Sau Phase 1 (LHS screening → top-3 tham số ảnh hưởng nhất), Phase 2 dùng
các thuật toán tối ưu hóa dựa trên gradient-free global search để tìm bộ tham số
tối ưu cho từng cặp (seed, objective).

Thuật toán hỗ trợ:
  - differential_evolution (DE, mặc định) - global search, robust, handle constraints
  - shgo (simplicial homology global opt) - alternative global method
  - basinhopping (BH) - stochastic + local refinement
  - refine (L-BFGS-B từ Phase 1 best) - local refinement

Usage:
    python -m pipeline.phase2_tuning --objective auxetic --seed circle --method de --n_iter 100
    python -m pipeline.phase2_tuning --all --method de --n_iter 80
    python -m pipeline.phase2_tuning --all --method de --n_iter 80 --max_simp_iters 100
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np

# Thêm thư mục gốc vào PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline.params import (
    PARAM_SPACE, FIXED_PARAMS, get_active_params,
    get_param_bounds, SEEDS, OBJECTIVES,
)
from simp.runner import run_simp

# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

class SimpObjective:
    """Wrapper biến các tham số SIMP thành hàm mục tiêu cho optimizer.

    Mỗi lần gọi: (1) xây dựng dict params, (2) chạy SIMP, (3) trả về objective.
    Ghi nhận lịch sử để phân tích hội tụ.
    """

    def __init__(
        self,
        seed_name: str,
        objective: str,
        fixed: Optional[Dict] = None,
        output_base: str = 'outputs/pipeline/phase2',
        verbose: bool = False,
        max_simp_iters: int = 150,
    ):
        self.seed_name = seed_name
        self.objective = objective
        self.fixed = fixed or dict(FIXED_PARAMS)
        self.fixed['save_every'] = 9999  # luôn tắt lưu ảnh trung gian
        self.output_base = output_base
        self.verbose = verbose
        self.max_simp_iters = max_simp_iters

        self.eval_history: List[Dict] = []
        self.n_eval = 0
        self.param_names = get_active_params(objective)
        self.bounds = get_param_bounds(objective)

        # Tạo thư mục output
        seed_obj_dir = os.path.join(output_base, seed_name, objective)
        os.makedirs(seed_obj_dir, exist_ok=True)

    def _params_to_dict(self, x: np.ndarray) -> Dict:
        """Chuyển vector tham số (giá trị raw) thành dict."""
        params = {}
        for i, name in enumerate(self.param_names):
            # Clip về bounds để đảm bảo an toàn
            lo, hi = self.bounds[i]
            params[name] = float(np.clip(x[i], lo, hi))
        params.update(self.fixed)
        params['seed'] = self.seed_name
        params['objective'] = self.objective
        params['max_iter'] = self.max_simp_iters
        return params

    def __call__(self, x: np.ndarray) -> float:
        """Đánh giá SIMP với tham số `x` (vector raw values).

        Args:
            x: Vector tham số (giá trị thực, không chuẩn hóa).

        Returns:
            Giá trị objective (càng thấp càng tốt).
        """
        params = self._params_to_dict(x)

        t0 = time.time()
        eval_id = self.n_eval
        self.n_eval += 1

        # Output dir riêng cho eval này
        eval_dir = os.path.join(
            self.output_base, self.seed_name, self.objective,
            f'eval_{eval_id:04d}',
        )
        params['output_dir'] = eval_dir

        try:
            result = run_simp(params)
            elapsed = time.time() - t0

            obj_val = float(result['objective'])
            v12 = float(result['v12'])
            n_iters = int(result['n_iters'])
            converged = bool(result['converged'])

            entry = {
                'eval_id': eval_id,
                'params': {k: float(x[i]) for i, k in enumerate(self.param_names)},
                'objective': obj_val,
                'v12': v12,
                'n_simp_iters': n_iters,
                'converged': converged,
                'elapsed': elapsed,
                'success': True,
                'error': None,
            }

            if self.verbose and (eval_id % 5 == 0 or eval_id < 5):
                print(
                    f'  [eval {eval_id:4d}] obj={obj_val:+.4e}  '
                    f'v12={v12:.4f}  n_iter={n_iters}  '
                    f'conv={converged}  ({elapsed:.1f}s)'
                )

        except Exception as e:
            elapsed = time.time() - t0
            obj_val = 1e10  # penalty lớn cho mẫu lỗi
            entry = {
                'eval_id': eval_id,
                'params': {k: float(x[i]) for i, k in enumerate(self.param_names)},
                'objective': obj_val,
                'v12': None,
                'n_simp_iters': None,
                'converged': False,
                'elapsed': elapsed,
                'success': False,
                'error': str(e),
            }
            if self.verbose:
                print(f'  [eval {eval_id:4d}] ERROR: {e}')

        self.eval_history.append(entry)
        return obj_val

    def get_best(self) -> Dict:
        """Trả về eval entry có objective tốt nhất."""
        valid = [e for e in self.eval_history if e['success']]
        if not valid:
            return {}
        return min(valid, key=lambda e: e['objective'])

    def summary(self) -> Dict:
        """Tổng hợp kết quả."""
        valid = [e for e in self.eval_history if e['success']]
        if not valid:
            return {'n_evals': self.n_eval, 'n_success': 0}

        obj_vals = [e['objective'] for e in valid]
        return {
            'n_evals': self.n_eval,
            'n_success': len(valid),
            'best_objective': float(min(obj_vals)),
            'best_params': self.get_best()['params'],
            'best_v12': self.get_best().get('v12'),
            'mean_objective': float(np.mean(obj_vals)),
            'median_objective': float(np.median(obj_vals)),
            'std_objective': float(np.std(obj_vals)),
        }


# ──────────────────────────────────────────────
#  Các thuật toán optimization
# ──────────────────────────────────────────────

def _objective_wrapper(
    x_norm: np.ndarray,
    bounds: List[Tuple[float, float]],
    obj_func: SimpObjective,
) -> float:
    """Wrapper: denormalize vector [0,1]^d → raw bounds → evaluate.

    Dùng cho các optimizer yêu cầu đầu vào chuẩn hóa (DE, SHGO).
    """
    # Denormalize: [0,1] → [lo, hi]
    x_raw = np.array([
        x_norm[i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]
        for i in range(len(x_norm))
    ])
    return obj_func(x_raw)


def run_differential_evolution(
    obj: SimpObjective,
    n_iter: int = 100,
    seed: int = 42,
    popsize: int = 15,
    tol: float = 1e-6,
    mutation: Tuple[float, float] = (0.5, 1.0),
    recombination: float = 0.7,
) -> Dict:
    """Tuning bằng Differential Evolution (scipy).

    Args:
        obj: SimpObjective wrapper.
        n_iter: Số vòng lặp DE (mỗi vòng ~popsize*len(bounds) evaluations).
        seed: Random seed.
        popsize: Multiplier cho kích thước quần thể.
        tol: Tolerance hội tụ.
        mutation: (min, max) mutation rate.
        recombination: Crossover probability.

    Returns:
        Dict kết quả.
    """
    from scipy.optimize import differential_evolution

    bounds = obj.bounds
    param_names = obj.param_names
    n_params = len(bounds)

    # Normalized bounds cho DE (hoạt động trong [0,1])
    norm_bounds = [(0.0, 1.0)] * n_params

    t_start = time.time()

    def evaluate(x_norm):
        return _objective_wrapper(x_norm, bounds, obj)

    # Đảm bảo nó có bounds cho mỗi biến
    result = differential_evolution(
        evaluate,
        bounds=norm_bounds,
        maxiter=n_iter,
        popsize=popsize,
        tol=tol,
        mutation=mutation,
        recombination=recombination,
        seed=seed,
        polish=True,       # local refinement tại điểm tốt nhất
        disp=False,
        workers=1,
    )

    elapsed = time.time() - t_start

    # Denormalize kết quả
    best_x_raw = np.array([
        result.x[i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]
        for i in range(n_params)
    ])

    # Chạy evaluation cuối để chắc chắn có kết quả đầy đủ
    final_obj = obj(best_x_raw)
    best_entry = obj.get_best()

    return {
        'algorithm': 'differential_evolution',
        'n_evaluations': obj.n_eval,
        'elapsed': elapsed,
        'success': result.success,
        'message': result.message,
        'best_objective': float(result.fun) if not np.isnan(result.fun) else final_obj,
        'best_params': {k: float(best_x_raw[i]) for i, k in enumerate(param_names)},
        'best_entry': best_entry,
        'niter': result.nit,
        'nfev': result.nfev,
    }


def run_shgo(
    obj: SimpObjective,
    n_iter: int = 100,
    seed: int = 42,
    n: int = 100,
    iters: int = 3,
) -> Dict:
    """Tuning bằng SHGO (Simplicial Homology Global Optimization).

    Args:
        obj: SimpObjective wrapper.
        n_iter: Tổng số evaluations tối đa (approximate).
        seed: Random seed.
        n: Số mẫu khởi tạo mỗi vòng.
        iters: Số vòng SHGO.

    Returns:
        Dict kết quả.
    """
    from scipy.optimize import shgo

    bounds = obj.bounds
    param_names = obj.param_names

    t_start = time.time()

    def evaluate(x_raw):
        return obj(x_raw)

    # Chuyển bounds về list of tuples
    scipy_bounds = list(bounds)

    result = shgo(
        evaluate,
        bounds=scipy_bounds,
        n=n,
        iters=iters,
        seed=seed,
        sampling_method='sobol',
        minimize_every_iter=True,
    )

    elapsed = time.time() - t_start

    best_entry = obj.get_best()

    return {
        'algorithm': 'shgo',
        'n_evaluations': obj.n_eval,
        'elapsed': elapsed,
        'success': result.success,
        'message': str(result.message) if hasattr(result, 'message') else '',
        'best_objective': float(result.fun) if hasattr(result, 'fun') else (best_entry.get('objective', 1e10) if best_entry else 1e10),
        'best_params': (
            {k: float(result.x[i]) for i, k in enumerate(param_names)}
            if hasattr(result, 'x') and result.x is not None
            else best_entry.get('params', {})
        ),
        'best_entry': best_entry,
        'nfev': result.nfev if hasattr(result, 'nfev') else obj.n_eval,
    }


def run_basinhopping(
    obj: SimpObjective,
    n_iter: int = 50,
    seed: int = 42,
    stepsize: float = 0.3,
    T: float = 1.0,
    x0: Optional[np.ndarray] = None,
) -> Dict:
    """Tuning bằng Basinhopping (Stochastic + Local Refinement).

    Kết hợp global random step với local minimization (L-BFGS-B).

    Args:
        obj: SimpObjective wrapper.
        n_iter: Số vòng basinhopping.
        seed: Random seed.
        stepsize: Kích thước bước cho proposal distribution.
        T: Nhiệt độ cho Metropolis acceptance.
        x0: Điểm khởi tạo (None → random).

    Returns:
        Dict kết quả.
    """
    from scipy.optimize import basinhopping

    bounds = obj.bounds
    param_names = obj.param_names

    t_start = time.time()

    # Hàm mục tiêu cho basinhopping (nhận vector raw)
    def evaluate(x_raw):
        # Clip về bounds để an toàn
        x_clipped = np.clip(x_raw, [b[0] for b in bounds], [b[1] for b in bounds])
        return obj(x_clipped)

    # Khởi tạo
    if x0 is None:
        np.random.seed(seed)
        x0 = np.array([
            np.random.uniform(b[0], b[1]) for b in bounds
        ])

    # Minimizer local
    from scipy.optimize import minimize
    def local_minimizer(fun, x0_local, args=()):
        bounds_local = list(bounds)
        return minimize(
            fun, x0_local, args=args, method='L-BFGS-B',
            bounds=bounds_local,
            options={'maxiter': 10, 'ftol': 1e-3, 'gtol': 1e-3},
        )

    result = basinhopping(
        evaluate,
        x0,
        niter=n_iter,
        T=T,
        stepsize=stepsize,
        minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': list(bounds)},
        seed=seed,
        disp=False,
    )

    elapsed = time.time() - t_start

    best_entry = obj.get_best()

    return {
        'algorithm': 'basinhopping',
        'n_evaluations': obj.n_eval,
        'elapsed': elapsed,
        'success': True,
        'message': result.message[0] if hasattr(result, 'message') and result.message else '',
        'best_objective': float(result.fun) if hasattr(result, 'fun') else (best_entry.get('objective', 1e10) if best_entry else 1e10),
        'best_params': {k: float(result.x[i]) for i, k in enumerate(param_names)},
        'best_entry': best_entry,
        'nfev': result.nfev if hasattr(result, 'nfev') else obj.n_eval,
        'njev': result.njev if hasattr(result, 'njev') else 0,
    }


def run_refine_from_phase1(
    obj: SimpObjective,
    phase1_json: str,
    n_starting_points: int = 5,
    max_local_iters: int = 50,
) -> Dict:
    """Local refinement từ các điểm tốt nhất của Phase 1.

    Đọc kết quả Phase 1 JSON, lấy top-k mẫu tốt nhất,
    chạy L-BFGS-B local refinement từ mỗi điểm.

    Args:
        obj: SimpObjective wrapper.
        phase1_json: Path đến file JSON Phase 1.
        n_starting_points: Số điểm khởi tạo tốt nhất dùng.
        max_local_iters: Số vòng lặp tối đa cho mỗi local search.

    Returns:
        Dict kết quả.
    """
    from scipy.optimize import minimize

    bounds = obj.bounds
    param_names = obj.param_names

    # Đọc Phase 1 results
    try:
        with open(phase1_json) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f'  [WARN] Không đọc được Phase 1 JSON ({e}), dùng random start')
        return _run_random_refine(obj, n_starting_points, max_local_iters)

    results_raw = data.get('results', [])
    valid = [r for r in results_raw if r.get('success')]
    if len(valid) < 2:
        print(f'  [WARN] Chỉ có {len(valid)} mẫu Phase 1 thành công, dùng random start')
        return _run_random_refine(obj, n_starting_points, max_local_iters)

    # Sort theo objective (tăng dần = tốt hơn)
    sorted_results = sorted(valid, key=lambda r: r.get('obj_value', 1e10))

    t_start = time.time()
    all_starts = []

    for i in range(min(n_starting_points, len(sorted_results))):
        r = sorted_results[i]
        x0 = np.array([r['params'].get(p, 0.5) for p in param_names])
        # Clip
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])

        def evaluate(x_raw):
            return obj(x_raw)

        try:
            res = minimize(
                evaluate,
                x0,
                method='L-BFGS-B',
                bounds=list(bounds),
                options={'maxiter': max_local_iters, 'ftol': 1e-6, 'gtol': 1e-5},
            )
            all_starts.append({
                'start_idx': i,
                'start_objective': r.get('obj_value', 1e10),
                'start_params': {k: float(x0[j]) for j, k in enumerate(param_names)},
                'success': res.success,
                'final_objective': float(res.fun),
                'final_params': {k: float(res.x[j]) for j, k in enumerate(param_names)},
                'n_iter': res.nit,
                'n_eval_local': res.nfev,
            })
        except Exception as e:
            all_starts.append({
                'start_idx': i,
                'start_objective': r.get('obj_value', 1e10),
                'start_params': {k: float(x0[j]) for j, k in enumerate(param_names)},
                'success': False,
                'final_objective': None,
                'error': str(e),
            })

    elapsed = time.time() - t_start

    # Kết quả tốt nhất
    valid_starts = [s for s in all_starts if s.get('success') and s.get('final_objective') is not None]
    best_entry = obj.get_best()
    best_of_all = None
    if valid_starts:
        best_of_all = min(valid_starts, key=lambda s: s['final_objective'])

    return {
        'algorithm': 'refine_lbfgsb',
        'n_evaluations': obj.n_eval,
        'n_starting_points': n_starting_points,
        'n_valid_starts': len(valid_starts),
        'elapsed': elapsed,
        'best_objective': best_of_all['final_objective'] if best_of_all else (best_entry.get('objective', 1e10) if best_entry else 1e10),
        'best_params': best_of_all['final_params'] if best_of_all else (best_entry.get('params', {}) if best_entry else {}),
        'best_entry': best_entry,
        'starting_points_details': all_starts,
    }


def _run_random_refine(
    obj: SimpObjective,
    n_starts: int = 5,
    max_local_iters: int = 50,
) -> Dict:
    """Fallback: local refinement từ random points."""
    from scipy.optimize import minimize

    bounds = obj.bounds
    param_names = obj.param_names
    np.random.seed(42)

    t_start = time.time()
    all_starts = []

    for i in range(n_starts):
        x0 = np.array([np.random.uniform(b[0], b[1]) for b in bounds])

        def evaluate(x_raw):
            return obj(x_raw)

        try:
            res = minimize(
                evaluate,
                x0,
                method='L-BFGS-B',
                bounds=list(bounds),
                options={'maxiter': max_local_iters, 'ftol': 1e-6, 'gtol': 1e-5},
            )
            all_starts.append({
                'start_idx': i,
                'start_objective': None,
                'start_params': {k: float(x0[j]) for j, k in enumerate(param_names)},
                'success': res.success,
                'final_objective': float(res.fun),
                'final_params': {k: float(res.x[j]) for j, k in enumerate(param_names)},
                'n_iter': res.nit,
                'n_eval_local': res.nfev,
            })
        except Exception as e:
            all_starts.append({
                'start_idx': i,
                'start_params': {k: float(x0[j]) for j, k in enumerate(param_names)},
                'success': False,
                'error': str(e),
            })

    elapsed = time.time() - t_start
    valid_starts = [s for s in all_starts if s.get('success') and s.get('final_objective') is not None]
    best_entry = obj.get_best()
    best_of_all = min(valid_starts, key=lambda s: s['final_objective']) if valid_starts else None

    return {
        'algorithm': 'random_refine_lbfgsb',
        'n_evaluations': obj.n_eval,
        'n_starting_points': n_starts,
        'n_valid_starts': len(valid_starts),
        'elapsed': elapsed,
        'best_objective': best_of_all['final_objective'] if best_of_all else (best_entry.get('objective', 1e10) if best_entry else 1e10),
        'best_params': best_of_all['final_params'] if best_of_all else (best_entry.get('params', {}) if best_entry else {}),
        'best_entry': best_entry,
        'starting_points_details': all_starts,
    }


# ──────────────────────────────────────────────
#  Save & Report
# ──────────────────────────────────────────────

METHODS = {
    'de': run_differential_evolution,
    'shgo': run_shgo,
    'basinhopping': run_basinhopping,
    'refine': run_refine_from_phase1,
}

METHOD_DESCRIPTIONS = {
    'de': 'Differential Evolution - global search, robust với constraints',
    'shgo': 'Simplicial Homology Global Optimization - alternative global method',
    'basinhopping': 'Basinhopping - stochastic global + local refinement (L-BFGS-B)',
    'refine': 'L-BFGS-B local refinement từ Phase 1 best points',
}

METHOD_DEFAULTS = {
    'de': {'n_iter': 80, 'popsize': 15},
    'shgo': {'n_iter': 100, 'n': 80, 'iters': 3},
    'basinhopping': {'n_iter': 40, 'stepsize': 0.3, 'T': 1.0},
    'refine': {'n_starting_points': 5, 'max_local_iters': 50},
}


def save_phase2_results(
    result: Dict,
    output_dir: str,
    seed_name: str,
    objective: str,
    algorithm: str,
    objective_obj: SimpObjective,
) -> str:
    """Lưu kết quả Phase 2 ra JSON + summary.

    Returns:
        Path đến file JSON.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Chuẩn hóa dữ liệu cho JSON
    eval_history_serializable = []
    for e in objective_obj.eval_history:
        entry = dict(e)
        if entry.get('params'):
            entry['params'] = {k: float(v) for k, v in entry['params'].items()}
        if entry.get('error'):
            entry['error'] = str(entry['error'])
        eval_history_serializable.append(entry)

    payload = {
        'metadata': {
            'phase': 'phase2',
            'algorithm': algorithm,
            'objective': objective,
            'seed': seed_name,
            'timestamp': datetime.now().isoformat(),
            'n_evals': objective_obj.n_eval,
            'n_success': sum(1 for e in eval_history_serializable if e.get('success')),
            'param_names': objective_obj.param_names,
            'param_bounds': objective_obj.bounds,
        },
        'result': {
            'algorithm': result.get('algorithm', algorithm),
            'elapsed': result.get('elapsed', 0),
            'n_evaluations': result.get('n_evaluations', objective_obj.n_eval),
            'success': result.get('success', True),
            'message': result.get('message', ''),
            'best_objective': result.get('best_objective', None),
            'best_params': result.get('best_params', {}),
            'niter': result.get('niter', None),
            'nfev': result.get('nfev', None),
            # Lưu extra theo algorithm
            'details': {k: v for k, v in result.items()
                        if k not in ['best_params', 'best_entry',
                                      'best_objective', 'algorithm',
                                      'n_evaluations', 'elapsed',
                                      'success', 'message',
                                      'niter', 'nfev']},
        },
        'summary': objective_obj.summary(),
        'eval_history': eval_history_serializable,
    }

    json_path = os.path.join(
        output_dir,
        f'phase2_{seed_name}_{objective}_{algorithm}.json',
    )
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'  → JSON: {json_path}')

    return json_path


# ──────────────────────────────────────────────
#  Run single combo
# ──────────────────────────────────────────────

def run_phase2(
    objective: str,
    seed_name: str,
    method: str = 'de',
    n_iter: Optional[int] = None,
    output_base: str = 'outputs/pipeline/phase2',
    verbose: bool = True,
    max_simp_iters: int = 150,
    **method_kwargs,
) -> Dict:
    """Chạy Phase 2 tuning cho một cặp (seed, objective).

    Args:
        objective: 'auxetic' | 'first' | 'second'.
        seed_name: Tên seed.
        method: Tên thuật toán ('de', 'shgo', 'basinhopping', 'refine').
        n_iter: Override số vòng lặp (None → dùng default của method).
        output_base: Thư mục gốc đầu ra.
        verbose: In log chi tiết.
        max_simp_iters: Số vòng lặp SIMP tối đa.
        **method_kwargs: Tham số bổ sung cho optimizer.

    Returns:
        Dict kết quả tuning.
    """
    seed_obj_dir = os.path.join(output_base, seed_name, objective)
    os.makedirs(seed_obj_dir, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'Phase 2: Tuning - {seed_name} / {objective}')
    print(f'  Algorithm: {method} ({METHOD_DESCRIPTIONS.get(method, "")})')
    print(f'  Output:    {seed_obj_dir}')
    print(f'  SIMP iters: {max_simp_iters}')
    print(f'='*60)

    # Khởi tạo objective wrapper
    fixed = dict(FIXED_PARAMS)
    fixed['max_iter'] = max_simp_iters
    simp_obj = SimpObjective(
        seed_name=seed_name,
        objective=objective,
        fixed=fixed,
        output_base=output_base,
        verbose=verbose,
    )

    # Lấy default params cho method nếu không override
    effective_kwargs = dict(METHOD_DEFAULTS.get(method, {}))
    if n_iter is not None:
        effective_kwargs['n_iter'] = n_iter
    effective_kwargs.update(method_kwargs)

    # Nếu method là 'refine', tìm Phase 1 JSON
    if method == 'refine':
        phase1_json = os.path.join(
            'outputs/pipeline/phase1', seed_name, objective,
            f'phase1_{seed_name}_{objective}.json',
        )
        effective_kwargs['phase1_json'] = phase1_json

    # Chạy optimizer
    func = METHODS.get(method)
    if func is None:
        raise ValueError(f'Unknown method: {method}. Chọn: {list(METHODS.keys())}')

    result = func(simp_obj, **effective_kwargs)

    # Lưu kết quả
    json_path = save_phase2_results(
        result, seed_obj_dir, seed_name, objective, method, simp_obj,
    )

    # In summary
    print(f'\n{"─"*60}')
    print(f'Summary: {seed_name} / {objective} / {method}')
    print(f'  Evaluations: {simp_obj.n_eval}')
    print(f'  Best obj:    {result["best_objective"]:.4e}')
    if result.get('best_params'):
        print(f'  Best params:')
        for k, v in result['best_params'].items():
            print(f'    {k} = {v:.4f}')
    if result.get('niter') is not None:
        print(f'  Iterations:  {result["niter"]}')
    if result.get('elapsed'):
        print(f'  Elapsed:     {result["elapsed"]:.1f}s')
    print(f'{"─"*60}')

    result['_objective_obj'] = simp_obj
    result['_json_path'] = json_path
    return result


def run_all_combinations(
    method: str = 'de',
    n_iter: Optional[int] = None,
    seeds_subset: Optional[List[str]] = None,
    objectives_subset: Optional[List[str]] = None,
    max_simp_iters: int = 150,
) -> List[Dict]:
    """Chạy Phase 2 cho nhiều combo (seed × objective).

    Args:
        method: Thuật toán.
        n_iter: Số vòng lặp (None → default).
        seeds_subset: Danh sách seeds (None → tất cả).
        objectives_subset: Danh sách objectives (None → tất cả).
        max_simp_iters: Số vòng lặp SIMP tối đa.

    Returns:
        List kết quả.
    """
    targets = []
    seeds = seeds_subset or SEEDS
    objs = objectives_subset or OBJECTIVES

    for obj in objs:
        for seed in seeds:
            targets.append((obj, seed))

    all_results = []
    for obj, seed in targets:
        try:
            result = run_phase2(
                objective=obj,
                seed_name=seed,
                method=method,
                n_iter=n_iter,
                verbose=True,
                max_simp_iters=max_simp_iters,
            )
            all_results.append(result)
        except Exception as e:
            print(f'\n[ERROR] {seed}/{obj}: {e}')
            all_results.append({
                'objective': obj,
                'seed': seed,
                'error': str(e),
            })

    # Ghi tổng hợp
    output_dir = f'outputs/pipeline/phase2'
    os.makedirs(output_dir, exist_ok=True)
    serializable = []
    for r in all_results:
        entry = {
            'objective': r.get('objective'),
            'seed': r.get('seed'),
            'algorithm': r.get('algorithm', method),
            'n_evaluations': r.get('n_evaluations'),
            'best_objective': r.get('best_objective'),
            'best_params': r.get('best_params'),
            'elapsed': r.get('elapsed'),
            'error': r.get('error'),
        }
        serializable.append(entry)

    summary_path = os.path.join(output_dir, '_all_phase2_summaries.json')
    with open(summary_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f'\nTotal summary: {summary_path}')

    return all_results


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Phase 2: Tuning tham số với optimization algorithms',
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
        '--method', type=str, default='de',
        choices=list(METHODS.keys()),
        help='Thuật toán optimization (mặc định: de)',
    )
    parser.add_argument(
        '--n_iter', type=int, default=None,
        help='Số vòng lặp optimizer (mặc định: theo method)',
    )
    parser.add_argument(
        '--max_simp_iters', type=int, default=150,
        help='Số vòng lặp SIMP tối đa (mặc định: 150)',
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Quét nhiều combo (seed × objective)',
    )
    parser.add_argument(
        '--seeds', type=str, nargs='+', default=None,
        choices=SEEDS,
        help='Filter seeds (dùng với --all)',
    )
    parser.add_argument(
        '--objectives', type=str, nargs='+', default=None,
        choices=OBJECTIVES,
        help='Filter objectives (dùng với --all)',
    )
    parser.add_argument(
        '--output', type=str, default='outputs/pipeline/phase2',
        help='Thư mục đầu ra gốc',
    )
    parser.add_argument(
        '--verbose', action='store_true', default=True,
        help='In log chi tiết',
    )
    parser.add_argument(
        '--popsize', type=int, default=15,
        help='Popsize cho DE (mặc định: 15)',
    )
    # method-specific extras
    parser.add_argument(
        '--stepsize', type=float, default=0.3,
        help='Stepsize cho basinhopping (mặc định: 0.3)',
    )
    parser.add_argument(
        '--n_starting_points', type=int, default=5,
        help='Số starting points cho refine (mặc định: 5)',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Build method kwargs
    method_kwargs = {}
    if args.method == 'de':
        method_kwargs['popsize'] = args.popsize
    elif args.method == 'basinhopping':
        method_kwargs['stepsize'] = args.stepsize
    elif args.method == 'refine':
        method_kwargs['n_starting_points'] = args.n_starting_points

    if args.all:
        run_all_combinations(
            method=args.method,
            n_iter=args.n_iter,
            seeds_subset=args.seeds,
            objectives_subset=args.objectives,
            max_simp_iters=args.max_simp_iters,
        )
    else:
        run_phase2(
            objective=args.objective,
            seed_name=args.seed,
            method=args.method,
            n_iter=args.n_iter,
            output_base=args.output,
            verbose=args.verbose,
            max_simp_iters=args.max_simp_iters,
            **method_kwargs,
        )


if __name__ == '__main__':
    main()
