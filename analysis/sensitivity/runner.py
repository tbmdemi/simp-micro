"""
Runner / Orchestrator for Sensitivity Analysis
================================================
Điều phối toàn bộ quy trình phân tích độ nhạy trên tất cả seed × objective.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from analysis.sensitivity import regression, sobol, anova, classify, visualize

# ──────────────────────────────────────────────
#  Cấu hình mặc định
# ──────────────────────────────────────────────
PHASE1_DIR = Path('outputs/pipeline/phase1')

PARAM_COLS = ['volfrac', 'penal', 'rmin', 'move', 'void_size_frac', 'rotation_deg']
PARAM_COLS_FIRST = PARAM_COLS + ['beta']
PARAM_COLS_SECOND = PARAM_COLS + ['beta_second']

PARAM_BOUNDS: Dict[str, List[Tuple[float, float]]] = {
    'auxetic': [
        (0.2, 0.6), (1.0, 5.0), (1.0, 6.0), (0.05, 0.3),
        (0.2, 0.7), (0.0, 90.0),
    ],
    'first': [
        (0.2, 0.6), (1.0, 5.0), (1.0, 6.0), (0.05, 0.3),
        (0.2, 0.7), (0.0, 90.0), (0.3, 1.5),
    ],
    'second': [
        (0.2, 0.6), (1.0, 5.0), (1.0, 6.0), (0.05, 0.3),
        (0.2, 0.7), (0.0, 90.0), (0.5, 2.5),
    ],
}


def get_param_cols(objective: str) -> List[str]:
    """Lấy danh sách cột tham số theo objective."""
    if objective == 'first':
        return PARAM_COLS_FIRST
    elif objective == 'second':
        return PARAM_COLS_SECOND
    return PARAM_COLS


def find_csv_files(
    seed: str,
    objective: str,
) -> List[Path]:
    """Tìm các file CSV Phase 1 cho seed và objective.

    Returns:
        Danh sách Path đến các file CSV (thường là 1 file).
    """
    pattern = f'phase1_{seed}_{objective}.csv'
    paths = sorted(PHASE1_DIR.rglob(pattern))
    if not paths:
        # Fallback: tìm theo cấu trúc thư mục seed/objective/
        alt = PHASE1_DIR / seed / objective
        if alt.exists():
            paths = sorted(alt.glob('*.csv'))
    return paths


def run_sensitivity_for_seed_objective(
    csv_path: Path,
    objective: str,
    output_dir: Optional[Path] = None,
) -> Dict:
    """Chạy toàn bộ phân tích độ nhạy cho một file CSV.

    Args:
        csv_path: Đường dẫn file CSV.
        objective: 'auxetic', 'first', hoặc 'second'.
        output_dir: Thư mục lưu ảnh (nếu None, không lưu).

    Returns:
        Dict tổng hợp kết quả.
    """
    param_cols = get_param_cols(objective)
    bounds = PARAM_BOUNDS.get(objective, PARAM_BOUNDS['auxetic'])

    print(f'  [Sensitivity] Processing: {csv_path.name}')

    # 1. SRC
    src_result = regression.compute_src_from_csv(
        str(csv_path), param_cols, obj_col='obj_value',
    )

    # 2. Sobol
    sobol_result = sobol.compute_sobol_from_csv(
        str(csv_path), param_cols, bounds, obj_col='obj_value',
    )

    # 3. ANOVA
    anova_result = anova.compute_anova_from_csv(
        str(csv_path), param_cols, obj_col='obj_value',
    )

    # 4. Classification
    src_coef = src_result.get('coef', {})
    sobol_st = sobol_result.get('ST', {})
    sobol_s1 = sobol_result.get('S1', {})

    classification_result = classify.classify_parameters(
        src_coef, sobol_st, sobol_s1,
    )
    classification_summary = classify.summarize_classification(
        classification_result,
    )

    result = {
        'csv_file': str(csv_path),
        'objective': objective,
        'param_cols': param_cols,
        'n_samples': len(pd.read_csv(csv_path)),
        'src': src_result,
        'sobol': sobol_result,
        'anova': anova_result,
        'classification': classification_result,
        'classification_summary': classification_summary,
    }

    # Vẽ và lưu ảnh nếu có output_dir
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = csv_path.stem.replace('phase1_', '')

        try:
            fig = visualize.plot_src(src_result)
            fig.savefig(output_dir / f'src_{stem}.png', dpi=150, bbox_inches='tight')
            plt = __import__('matplotlib.pyplot', fromlist=[''])
            plt.close(fig)
        except Exception as e:
            print(f'    [WARN] plot_src failed: {e}')

        try:
            fig = visualize.plot_sobol(sobol_result)
            fig.savefig(output_dir / f'sobol_{stem}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f'    [WARN] plot_sobol failed: {e}')

        try:
            fig = visualize.plot_anova(anova_result)
            fig.savefig(output_dir / f'anova_{stem}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f'    [WARN] plot_anova failed: {e}')

        try:
            fig = visualize.plot_parameter_classification(classification_result)
            fig.savefig(output_dir / f'classification_{stem}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f'    [WARN] plot_classification failed: {e}')

    return result


def run_all_sensitivities(
    seeds: Optional[List[str]] = None,
    objectives: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    json_output: Optional[Path] = None,
) -> Dict:
    """Chạy sensitivity analysis cho tất cả seed × objective.

    Args:
        seeds: Danh sách seed (mặc định: tất cả seeds có dữ liệu).
        objectives: Danh sách objective.
        output_dir: Thư mục lưu ảnh.
        json_output: Đường dẫn lưu kết quả JSON tổng hợp.

    Returns:
        Dict {f'{seed}/{objective}': result}.
    """
    if seeds is None:
        seeds = [
            d.name for d in PHASE1_DIR.iterdir()
            if d.is_dir() and not d.name.startswith('_')
        ]
        if not seeds:
            # Fallback: tìm trong các file CSV
            seeds = sorted(set(
                f.name.split('_')[1] for f in PHASE1_DIR.rglob('phase1_*.csv')
            ))
    if objectives is None:
        objectives = ['auxetic', 'first', 'second']

    all_results: Dict = {}

    for seed in seeds:
        for obj in objectives:
            csv_files = find_csv_files(seed, obj)
            if not csv_files:
                print(f'  [SKIP] No CSV for {seed}/{obj}')
                continue
            key = f'{seed}/{obj}'
            all_results[key] = run_sensitivity_for_seed_objective(
                csv_files[0], obj, output_dir=output_dir,
            )

    # Lưu JSON nếu yêu cầu
    if json_output is not None:
        json_output = Path(json_output)
        json_output.parent.mkdir(parents=True, exist_ok=True)

        # Chuyển ndarray và các kiểu không JSON-serializable
        def _convert(o):
            if isinstance(o, (np.ndarray,)):
                return o.tolist()
            return o

        import numpy as np
        with open(json_output, 'w') as f:
            json.dump(all_results, f, indent=2, default=_convert)
        print(f'  [SAVE] Results saved to {json_output}')

    print(f'  [DONE] Sensitivity analysis complete: {len(all_results)} runs')

    return all_results


def print_summary(all_results: Dict) -> None:
    """In bảng tóm tắt kết quả phân loại tham số."""
    print('\n' + '=' * 70)
    print('  PARAMETER SENSITIVITY CLASSIFICATION SUMMARY')
    print('=' * 70)
    for key, res in sorted(all_results.items()):
        summary = res.get('classification_summary', {})
        n = res.get('n_samples', 0)
        print(f'\n  [{key}]  (n={n})')
        for cls, params in summary.items():
            print(f'    {cls:22s}: {", ".join(params)}')