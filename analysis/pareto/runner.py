"""
Runner / Orchestrator for Pareto Analysis
===========================================
Điều phối phân tích Pareto front trên dữ liệu Phase 1.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.pareto import frontier, visualize


def run_pareto_for_seed(
    csv_path: Path,
    obj_cols: List[str],
    output_dir: Optional[Path] = None,
    maximize: bool = False,
    seed_label: Optional[str] = None,
) -> Dict:
    """Chạy Pareto analysis cho một file CSV.

    Args:
        csv_path: Đường dẫn file CSV.
        obj_cols: Các cột objective (2 hoặc 3).
        output_dir: Thư mục lưu ảnh.
        maximize: True nếu maximize objectives.
        seed_label: Nhãn seed (in trong log).

    Returns:
        Dict kết quả.
    """
    label = seed_label or csv_path.stem
    print(f'  [Pareto] Processing: {label}')

    df = pd.read_csv(csv_path)

    front_result = frontier.compute_pareto_front(df, obj_cols, maximize=maximize)

    result = {
        'csv_file': str(csv_path),
        'n_total': front_result['n_total'],
        'n_frontier': front_result['n_frontier'],
        'ratio': front_result['n_frontier'] / max(front_result['n_total'], 1),
        'params': front_result['params'],
        'obj_cols': obj_cols,
    }

    # Hypervolume
    if front_result['n_frontier'] >= 2:
        front_values = front_result['frontier'][obj_cols].values.astype(float)
        hv = frontier.compute_hypervolume(front_values)
        result['hypervolume'] = hv

    # Vẽ và lưu ảnh
    if output_dir is not None and front_result['n_frontier'] > 0:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if len(obj_cols) == 2:
            fig = visualize.plot_pareto_front_2d(
                df, obj_cols, front_result['mask'], maximize=maximize,
            )
            fig.savefig(
                output_dir / f'pareto_{label.replace("/", "_")}.png',
                dpi=150, bbox_inches='tight',
            )
            import matplotlib.pyplot as plt
            plt.close(fig)

    return result


def run_all_pareto(
    seeds: Optional[List[str]] = None,
    objectives: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    json_output: Optional[Path] = None,
) -> Dict:
    """Chạy Pareto analysis cho tất cả seed × objective.

    Args:
        seeds: Danh sách seed.
        objectives: Danh sách objective loại.
        output_dir: Thư mục lưu ảnh.
        json_output: Đường dẫn lưu JSON.

    Returns:
        Dict {f'{seed}/{obj}': result}.
    """
    if seeds is None:
        from analysis.sensitivity.runner import find_csv_files
        # Use same CSV discovery logic
    if objectives is None:
        objectives = ['auxetic']

    phase1_dir = Path('outputs/pipeline/phase1')
    if seeds is None:
        seeds = sorted(set(
            f.name.split('_')[1] for f in phase1_dir.rglob('phase1_*.csv')
        ))

    # Map objective → cột
    obj_map = {
        'auxetic': ['v12', 'auxetic_ratio'],
    }

    all_results: Dict = {}

    for seed in seeds:
        for obj in objectives:
            obj_cols = obj_map.get(obj, ['obj_value'])
            pattern = f'phase1_{seed}_{obj}.csv'
            csv_files = sorted(phase1_dir.rglob(pattern))
            if not csv_files:
                print(f'  [SKIP] No CSV for {seed}/{obj}')
                continue
            key = f'{seed}/{obj}'
            all_results[key] = run_pareto_for_seed(
                csv_files[0], obj_cols, output_dir=output_dir,
                seed_label=key,
            )

    # Lưu JSON
    if json_output is not None:
        json_output = Path(json_output)
        json_output.parent.mkdir(parents=True, exist_ok=True)
        with open(json_output, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f'  [SAVE] Pareto results saved to {json_output}')

    return all_results


def print_pareto_summary(all_results: Dict) -> None:
    """In bảng tóm tắt Pareto."""
    print('\n' + '=' * 70)
    print('  PARETO FRONT SUMMARY')
    print('=' * 70)
    for key, res in sorted(all_results.items()):
        hv = res.get('hypervolume', 'N/A')
        hv_str = f'{hv:.4f}' if isinstance(hv, float) else str(hv)
        print(f'  {key:30s} | n={res["n_total"]:4d} | front={res["n_frontier"]:3d} | HV={hv_str}')