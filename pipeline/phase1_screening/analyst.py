"""
Phase 1 Analyst — Aggregation Script.

Quét toàn bộ dữ liệu Phase 1 (outputs/pipeline/phase1/) và tạo hai file tổng hợp:
  1. _all_correlations.json — hệ số tương quan Spearman cho mỗi cặp (seed, objective)
  2. _all_summaries_parallel.json — tóm tắt gọn: top 3 tham số, best obj, elapsed time...

Luồng xử lý: phát hiện seeds -> tìm CSV tổng hợp phase1_{seed}_{objective}.csv
(fallback đọc từ thư mục sample_* nếu thiếu) -> tự động phát hiện param_names ->
tính Spearman correlation với obj_value -> ghi hai file JSON trên.
"""

import json
import os
import sys
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ──────────────────────────────────────────────
#  Cấu hình
# ──────────────────────────────────────────────

# Các cột metadata (không phải tham số) trong CSV tổng hợp
METADATA_COLUMNS: List[str] = [
    'sample_id', 'success', 'v12', 'v21', 'obj_value',
    'n_iters', 'converged', 'elapsed_time', 'error',
]

# Các tham số kỹ thuật cố định cần loại bỏ (dự phòng nếu xuất hiện)
FIXED_PARAM_KEYS: List[str] = [
    'nelx', 'nely', 'ft', 'E0', 'Emin', 'nu', 'max_iter',
    'tol_change', 'tol_obj', 'window_size', 'save_every',
    'scale_factor', 'mu', 'beta', 'rotation_deg', 'seed', 'objective',
]


# ──────────────────────────────────────────────
#  Bước 1: Phát hiện seeds
# ──────────────────────────────────────────────

def discover_seeds(root_dir: str) -> List[str]:
    """Quét thư mục phase1 và trả về danh sách seed có dữ liệu.

    Args:
        root_dir: Đường dẫn đến thư mục outputs/pipeline/phase1

    Returns:
        Danh sách tên seed (tên thư mục con), sắp xếp alphabet.
    """
    entries = sorted(os.listdir(root_dir))
    seeds: List[str] = []
    for entry in entries:
        full_path = os.path.join(root_dir, entry)
        if os.path.isdir(full_path) and not entry.startswith('_'):
            seeds.append(entry)
    return seeds


# ──────────────────────────────────────────────
#  Bước 2: Phát hiện configs (seed, objective)
# ──────────────────────────────────────────────

def discover_configs(seed_dir: str, seed: str) -> List[Dict[str, Any]]:
    """Phát hiện các cặp (seed, objective) có dữ liệu.

    Ưu tiên đọc file CSV tổng hợp. Nếu không có, tìm sample_* directories.

    Args:
        seed_dir: Thư mục của seed (ví dụ: outputs/pipeline/phase1/circle)
        seed: Tên seed (ví dụ: 'circle')

    Returns:
        Danh sách dict: [{'seed': ..., 'objective': ..., 'csv_path': ...}]
    """
    configs: List[Dict[str, Any]] = []

    # Các objective đã biết trong pipeline Phase 1
    KNOWN_OBJECTIVES = ['auxetic', 'first', 'second']

    # Tìm file CSV tổng hợp — pattern phase1_{seed}_{objective}.csv
    # Vì seed có thể chứa underscore (vd: circle_half_quarter) nên không thể
    # split('_') đơn giản. Thay vào đó, thử từng known objective làm suffix.
    csv_files = glob(os.path.join(seed_dir, 'phase1_*.csv'))
    for csv_path in csv_files:
        basename = os.path.basename(csv_path)          # phase1_circle_half_quarter_auxetic.csv
        rest = basename.replace('phase1_', '', 1).replace('.csv', '')

        matched = False
        for known_obj in KNOWN_OBJECTIVES:
            suffix = f'_{known_obj}'
            if rest.endswith(suffix):
                potential_seed = rest[:-len(suffix)]   # circle_half_quarter
                if potential_seed == seed:
                    configs.append({
                        'seed': seed,
                        'objective': known_obj,
                        'csv_path': csv_path,
                    })
                    matched = True
                    break

        # Fallback nếu filename không theo pattern chuẩn (vd: phase1_circle_first.csv):
        # thử split('_', 1) đơn giản.
        if not matched:
            parts = rest.split('_', 1)
            if len(parts) == 2:
                fseed, fobjective = parts
                if fseed == seed:
                    configs.append({
                        'seed': seed,
                        'objective': fobjective,
                        'csv_path': csv_path,
                    })

    # Fallback: nếu không có CSV tổng hợp, tìm sample_* directories
    if not configs:
        sample_dirs = sorted(glob(os.path.join(seed_dir, 'sample_*')))
        if sample_dirs:
            # Đoán objective từ sample_0000/metadata.json nếu có
            # Mặc định là 'auxetic' (objective mặc định của Phase 1)
            objective = 'auxetic'
            for smp in sample_dirs:
                meta_path = os.path.join(smp, 'metadata.json')
                if os.path.isfile(meta_path):
                    try:
                        with open(meta_path, 'r') as f:
                            meta = json.load(f)
                        obj = meta.get('objective', meta.get('params', {}).get('objective'))
                        if obj:
                            objective = obj
                            break
                    except Exception:
                        pass
            configs.append({
                'seed': seed,
                'objective': objective,
                'sample_dirs': sample_dirs,
            })

    return configs


# ──────────────────────────────────────────────
#  Bước 3: Đọc dữ liệu từ CSV hoặc sample_*
# ──────────────────────────────────────────────

def load_data_from_csv(csv_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """Đọc dữ liệu từ file CSV tổng hợp.

    Args:
        csv_path: Đường dẫn file CSV

    Returns:
        Tuple (DataFrame với cột obj_value + các cột param, danh sách param_names)
    """
    df = pd.read_csv(csv_path)

    # Xác định cột objective
    obj_col = 'obj_value'
    if obj_col not in df.columns:
        # Fallback: tìm cột chứa 'obj'
        obj_candidates = [c for c in df.columns if 'obj' in c.lower()]
        obj_col = obj_candidates[0] if obj_candidates else None
        if obj_col is None:
            raise ValueError(f"Không tìm thấy cột objective trong {csv_path}")

    # Xác định param_names: lấy các cột không phải metadata
    all_cols = list(df.columns)
    param_names = [
        c for c in all_cols
        if c not in METADATA_COLUMNS and c != obj_col
    ]

    # Lọc: chỉ giữ param có ít nhất 2 giá trị khác nhau
    param_names = [
        p for p in param_names
        if df[p].nunique() >= 2
    ]

    # Lọc bỏ các param kỹ thuật cố định (nếu sót)
    param_names = [p for p in param_names if p not in FIXED_PARAM_KEYS]

    # Tạo DataFrame chỉ giữ các cột cần thiết
    cols_to_keep = param_names + [obj_col]
    df_out = df[cols_to_keep].copy()

    # Chuyển đổi kiểu dữ liệu
    for col in df_out.columns:
        df_out[col] = pd.to_numeric(df_out[col], errors='coerce')

    # Drop NaN
    df_out = df_out.dropna()

    return df_out, param_names


def load_run_metadata(csv_path: str) -> Dict[str, Optional[float]]:
    """Đọc elapsed_time / n_workers thật từ file JSON song hành với CSV.

    CSV tổng hợp `phase1_{seed}_{objective}.csv` luôn đi kèm file JSON cùng
    tên (`.json`) do `save_results()` trong phase1_screening/screening_parallel.py ghi
    ra. File JSON này chứa `metadata.elapsed_time` (tổng thời gian chạy,
    đơn vị giây) và `metadata.n_workers` — không thể suy ra 2 giá trị này
    một cách đáng tin cậy chỉ từ CSV, nên đọc trực tiếp từ JSON nếu có.

    Args:
        csv_path: Đường dẫn file CSV tổng hợp.

    Returns:
        Dict {'elapsed_time': float | None, 'n_workers': int | None}.
        Trả về None cho cả hai nếu không tìm thấy JSON song hành.
    """
    json_path = csv_path[:-len('.csv')] + '.json' if csv_path.endswith('.csv') else csv_path + '.json'
    if not os.path.isfile(json_path):
        return {'elapsed_time': None, 'n_workers': None}
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        meta = data.get('metadata', {})
        return {
            'elapsed_time': meta.get('elapsed_time'),
            'n_workers': meta.get('n_workers'),
        }
    except Exception:
        return {'elapsed_time': None, 'n_workers': None}


def load_data_from_samples(sample_dirs: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Fallback: đọc dữ liệu từ sample_* directories.

    Args:
        sample_dirs: Danh sách đường dẫn thư mục sample_*

    Returns:
        Tuple (DataFrame, param_names)
    """
    records: List[Dict[str, Any]] = []
    all_param_keys: set = set()

    for sample_path in sample_dirs:
        # Đọc metadata.json
        meta_path = os.path.join(sample_path, 'metadata.json')
        if not os.path.isfile(meta_path):
            continue

        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        except Exception:
            continue

        # Lấy params
        params = {}
        if 'params' in meta:
            params = meta['params']
        elif 'parameters' in meta:
            params = meta['parameters']

        if not params:
            continue

        # Lấy objective
        objective = params.get('objective', meta.get('objective', 'auxetic'))

        # Đọc iteration_data.csv → lấy dòng cuối, cột Objective
        iter_path = os.path.join(sample_path, 'iteration_data.csv')
        if not os.path.isfile(iter_path):
            continue

        try:
            iter_df = pd.read_csv(iter_path)
            obj_value = iter_df['Objective'].iloc[-1]
        except Exception:
            continue

        # Ghi nhận
        record = {k: v for k, v in params.items()}
        record['obj_value'] = obj_value
        records.append(record)
        all_param_keys.update(params.keys())

    if not records:
        raise ValueError("Không thể đọc dữ liệu từ sample directories")

    # Lọc param_names
    exclude_keys = set(FIXED_PARAM_KEYS)
    param_candidates = [k for k in all_param_keys if k not in exclude_keys]

    # Kiểm tra biến thiên
    param_names: List[str] = []
    values_dict: Dict[str, list] = {k: [] for k in param_candidates}
    for rec in records:
        for k in param_candidates:
            v = rec.get(k)
            try:
                values_dict[k].append(float(v))
            except (TypeError, ValueError):
                values_dict[k].append(np.nan)

    for k in param_candidates:
        arr = np.array(values_dict[k])
        arr = arr[~np.isnan(arr)]
        if len(np.unique(arr)) >= 2:
            param_names.append(k)

    # Tạo DataFrame
    rows: List[Dict[str, Any]] = []
    for rec in records:
        row: Dict[str, Any] = {'obj_value': rec['obj_value']}
        for p in param_names:
            val = rec.get(p)
            try:
                row[p] = float(val)
            except (TypeError, ValueError):
                row[p] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.dropna()

    return df, param_names


def load_data(config: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str], str, str]:
    """Đọc dữ liệu cho một config.

    Args:
        config: Dict từ discover_configs

    Returns:
        Tuple (DataFrame, param_names, seed, objective)
    """
    seed = config['seed']
    objective = config['objective']

    if 'csv_path' in config:
        df, param_names = load_data_from_csv(config['csv_path'])
    elif 'sample_dirs' in config:
        df, param_names = load_data_from_samples(config['sample_dirs'])
    else:
        raise ValueError(f"Config không có csv_path hoặc sample_dirs: {config}")

    return df, param_names, seed, objective


# ──────────────────────────────────────────────
#  Bước 4: Tính tương quan Spearman
# ──────────────────────────────────────────────

def compute_correlations(
    df: pd.DataFrame,
    param_names: List[str],
    objective: str,
) -> Tuple[List[float], List[float], List[List[Any]], float, int]:
    """Tính Spearman correlation giữa từng param và obj_value.

    Dùng Spearman (rank-based) thay vì Pearson vì quan hệ giữa tham số SIMP
    và obj_value không được giả định là tuyến tính — khớp với phương pháp
    dùng trong pipeline/phase1_screening/screening_parallel.py (nguồn dữ liệu thật).

    Args:
        df: DataFrame với cột 'obj_value' và các cột param
        param_names: Danh sách tên tham số
        objective: Tên objective (để xác định best là max hay min)

    Returns:
        Tuple:
          - corr_list: hệ số tương quan (theo thứ tự param_names)
          - pval_list: p-value tương ứng
          - top3: 3 tham số có |r| lớn nhất, mỗi phần tử [name, r, p]
          - best_obj_value: giá trị objective tốt nhất
          - n_valid: số mẫu hợp lệ
    """
    n_valid = len(df)
    corr_list: List[float] = []
    pval_list: List[float] = []
    top_candidates: List[Tuple[str, float, float]] = []

    for p in param_names:
        r, pv = spearmanr(df[p], df['obj_value'])
        corr_list.append(r)
        pval_list.append(pv)
        top_candidates.append((p, r, pv))

    # Sắp xếp theo |r| giảm dần
    top_candidates.sort(key=lambda x: abs(x[1]), reverse=True)
    top3 = [[name, r, p] for name, r, p in top_candidates[:3]]

    # Best obj_value: mọi objective hiện tại trong pipeline đều MINIMIZE.
    # 'auxetic': c = Q12 - mu*(Q11+Q22) + penalty — Q12 càng âm càng tốt,
    # nên "best" = giá trị NHỎ NHẤT, không phải lớn nhất.
    best_obj_value = float(df['obj_value'].min())

    # Đảm bảo corr_list và pval_list là Python số thực (không phải numpy)
    corr_list = [float(v) for v in corr_list]
    pval_list = [float(v) for v in pval_list]

    return corr_list, pval_list, top3, best_obj_value, n_valid


# ──────────────────────────────────────────────
#  Bước 5: Đếm success / converged
# ──────────────────────────────────────────────

def count_success_converged(config: Dict[str, Any]) -> Tuple[int, int, int]:
    """Đếm số samples, số success, số converged.

    Args:
        config: Dict từ discover_configs

    Returns:
        Tuple (n_samples, n_success, n_converged)
    """
    if 'csv_path' in config:
        df = pd.read_csv(config['csv_path'])
        n_samples = len(df)
        n_success = int(df['success'].sum()) if 'success' in df.columns else n_samples
        n_converged = int(df['converged'].sum()) if 'converged' in df.columns else n_samples
        return n_samples, n_success, n_converged

    if 'sample_dirs' in config:
        sample_dirs = config['sample_dirs']
        n_samples = len(sample_dirs)
        n_success = 0
        n_converged = 0
        for smp in sample_dirs:
            meta_path = os.path.join(smp, 'metadata.json')
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    if meta.get('success', True):
                        n_success += 1
                        if meta.get('converged', False):
                            n_converged += 1
                except Exception:
                    n_success += 1  # fallback
                    n_converged += 1
            else:
                n_success += 1
                n_converged += 1
        return n_samples, n_success, n_converged

    return 0, 0, 0


# ──────────────────────────────────────────────
#  Bước 6: Ghi output
# ──────────────────────────────────────────────

def build_and_write_outputs(
    config_data: List[Dict[str, Any]],
    root_dir: str,
) -> None:
    """Xây dựng và ghi hai file JSON.

    Args:
        config_data: Danh sách kết quả phân tích từng config
        root_dir: Thư mục đầu ra (outputs/pipeline/phase1)
    """
    # Thu thập tất cả param_names (có thể khác nhau giữa các objective)
    # Dùng param_names phổ biến nhất, hoặc union
    all_param_names: List[str] = []
    param_name_counts: Dict[str, int] = {}
    for cd in config_data:
        for p in cd.get('param_names', []):
            param_name_counts[p] = param_name_counts.get(p, 0) + 1

    # Sắp xếp param_names: theo tần suất xuất hiện giảm dần
    all_param_names = sorted(param_name_counts.keys())
    # Hoặc giữ thứ tự ưu tiên: volfrac, penal, rmin, move, void_size_frac ...
    priority_order = ['volfrac', 'penal', 'rmin', 'move', 'void_size_frac']
    all_param_names = [p for p in priority_order if p in param_name_counts]
    all_param_names += sorted(p for p in param_name_counts if p not in priority_order)

    # ── Xây dựng _all_correlations.json ──
    correlations_output: Dict[str, Any] = {
        'param_names': all_param_names,
        'configs': [],
    }

    for cd in config_data:
        # Map correlations về đúng thứ tự all_param_names
        param_names_local = cd['param_names']
        corr_map = dict(zip(param_names_local, cd['corr']))
        pval_map = dict(zip(param_names_local, cd['pval']))

        corr_full = [corr_map.get(p, None) for p in all_param_names]
        pval_full = [pval_map.get(p, None) for p in all_param_names]

        correlations_output['configs'].append({
            'seed': cd['seed'],
            'objective': cd['objective'],
            'best_obj_value': cd['best_obj_value'],
            'n_success': cd['n_success'],
            'n_converged': cd['n_converged'],
            'corr': corr_full,
            'pval': pval_full,
            'top3': cd['top3'],
        })

    # ── Xây dựng _all_summaries_parallel.json ──
    summaries_output: List[Dict[str, Any]] = []

    for cd in config_data:
        top3_formatted = [
            {'name': t[0], 'r': t[1], 'p': t[2]}
            for t in cd['top3']
        ]
        summaries_output.append({
            'objective': cd['objective'],
            'seed': cd['seed'],
            'n_samples': cd['n_samples'],
            'n_success': cd['n_success'],
            'n_converged': cd['n_converged'],
            'n_valid_analysis': cd['n_valid'],
            'top_3_params': top3_formatted,
            'best_obj_value': cd['best_obj_value'],
            'elapsed_time': cd.get('elapsed_time'),
            'n_workers': cd.get('n_workers'),
        })

    # ── Ghi file ──
    corr_path = os.path.join(root_dir, '_all_correlations.json')
    with open(corr_path, 'w') as f:
        json.dump(correlations_output, f, indent=2, ensure_ascii=False)
    print(f'[OK] Đã ghi: {corr_path}')

    summ_path = os.path.join(root_dir, '_all_summaries_parallel.json')
    with open(summ_path, 'w') as f:
        json.dump(summaries_output, f, indent=2, ensure_ascii=False)
    print(f'[OK] Đã ghi: {summ_path}')


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main(root_dir: str = 'outputs/pipeline/phase1') -> None:
    """Hàm chính: quét, phân tích, xuất JSON.

    Args:
        root_dir: Thư mục gốc chứa dữ liệu Phase 1
    """
    # Kiểm tra thư mục đầu vào
    if not os.path.isdir(root_dir):
        print(f'[ERROR] Thư mục không tồn tại: {root_dir}')
        sys.exit(1)

    print(f'[INFO] Quét dữ liệu Phase 1 từ: {root_dir}')
    print()

    # Bước 1: Phát hiện seeds
    seeds = discover_seeds(root_dir)
    print(f'[INFO] Tìm thấy {len(seeds)} seeds: {", ".join(seeds)}')
    print()

    # Bước 2: Phát hiện configs
    all_configs: List[Dict[str, Any]] = []
    for seed in seeds:
        seed_dir = os.path.join(root_dir, seed)
        configs = discover_configs(seed_dir, seed)
        all_configs.extend(configs)
        for cfg in configs:
            src = 'CSV' if 'csv_path' in cfg else 'samples'
            print(f'  [Config] {cfg["seed"]} / {cfg["objective"]} ({src})')

    print()
    print(f'[INFO] Tổng cộng {len(all_configs)} configs cần xử lý')
    print()

    # Bước 3-4: Xử lý từng config
    results: List[Dict[str, Any]] = []
    for config in all_configs:
        seed = config['seed']
        objective = config['objective']

        try:
            # Đọc dữ liệu
            df, param_names, _, _ = load_data(config)

            if len(df) < 3:
                print(f'  [WARN] {seed} / {objective}: chỉ có {len(df)} mẫu hợp lệ, bỏ qua')
                continue

            # Tính tương quan
            corr, pval, top3, best_obj, n_valid = compute_correlations(
                df, param_names, objective,
            )

            # Đếm success/converged
            n_samples, n_success, n_converged = count_success_converged(config)

            # Đọc elapsed_time / n_workers thật (nếu có JSON song hành)
            run_meta = (
                load_run_metadata(config['csv_path'])
                if 'csv_path' in config
                else {'elapsed_time': None, 'n_workers': None}
            )

            # Ghi nhận
            entry = {
                'seed': seed,
                'objective': objective,
                'param_names': param_names,
                'corr': corr,
                'pval': pval,
                'top3': top3,
                'best_obj_value': best_obj,
                'n_valid': n_valid,
                'n_samples': n_samples,
                'n_success': n_success,
                'n_converged': n_converged,
                'elapsed_time': run_meta['elapsed_time'],
                'n_workers': run_meta['n_workers'],
            }
            results.append(entry)

            # In progress
            print(f'  [OK] {seed:25s} / {objective:8s}  '
                  f'best={best_obj:+.6f}  '
                  f'samples={n_valid}/{n_samples}  '
                  f'top1={top3[0][0]} (r={top3[0][1]:+.4f})')

        except Exception as e:
            print(f'  [FAIL] {seed} / {objective}: {e}')
            continue

    print()
    print(f'[INFO] Xử lý thành công {len(results)}/{len(all_configs)} configs')
    print()

    # Bước 5: Ghi output
    if results:
        build_and_write_outputs(results, root_dir)
    else:
        print('[WARN] Không có kết quả nào để ghi!')

    print()
    print('[DONE] Hoàn tất.')


if __name__ == '__main__':
    main()