"""
Module cung cấp các công cụ phân tích tập dữ liệu và kiểm tra hội tụ cho kết quả SIMP.

Module này phân tích dữ liệu vòng lặp từ các file CSV của nhiều lần chạy SIMP để
tạo ra các bảng thống kê, phân loại vật liệu auxetic và tính toán các chỉ số hội tụ.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Map tên cột thực tế → tên chuẩn
_COLUMN_ALIASES = {
    'Volume_Fraction': 'MeanDensity',
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa tên cột về định dạng chuẩn."""
    rename_map = {}
    for col in df.columns:
        if col in _COLUMN_ALIASES:
            rename_map[col] = _COLUMN_ALIASES[col]
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def load_iteration_data(csv_path: str) -> pd.DataFrame:
    """
    Tải dữ liệu vòng lặp từ một file CSV của SIMP.

    Hỗ trợ tên cột linh hoạt: nếu file có 'Volume_Fraction' thay vì 'MeanDensity',
    hàm tự động chuẩn hóa.

    Args:
        csv_path (str): Đường dẫn đến file CSV.

    Returns:
        pd.DataFrame: DataFrame chứa các cột: Iteration, Poisson_v12, Poisson_v21,
            Objective, MeanDensity.

    Raises:
        FileNotFoundError: Nếu không tìm thấy file CSV.
        ValueError: Nếu file CSV bị sai định dạng hoặc thiếu cột.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f'Không tìm thấy file CSV: {csv_path}')

    df = pd.read_csv(csv_path)
    df = _normalize_columns(df)

    required_cols = [
        'Iteration', 'Poisson_v12', 'Poisson_v21',
        'Objective', 'MeanDensity',
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f'File CSV thiếu các cột sau: {missing}')

    # Chuyển đổi các cột số, ép lỗi thành NaN
    numeric_cols = ['Poisson_v12', 'Poisson_v21', 'Objective', 'MeanDensity']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def compute_convergence_metrics(
    df: pd.DataFrame,
    window: int = 20,
) -> Dict[str, float]:
    """
    Tính toán các chỉ số hội tụ từ dữ liệu vòng lặp.

    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu vòng lặp.
        window (int): Kích thước cửa sổ trượt để kiểm tra độ ổn định của hàm mục tiêu.

    Returns:
        Dict[str, float]: Từ điển chứa các chỉ số:
            - n_iters: Tổng số vòng lặp.
            - final_objective: Giá trị hàm mục tiêu cuối cùng.
            - final_v12: Hệ số Poisson ν₁₂ cuối cùng.
            - final_v21: Hệ số Poisson ν₂₁ cuối cùng.
            - final_volume: Tỉ lệ thể tích cuối cùng.
            - obj_change_last_10: Thay đổi tương đối của hàm mục tiêu trong 10 vòng cuối.
            - obj_stable: True nếu hàm mục tiêu ổn định trong `window` vòng cuối.
    """
    df_clean = df.dropna(subset=['Objective'])

    if len(df_clean) < 2:
        return {
            'n_iters': len(df),
            'final_objective': float('nan'),
            'final_v12': float('nan'),
            'final_v21': float('nan'),
            'final_volume': float('nan'),
            'obj_change_last_10': float('nan'),
            'obj_stable': False,
        }

    last = df_clean.iloc[-1]
    first = df_clean.iloc[0]

    # Tính thay đổi hàm mục tiêu trong 10 vòng lặp cuối
    if len(df_clean) >= 10:
        obj_10_ago = df_clean.iloc[-10]['Objective']
        obj_change = abs(last['Objective'] - obj_10_ago) / max(
            abs(obj_10_ago), 1e-15,
        )
    else:
        obj_change = abs(last['Objective'] - first['Objective']) / max(
            abs(first['Objective']), 1e-15,
        )

    # Kiểm tra độ ổn định của hàm mục tiêu
    if len(df_clean) >= window:
        recent = df_clean.iloc[-window:]
        obj_changes = recent['Objective'].pct_change().abs().dropna()
        # Coi là ổn định nếu tất cả thay đổi trong cửa sổ đều < 5%
        obj_stable = bool((obj_changes < 0.05).all())
    else:
        obj_stable = False

    return {
        'n_iters': len(df),
        'final_objective': float(last['Objective']),
        'final_v12': float(last['Poisson_v12']),
        'final_v21': float(last['Poisson_v21']),
        'final_volume': float(last['MeanDensity']),
        'obj_change_last_10': float(obj_change),
        'obj_stable': obj_stable,
    }


def classify_auxetic(
    v12: float,
    v21: float,
    threshold: float = 0.0,
) -> str:
    """
    Phân loại vật liệu là auxetic hay thông thường dựa trên hệ số Poisson.

    Một vật liệu được coi là auxetic nếu bất kỳ hệ số Poisson nào (ν₁₂ hoặc ν₂₁)
    nhỏ hơn ngưỡng cho trước. Tuy nhiên, vì gradient của ν₁₂ không phù hợp với tối ưu hóa,
    phương pháp thay thế thường sẽ dựa trên Q₁₂ để định lượng tính auxetic.

    Args:
        v12 (float): Hệ số Poisson ν₁₂.
        v21 (float): Hệ số Poisson ν₂₁.
        threshold (float): Ngưỡng phân loại auxetic (mặc định 0.0).

    Returns:
        str: 'Auxetic' nếu thỏa mãn điều kiện, ngược lại là 'Conventional'.
    """
    if v12 < threshold or v21 < threshold:
        return 'Auxetic'
    return 'Conventional'


def build_classification_table(
    data_dir: str,
    objective_type: str = 'auxetic',
) -> pd.DataFrame:
    """
    Xây dựng bảng phân loại từ tất cả các thư mục kết quả SIMP.

    Quét các thư mục con trong `data_dir` để tìm file CSV và phân loại từng kết quả.

    Args:
        data_dir (str): Thư mục gốc chứa các thư mục kết quả SIMP.
        objective_type (str): 'auxetic' - dùng để lọc thư mục (chỉ hỗ trợ auxetic).

    Returns:
        pd.DataFrame: DataFrame với các cột: Shape, Poisson_v12, Poisson_v21,
            Classification, Objective, Volume, Iterations.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning('Không tìm thấy thư mục dữ liệu: %s', data_dir)
        return pd.DataFrame()

    rows = []
    # Tìm tất cả các file iteration_data.csv trong các thư mục con
    csv_files = list(data_path.rglob('iteration_data.csv'))

    if not csv_files:
        logger.warning('Không tìm thấy file iteration_data.csv nào trong %s', data_dir)
        return pd.DataFrame()

    for csv_path in csv_files:
        try:
            df = load_iteration_data(str(csv_path))
            metrics = compute_convergence_metrics(df)

            # Lấy tên hình dạng từ tên thư mục cha
            shape_name = csv_path.parent.name

            classification = classify_auxetic(
                metrics['final_v12'],
                metrics['final_v21'],
            )

            rows.append({
                'Shape': shape_name,
                'Poisson_v12': metrics['final_v12'],
                'Poisson_v21': metrics['final_v21'],
                'Classification': classification,
                'Objective': metrics['final_objective'],
                'Volume': metrics['final_volume'],
                'Iterations': metrics['n_iters'],
            })
        except (FileNotFoundError, ValueError) as e:
            logger.warning('Bỏ qua %s: %s', csv_path, e)

    result = pd.DataFrame(rows)
    if not result.empty:
        # Sắp xếp theo tên hình dạng để bảng nhất quán
        result = result.sort_values('Shape').reset_index(drop=True)

    return result