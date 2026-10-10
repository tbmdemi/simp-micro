"""
Ghi dữ liệu vòng lặp tối ưu hóa SIMP ra file CSV (`save_csv`, dùng bởi
`simp/runner.py` và `simp/mma_runner.py`).
"""

import os

import numpy as np


def save_csv(output_dir: str, data: dict) -> str:
    """Ghi dữ liệu vòng lặp ra file CSV.

    Args:
        output_dir: Thư mục đầu ra.
        data: Từ điển với các key:
            'iteration', 'v12', 'v21', 'objective', 'volume'
            mỗi giá trị là mảng 1-D.

    Returns:
        Đường dẫn file CSV đã ghi.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'iteration_data.csv')

    header = 'Iteration,Poisson_v12,Poisson_v21,Objective,Volume_Fraction'
    arr = np.column_stack([
        data['iteration'],
        data['v12'],
        data['v21'],
        data['objective'],
        data['volume'],
    ])
    np.savetxt(filepath, arr, delimiter=',', header=header,
               comments='', fmt='%d,%.6f,%.6f,%.6f,%.6f')
    return filepath
