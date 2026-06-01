"""
Ghi dữ liệu vòng lặp tối ưu hóa SIMP ra file CSV.

Bao gồm cả một logger nhẹ và hàm save_csv tiện lợi.
"""

import os
from typing import List, Tuple

import numpy as np


class SimpLogger:
    def __init__(self, output_dir: str, buffer_size: int = 10) -> None:
        self.output_dir = output_dir
        self.buffer_size = buffer_size
        self.filename = os.path.join(output_dir, 'iteration_data.csv')

        self.iterations: List[int] = []
        self.poisson_ratios_v12: List[float] = []
        self.poisson_ratios_v21: List[float] = []
        self.objectives: List[float] = []
        self.volume_fractions: List[float] = []
        self._buffer: List[Tuple[int, float, float, float, float]] = []

        os.makedirs(output_dir, exist_ok=True)
        self._write_header()

    def _write_header(self) -> None:
        with open(self.filename, 'w', encoding='utf-8') as f:
            f.write('Iteration,Poisson_v12,Poisson_v21,Objective,Volume_Fraction\n')

    def log(self, iteration: int, v12: float, v21: float, objective: float, volume: float) -> None:
        self.iterations.append(iteration)
        self.poisson_ratios_v12.append(v12)
        self.poisson_ratios_v21.append(v21)
        self.objectives.append(objective)
        self.volume_fractions.append(volume)
        self._buffer.append((iteration, v12, v21, objective, volume))

        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def log_initial(self, iteration: int, volume: float) -> None:
        self.log(iteration, float('nan'), float('nan'), float('nan'), volume)

    def flush(self) -> None:
        if not self._buffer:
            return

        with open(self.filename, 'a', encoding='utf-8') as f:
            for row in self._buffer:
                f.write(f"{row[0]},{row[1]:.6f},{row[2]:.6f},{row[3]:.6f},{row[4]:.6f}\n")

        self._buffer.clear()

    @property
    def n_iters(self) -> int:
        return len(self.iterations)


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
