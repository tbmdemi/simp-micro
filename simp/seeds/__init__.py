"""
Gói seeds — các bộ sinh mẫu lỗ rỗng ban đầu cho tối ưu hóa SIMP.

Mỗi module định nghĩa một hàm seed nhận (nelx, nely, volfrac)
và trả về mảng mật độ (nely, nelx) với các lỗ rỗng được bố trí
theo một mẫu hình học cụ thể.

Các mẫu có sẵn:
    - circle:              Lỗ tròn đơn.
    - square:              Lỗ vuông đơn.
    - hourglass:           Hình đồng hồ cát.
    - four_circle:         Bốn lỗ tròn.
    - hexagonal:           Lỗ lục giác.
    - nine_circle:         Chín lỗ tròn (lưới 3×3).
    - cross_rectangular:   Lỗ hình chữ thập.
    - grid_circular_voids: Lưới lỗ tròn.
    - small_square_cross:  Chữ thập vuông nhỏ.
    - circle_half_quarter: Lỗ tròn kết hợp 1/4 hình tròn.
"""

from .circle import circle_seed
from .square import square_seed
from .hourglass import hourglass_seed
from .four_circle import four_circle_seed
from .hexagonal import hexagonal_seed
from .nine_circle import nine_circle_seed
from .cross_rectangular import cross_rectangular_seed
from .grid_circular_voids import grid_circular_voids_seed
from .small_square_cross import small_square_cross_seed
from .circle_half_quarter import circle_half_quarter_seed

__all__ = [
    'circle_seed',
    'square_seed',
    'hourglass_seed',
    'four_circle_seed',
    'hexagonal_seed',
    'nine_circle_seed',
    'cross_rectangular_seed',
    'grid_circular_voids_seed',
    'small_square_cross_seed',
    'circle_half_quarter_seed',
]
