"""
Gói objectives — các hàm mục tiêu cho tối ưu hóa hình dạng SIMP.

Bao gồm:
    - first_obj:  Hàm mục tiêu loại thứ nhất (tối đa hóa |ν₁₂|).
    - second_obj: Hàm mục tiêu loại thứ hai (tối thiểu hóa C).
    - auxetic:    Hàm mục tiêu auxetic (tối đa hóa ν₁₂ âm).
"""

from .first_obj import compute_first_objective
from .second_obj import compute_second_objective
from .auxetic import compute_auxetic_objective

__all__ = ['compute_first_objective', 'compute_second_objective', 'compute_auxetic_objective']