"""
Gói objectives - các hàm mục tiêu cho tối ưu hóa hình dạng SIMP.

Bao gồm:
    - auxetic: Hàm mục tiêu auxetic (tối đa hóa ν₁₂ âm).
"""

from .auxetic import compute_auxetic_q12_objective

__all__ = ['compute_auxetic_q12_objective']
