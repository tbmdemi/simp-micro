"""
Gói homogenization — tính toán đồng nhất hóa cho vật liệu tuần hoàn.

Bao gồm:
    - compute: Tính ten-xơ độ cứng đồng nhất hóa và đạo hàm của nó.
"""

from .compute import compute_homogenized_tensor

__all__ = ['compute_homogenized_tensor']
