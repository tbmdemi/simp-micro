"""
Gói core — các thành phần cốt lõi của thuật toán SIMP.

Bao gồm:
    - fem:       Xây dựng lưới phần tử hữu hạn và ánh xạ bậc tự do.
    - filter:    Bộ lọc mật độ hình nón để tránh checkerboard.
    - pbc:       Điều kiện biên tuần hoàn cho ô cơ sở.
    - solver:    Bộ giải FE thưa với ràng buộc PBC.
    - oc:        Cập nhật theo tiêu chí tối ưu (Optimality Criteria).
    - convergence: Phát hiện hội tụ dựa trên thay đổi thiết kế và hàm mục tiêu.
"""

from .convergence import ConvergenceChecker

__all__ = ['ConvergenceChecker']
