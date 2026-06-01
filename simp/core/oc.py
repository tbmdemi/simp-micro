"""
Cập nhật theo tiêu chí tối ưu (OC) cho tối ưu hóa hình dạng SIMP.

Thực hiện thuật toán cập nhật OC (Optimality Criteria) cổ điển
để cập nhật biến thiết kế dựa trên độ nhạy và ràng buộc thể tích.
"""

import numpy as np
from scipy.sparse import csr_matrix


def oc_update(
    x: np.ndarray,
    dc: np.ndarray,
    dv: np.ndarray,
    volfrac: float,
    move: float,
    H: csr_matrix,
    Hs: np.ndarray,
    ft: int,
):
    """Cập nhật biến thiết kế dùng tiêu chí tối ưu (OC).

    Thực hiện cập nhật OC với tìm kiếm nhị phân trên hệ số Lagrange
    để thỏa mãn ràng buộc thể tích.

    Args:
        x: Mảng (nely, nelx) biến thiết kế hiện tại.
        dc: Mảng (nely, nelx) độ nhạy hàm mục tiêu.
        dv: Mảng (nely, nelx) độ nhạy thể tích.
        volfrac: Tỉ lệ thể tích yêu cầu.
        move: Giới hạn thay đổi cho phép mỗi vòng lặp.
        H: Ma trận lọc thưa.
        Hs: Vector tổng trọng số lọc.
        ft: Loại bộ lọc (1=độ nhạy, 2=mật độ).

    Returns:
        Bộ (xnew, xPhys) với:
            xnew : Mảng (nely, nelx) biến thiết kế mới (chưa lọc).
            xPhys: Mảng (nely, nelx) mật độ vật lý (đã lọc).
    """
    nely, nelx = x.shape
    l1 = 0
    l2 = 1e9

    # Tìm kiếm nhị phân cho hệ số Lagrange
    for _ in range(100):
        lmid = (l1 + l2) / 2

        # Quy tắc cập nhật OC
        xnew = np.maximum(
            0,
            np.maximum(
                x - move,
                np.minimum(
                    1,
                    np.minimum(
                        x + move,
                        x * np.sqrt(np.maximum(0, -dc / (dv * lmid + 1e-15))),
                    ),
                ),
            ),
        )

        # Áp dụng bộ lọc mật độ
        if ft == 1:
            xPhys = xnew.copy()
        elif ft == 2:
            xPhys_flat = H @ xnew.flatten('F') / Hs
            xPhys = np.reshape(xPhys_flat, (nely, nelx), order='F')

        # Kiểm tra ràng buộc thể tích
        if np.mean(xPhys) > volfrac:
            l1 = lmid
        else:
            l2 = lmid

    return xnew, xPhys
