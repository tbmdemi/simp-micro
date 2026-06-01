"""
Các tiện ích lắp ráp phần tử hữu hạn cho tối ưu hóa hình dạng SIMP.

Xử lý đánh số nút, ánh xạ bậc tự do phần tử và
vector chỉ số ma trận độ cứng toàn cục.
"""

import numpy as np


def build_dof_mesh(nelx: int, nely: int):
    """Xây dựng đánh số nút và ánh xạ bậc tự do phần tử.

    Args:
        nelx: Số phần tử theo phương x.
        nely: Số phần tử theo phương y.

    Returns:
        Bộ (nodenrs, edofVec, edofMat, iK, jK) với:
            nodenrs : Mảng (nely+1) × (nelx+1) các ID nút.
            edofVec : Vector (nelx*nely,) bậc tự do đầu tiên của mỗi phần tử.
            edofMat : Ma trận (nelx*nely, 8) ánh xạ phần tử → bậc tự do toàn cục.
            iK      : Vector (64*nelx*nely,) chỉ số hàng cho lắp ráp K thưa.
            jK      : Vector (64*nelx*nely,) chỉ số cột cho lắp ráp K thưa.
    """
    # Đánh số nút: theo cột (tương thích MATLAB)
    nodenrs = np.reshape(
        np.arange(1, (1 + nelx) * (1 + nely) + 1),
        (1 + nely, 1 + nelx),
        order='F'
    )

    # Vector bậc tự do phần tử: bậc tự do đầu tiên của mỗi phần tử
    edofVec = np.reshape(
        2 * nodenrs[:-1, :-1] + 1,
        nelx * nely,
        order='F'
    )

    # Ma trận bậc tự do phần tử: tất cả 8 bậc tự do cho mỗi phần tử
    # Ghép thủ công để tránh lỗi mảng không đồng nhất
    offset = np.concatenate([
        [0], [1],
        2 * nely + np.array([2, 3, 0, 1]),
        [-2], [-1],
    ])
    edofMat = np.tile(edofVec, (8, 1)).T + np.tile(
        offset,
        (nelx * nely, 1)
    )

    # Chuyển về chỉ số 0-based cho ma trận thưa Python
    edofMat_0 = edofMat - 1

    # Vector chỉ số cho ma trận độ cứng toàn cục thưa
    iK = np.reshape(np.kron(edofMat_0, np.ones((8, 1), dtype=int)).T,
                    64 * nelx * nely, order='F')
    jK = np.reshape(np.kron(edofMat_0, np.ones((1, 8), dtype=int)).T,
                    64 * nelx * nely, order='F')

    return nodenrs, edofVec, edofMat, iK, jK
