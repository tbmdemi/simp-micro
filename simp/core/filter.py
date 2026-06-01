"""
Bộ lọc mật độ hình nón cho tối ưu hóa hình dạng SIMP.

Thực hiện bộ lọc mật độ dạng hình nón (cone-shaped density filter)
để ngăn chặn checkerboard và đảm bảo tính khả thi sản xuất
của thiết kế tối ưu.
"""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


def build_filter(nelx: int, nely: int, rmin: float):
    """Xây dựng ma trận lọc mật độ hình nón.

    Tạo ma trận thưa H và vector tổng Hs cho bộ lọc mật độ.
    Mỗi phần tử được lọc bằng trung bình có trọng số của các phần tử
    lân cận trong bán kính rmin, với trọng số giảm tuyến tính theo khoảng cách.

    Args:
        nelx: Số phần tử theo phương x.
        nely: Số phần tử theo phương y.
        rmin: Bán kính lọc (tính bằng phần tử).

    Returns:
        Bộ (H, Hs) với:
            H  : Ma trận thưa (nelx*nely, nelx*nely) các trọng số lọc.
            Hs : Vector (nelx*nely,) tổng trọng số cho mỗi phần tử.
    """
    nfilter = int(nelx * nely * (2 * np.ceil(rmin) + 1) ** 2)
    iH = np.zeros(nfilter)
    jH = np.zeros(nfilter)
    sH = np.zeros(nfilter)
    cc = 0

    for i in range(nelx):
        for j in range(nely):
            row = j * nelx + i
            kk1 = int(np.ceil(max(i - rmin, 0)))
            kk2 = int(np.ceil(min(i + rmin, nelx - 1)))
            ll1 = int(np.ceil(max(j - rmin, 0)))
            ll2 = int(np.ceil(min(j + rmin, nely - 1)))

            for k in range(kk1, kk2 + 1):
                for l in range(ll1, ll2 + 1):
                    col = l * nelx + k
                    fac = rmin - np.sqrt((i - k) ** 2 + (j - l) ** 2)
                    if fac > 0:
                        iH[cc] = row
                        jH[cc] = col
                        sH[cc] = max(0, fac)
                        cc += 1

    # Cắt bớt mảng về kích thước thực tế
    iH = iH[:cc]
    jH = jH[:cc]
    sH = sH[:cc]

    # Xây dựng ma trận thưa
    H = coo_matrix((sH, (iH, jH)), shape=(nelx * nely, nelx * nely)).tocsr()
    Hs = np.array(H.sum(axis=1)).flatten()

    return H, Hs


def apply_filter(field: np.ndarray, H: csr_matrix, Hs: np.ndarray) -> np.ndarray:
    """Áp dụng bộ lọc trung bình có trọng số lên một trường dữ liệu.

    Args:
        field: Mảng (nely, nelx) cần lọc.
        H: Ma trận lọc thưa.
        Hs: Vector tổng trọng số.

    Returns:
        Mảng (nely, nelx) đã được lọc.
    """
    nely, nelx = field.shape
    field_flat = field.flatten('F')
    filtered_flat = H @ field_flat / Hs
    return np.reshape(filtered_flat, (nely, nelx), order='F')


def apply_sensitivity_filter(dc: np.ndarray, x: np.ndarray, H: csr_matrix, Hs: np.ndarray, ft: int) -> np.ndarray:
    """Lọc độ nhạy theo loại bộ lọc ft.

    Args:
        dc: Độ nhạy hàm mục tiêu.
        x: Biến thiết kế.
        H: Ma trận lọc.
        Hs: Vector tổng trọng số.
        ft: Loại bộ lọc (1=độ nhạy, 2=mật độ).

    Returns:
        Độ nhạy đã được lọc.
    """
    if ft == 1:
        # Lọc độ nhạy: dc_filt = H(x * dc) / (Hs * x)
        weighted_dc = x * dc
        filtered_dc = apply_filter(weighted_dc, H, Hs)
        return filtered_dc / np.maximum(1e-3, x)
    elif ft == 2:
        # Lọc độ nhạy: dc_filt = H(dc) / Hs
        return apply_filter(dc, H, Hs)
    else:
        return dc
