"""
Phase 5 - aesthetics.py
============================================================
Trục "thẩm mỹ" trong composite scoring của best_of_n_eval.py - ưu tiên THẤP
(so với accuracy=cao, manufacturability=trung bình, xem docstring đầu
best_of_n_eval.py). Không có literature/metric nào sẵn có trong codebase
cho "đẹp" ở vi cấu trúc topology optimization - đây là 1 proxy hình học
đơn giản (đối xứng + độ trơn viền), cố tình giữ RẺ/không cần model riêng vì
đúng với trọng số thấp nhất trong 3 trục.

Cùng convention ảnh với manufacturability.py: [0,1] float density field
(64x64 mặc định), img_bin = img > 0.5 (1 = vật liệu rắn).
"""
import numpy as np


def symmetry_score(img: np.ndarray) -> float:
    """Trung bình đối xứng ngang (flip trái-phải) + dọc (flip trên-dưới):
    1 - mean(|img - flip|) mỗi trục, rồi trung bình 2 trục. [0,1], 1 = đối
    xứng hoàn hảo qua CẢ 2 trục, ~0.5 = ảnh nhiễu ngẫu nhiên (kỳ vọng thống
    kê của |a-b| giữa 2 pixel [0,1] độc lập)."""
    img = np.asarray(img, dtype=np.float64)
    sym_h = 1.0 - np.mean(np.abs(img - img[:, ::-1]))
    sym_v = 1.0 - np.mean(np.abs(img - img[::-1, :]))
    return float(np.clip((sym_h + sym_v) / 2.0, 0.0, 1.0))


def smoothness_score(img_bin: np.ndarray) -> float:
    """Nghịch đảo tỉ lệ chu vi/diện tích đã chuẩn hoá. Chu vi đo bằng số
    cặp pixel rắn-rỗng liền kề (so sánh ảnh với bản dịch 1 pixel theo mỗi
    trục, kể cả wrap-around - nhất quán với quy ước lát tuần hoàn của dự án,
    xem manufacturability.py check_periodicity docstring). Vi cấu trúc nét
    trơn/khối liền có chu vi thấp so với kích thước -> điểm cao; vi cấu
    trúc răng cưa/nhiễu nhiều chi tiết vụn -> chu vi cao -> điểm thấp.
    Chuẩn hoá theo chu vi tối đa lý thuyết (bàn cờ - mọi cặp liền kề đều là
    biên) để map về [0,1], không phụ thuộc kích thước ảnh."""
    solid = np.asarray(img_bin) > 0.5
    area = int(solid.sum())
    if area == 0 or area == solid.size:
        return 1.0  # toàn rỗng hoặc toàn đặc - không có biên nào, "trơn" tuyệt đối
    perimeter = 0
    for axis in (0, 1):
        shifted = np.roll(solid, 1, axis=axis)
        perimeter += int(np.sum(solid != shifted))
    max_perimeter = 2 * solid.size  # bàn cờ: mọi so sánh đều lệch, 2 hướng
    ratio = perimeter / max_perimeter
    return float(np.clip(1.0 - ratio, 0.0, 1.0))


def aesthetic_score(img: np.ndarray, img_bin: np.ndarray = None) -> float:
    """Kết hợp đối xứng + độ trơn viền, trọng số bằng nhau. img_bin: truyền
    sẵn nếu caller đã binarize (vd best_of_n_eval.py đã có `imgs[i] > 0.5`
    ở nơi khác) để tránh tính lại; None thì tự binarize từ `img`."""
    if img_bin is None:
        img_bin = (np.asarray(img) > 0.5).astype(np.float32)
    return float(0.5 * symmetry_score(img) + 0.5 * smoothness_score(img_bin))
