"""
Trực quan hóa trường mật độ cho tối ưu hóa hình dạng SIMP.

Cung cấp các hàm lưu ảnh trường mật độ dùng matplotlib,
hỗ trợ phóng đại và tùy chọn màu sắc.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend không tương tác cho lưu file
import matplotlib.pyplot as plt


def save_density_image(
    xPhys: np.ndarray,
    output_dir: str,
    iteration: int,
    scale_factor: int = 1,
) -> str:
    """Lưu ảnh trường mật độ vật lý.

    Tạo ảnh thang xám của trường mật độ và lưu ra file PNG.

    Args:
        xPhys: Mảng (nely, nelx) mật độ vật lý.
        output_dir: Thư mục đầu ra.
        iteration: Số vòng lặp (dùng trong tên file).
        scale_factor: Hệ số phóng đại ảnh (mặc định 1).

    Returns:
        Đường dẫn đầy đủ đến file ảnh đã lưu.
    """
    # Phóng đại ảnh nếu cần
    if scale_factor > 1:
        from scipy.ndimage import zoom
        img = zoom(xPhys, scale_factor, order=0)
    else:
        img = xPhys

    # Tạo ảnh
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.axis('off')

    # Lưu
    filename = f'iteration_{iteration:05d}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return filepath
