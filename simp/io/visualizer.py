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


def _bending_curvature_ratio(nu_star: float) -> float:
    """Tỉ số độ cong κ_yy/κ_xx cho tấm mỏng chịu uốn thuần túy dọc trục X.

    Quan hệ chuẩn của lý thuyết tấm mỏng Kirchhoff-Love: κ_yy = -nu* * κ_xx.
    nu* < 0 (auxetic) cho tỉ số dương -> 2 độ cong cùng dấu (bề mặt vòm cầu
    đồng tâm, "synclastic"); nu* > 0 (vật liệu thường) cho tỉ số âm -> 2 độ
    cong ngược dấu (bề mặt yên ngựa, "anticlastic").

    Args:
        nu_star: Hệ số Poisson vĩ mô (vd từ compute_nu12()/compute_nu21()
            trong simp/objectives/auxetic.py).

    Returns:
        Tỉ số κ_yy/κ_xx (float).
    """
    return -nu_star


def reconstruct_3d_bent_surface(
    nu_star: float,
    length: float = 10.0,
    width: float = 10.0,
    grid_resolution: int = 40,
    kappa_xx: float = 0.05,
):
    """Vẽ minh họa 3D bề mặt uốn của tấm dựa trên hệ số Poisson vĩ mô.

    QUAN TRỌNG - đây là hình minh họa HÌNH HỌC (xấp xỉ giải tích bậc 2 chuẩn
    của lý thuyết tấm mỏng Kirchhoff-Love, κ_yy = -nu* * κ_xx - xem
    _bending_curvature_ratio()), KHÔNG PHẢI kết quả mô phỏng FE uốn tấm thật:
    dự án hiện chỉ có homogenization TRONG mặt phẳng (simp/homogenization/
    compute.py), chưa có solver uốn ngoài mặt phẳng. kappa_xx là độ cong giả
    định TÙY CHỌN chỉ để hình có độ dốc hợp lý khi vẽ, không đo từ mô phỏng
    nào - KHÔNG dùng trục Z của hình này làm số liệu định lượng. Khi đưa vào
    bài báo, phải chú thích rõ đây là minh họa định tính (synclastic vs
    anticlastic), không phải kết quả FE.

    Args:
        nu_star: Hệ số Poisson vĩ mô (vd từ compute_nu12()/compute_nu21()).
        length: Kích thước minh họa theo trục X.
        width: Kích thước minh họa theo trục Y.
        grid_resolution: Số điểm lưới mỗi trục.
        kappa_xx: Độ cong chính giả định theo trục X (chỉ ảnh hưởng độ dốc
            hình minh họa, không có ý nghĩa vật lý tuyệt đối).

    Returns:
        matplotlib.figure.Figure chứa hình minh họa 3D.
    """
    x = np.linspace(-length / 2, length / 2, grid_resolution)
    y = np.linspace(-width / 2, width / 2, grid_resolution)
    X, Y = np.meshgrid(x, y)

    kappa_yy = _bending_curvature_ratio(nu_star) * kappa_xx
    Z = -0.5 * (kappa_xx * X ** 2 + kappa_yy * Y ** 2)

    fig = plt.figure(figsize=(10, 7), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(
        X, Y, Z, cmap='coolwarm', edgecolor='none', alpha=0.95
    )

    curvature_type = (
        'Synclastic (auxetic)' if nu_star < 0 else 'Anticlastic (thường)'
    )
    ax.set_title(
        f'Minh họa hình học uốn tấm (nu* = {nu_star:.3f}) - {curvature_type}',
        fontsize=12, fontweight='bold',
    )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (độ võng minh họa)')
    fig.colorbar(
        surf, ax=ax, shrink=0.5, aspect=10,
        label='Z (minh họa, không phải đơn vị đo thật)',
    )
    fig.tight_layout()

    return fig
