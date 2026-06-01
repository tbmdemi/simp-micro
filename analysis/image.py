"""
Module phân tích chất lượng hình ảnh cho kết quả SIMP.

Tính toán các chỉ số như tỉ lệ nhị phân, mật độ cạnh, tỉ lệ nhiễu và tính đối xứng
để đánh giá chất lượng của kết quả tối ưu hóa hình dạng.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)


def load_image(path: str, grayscale: bool = True) -> np.ndarray:
    """
    Tải một hình ảnh dưới dạng mảng numpy.

    Args:
        path (str): Đường dẫn đến file ảnh.
        grayscale (bool): Nếu True, chuyển đổi sang ảnh xám (mặc định True).

    Returns:
        np.ndarray: Ảnh dưới dạng mảng numpy 2D (xám) hoặc 3D (RGB).

    Raises:
        FileNotFoundError: Nếu không tìm thấy file ảnh.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f'Không tìm thấy ảnh: {path}')

    img = Image.open(path)
    if grayscale:
        img = img.convert('L')
    return np.array(img)


def compute_binary_rate(img: np.ndarray, threshold: int = 128) -> float:
    """
    Tính toán tỉ lệ nhị phân của ảnh xám.

    Đo lường mức độ gần của ảnh với một ảnh nhị phân hoàn hảo (đen/trắng).
    Tỉ lệ nhị phân cao cho thấy sự phân tách rõ ràng giữa vật liệu và lỗ rỗng.

    Args:
        img (np.ndarray): Mảng ảnh xám (giá trị 0–255).
        threshold (int): Ngưỡng để nhị phân hóa (mặc định 128).

    Returns:
        float: Tỉ lệ nhị phân từ 0 đến 1, trong đó 1 là nhị phân hoàn hảo.
    """
    # Đếm các pixel rõ ràng là đen (< 32) hoặc trắng (> 224)
    n_clear = np.sum((img < 32) | (img > 224))
    return n_clear / img.size


def compute_edge_density(
    img: np.ndarray,
    threshold: int = 128,
) -> float:
    """
    Tính toán mật độ cạnh sử dụng gradient kiểu Sobel.

    Đo lường tỉ lệ các pixel cạnh trong ảnh. Mật độ cạnh cao có thể cho thấy
    hiện tượng checkerboarding hoặc biên giới bị nhiễu.

    Args:
        img (np.ndarray): Mảng ảnh xám.
        threshold (int): Ngưỡng để phát hiện cạnh (mặc định 128).

    Returns:
        float: Mật độ cạnh từ 0 đến 1.
    """
    from scipy.ndimage import sobel

    binary = (img > threshold).astype(float)
    edges = np.abs(sobel(binary))
    return float(np.mean(edges > 0))


def compute_noise_ratio(
    img: np.ndarray,
    threshold: int = 128,
    kernel_size: int = 3,
) -> float:
    """
    Tính toán tỉ lệ nhiễu sử dụng phương sai cục bộ.

    Đo lường tỉ lệ các pixel khác biệt so với trung vị cục bộ của chúng.
    Tỉ lệ nhiễu cao cho thấy các vết nhiễu xám hoặc mật độ trung gian.

    Args:
        img (np.ndarray): Mảng ảnh xám.
        threshold (int): Ngưỡng nhị phân hóa (mặc định 128).
        kernel_size (int): Kích thước vùng lân cận cục bộ (mặc định 3).

    Returns:
        float: Tỉ lệ nhiễu từ 0 đến 1.
    """
    from scipy.ndimage import median_filter

    binary = (img > threshold).astype(float)
    local_median = median_filter(binary, size=kernel_size)
    noise = np.abs(binary - local_median)
    return float(np.mean(noise))


def compute_symmetry_lr(img: np.ndarray, threshold: int = 128) -> float:
    """
    Tính toán tính đối xứng trái-phải của ảnh đã nhị phân hóa.

    Đo lường sự khớp nhau từng pixel giữa nửa trái và nửa phải của ảnh.
    Tính đối xứng cao được mong đợi cho nhiều thiết kế đơn vị ô định kỳ.

    Args:
        img (np.ndarray): Mảng ảnh xám.
        threshold (int): Ngưỡng nhị phân hóa (mặc định 128).

    Returns:
        float: Điểm đối xứng từ 0 đến 1, trong đó 1 là đối xứng hoàn hảo.
    """
    binary = (img > threshold).astype(float)
    n_cols = binary.shape[1]
    mid = n_cols // 2

    left = binary[:, :mid]
    right = np.fliplr(binary[:, -mid:])

    # Xử lý trường hợp chiều rộng lẻ
    if left.shape[1] != right.shape[1]:
        min_w = min(left.shape[1], right.shape[1])
        left = left[:, :min_w]
        right = right[:, :min_w]

    agreement = np.mean(left == right)
    return float(agreement)


def analyze_image(path: str) -> Dict[str, float]:
    """
    Chạy phân tích chất lượng hình ảnh đầy đủ cho một ảnh duy nhất.

    Args:
        path (str): Đường dẫn đến file ảnh.

    Returns:
        Dict[str, float]: Từ điển chứa các khóa: binary_rate, edge_density,
            noise_ratio, symmetry_lr.
    """
    img = load_image(path)
    return {
        'binary_rate': compute_binary_rate(img),
        'edge_density': compute_edge_density(img),
        'noise_ratio': compute_noise_ratio(img),
        'symmetry_lr': compute_symmetry_lr(img),
    }


def analyze_image_directory(
    directory: str,
    pattern: str = 'iteration_*.png',
) -> pd.DataFrame:
    """
    Phân tích tất cả các ảnh trong một thư mục khớp với mẫu cho trước.

    Args:
        directory (str): Đường dẫn đến thư mục chứa ảnh.
        pattern (str): Mẫu glob cho các file ảnh (mặc định 'iteration_*.png').

    Returns:
        pd.DataFrame: DataFrame với các cột: filename, binary_rate, edge_density,
            noise_ratio, symmetry_lr.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        logger.warning('Không tìm thấy thư mục: %s', directory)
        return pd.DataFrame()

    image_paths = sorted(dir_path.glob(pattern))
    if not image_paths:
        logger.warning('Không tìm thấy ảnh nào khớp với %s trong %s', pattern, directory)
        return pd.DataFrame()

    rows = []
    for img_path in image_paths:
        try:
            metrics = analyze_image(str(img_path))
            metrics['filename'] = img_path.name
            rows.append(metrics)
        except Exception as e:
            logger.warning('Không thể phân tích %s: %s', img_path, e)

    return pd.DataFrame(rows)