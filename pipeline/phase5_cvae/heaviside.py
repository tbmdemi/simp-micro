"""
Phase 5 - heaviside.py
=============================================
Bản torch khả vi của các phép hậu xử lý mà pipeline verify áp lên ảnh cVAE
(best_of_n_eval.py / benchmark_physics_guided_refinement.py:
force_periodic -> nhị phân hóa ngưỡng 0,5 -> resize nearest về lưới FE).

Lý do tồn tại (plan.md v3 P1.1e): tối ưu z bằng gradient FE trên ảnh mật độ
LIÊN TỤC khai thác được vật liệu xám - loss liên tục về ~1e-10 nhưng sau khi
nhị phân hóa sai số FE lại lớn. Đưa đúng các phép này (bản khả vi) vào
objective để thứ được tối ưu trùng với thứ được kiểm chứng.

- heaviside_projection_torch: port simp/core/filter.py::apply_heaviside_projection
  (Wang, Lazarov & Sigmund 2011). β → ∞ tiến về bước nhảy tại η.
- force_periodic_torch: port manufacturability.force_periodic (trung bình rồi
  gán lại cạnh đối diện) - phép tuyến tính nên gradient chảy tự nhiên.
- resize_nearest_torch: khớp resize_to_fe_grid (PIL NEAREST) TUYỆT ĐỐI -
  lấy bảng chỉ số trực tiếp từ PIL rồi index_select (khả vi). Không dùng
  F.interpolate: mode="nearest" lệch ~40% pixel, "nearest-exact" vẫn lệch
  vài chỉ số do PIL làm tròn khác (vd 64->50: dst 12 lấy src 15, torch 16).
"""

from functools import lru_cache

import numpy as np
import torch
from PIL import Image


def heaviside_projection_torch(
    x: torch.Tensor, beta: float, eta: float = 0.5
) -> torch.Tensor:
    """Chiếu Heaviside làm mượt, khả vi theo x.

    Args:
        x: mật độ trong [0,1], shape bất kỳ.
        beta: độ dốc (>0). β nhỏ ~ gần tuyến tính, β lớn ~ bước nhảy 0/1.
        eta: ngưỡng chiếu (0,5 = khớp ngưỡng nhị phân hóa của verify).

    Returns:
        Tensor cùng shape, trong [0,1].
    """
    b = torch.as_tensor(beta, dtype=x.dtype, device=x.device)
    num = torch.tanh(b * eta) + torch.tanh(b * (x - eta))
    den = torch.tanh(b * eta) + torch.tanh(b * (1 - eta))
    return num / den


def force_periodic_torch(img: torch.Tensor) -> torch.Tensor:
    """Ép biên tuần hoàn (khả vi), cùng thứ tự với
    manufacturability.force_periodic: cột trước, hàng sau.

    Args:
        img: (..., H, W).

    Returns:
        Tensor mới cùng shape (không sửa in-place tensor đầu vào).
    """
    img = img.clone()
    col = 0.5 * (img[..., :, 0] + img[..., :, -1])
    img[..., :, 0] = col
    img[..., :, -1] = col
    row = 0.5 * (img[..., 0, :] + img[..., -1, :])
    img[..., 0, :] = row
    img[..., -1, :] = row
    return img


@lru_cache(maxsize=None)
def _pil_nearest_index(src: int, dst: int) -> tuple:
    """Chỉ số nguồn mà PIL NEAREST chọn cho từng pixel đích (1 chiều).

    Resize 1 dải chỉ số 0..src-1 bằng chính PIL; dùng mode "I" (int32) để
    không giới hạn src ≤ 256 như ảnh 8-bit.

    Args:
        src: kích thước nguồn.
        dst: kích thước đích.

    Returns:
        tuple độ dài dst các chỉ số nguồn.
    """
    ramp = np.arange(src, dtype=np.int32)[None, :]
    out = Image.fromarray(ramp, mode="I").resize((dst, 1), Image.NEAREST)
    return tuple(int(v) for v in np.asarray(out)[0])


def resize_nearest_torch(img: torch.Tensor, nely: int, nelx: int):
    """Resize nearest khớp tuyệt đối PIL (verify_fe.resize_to_fe_grid).

    Args:
        img: (..., H, W).
        nely: số hàng lưới FE.
        nelx: số cột lưới FE.

    Returns:
        (..., nely, nelx) - gradient chảy về đúng pixel nguồn được chọn.
    """
    rows = torch.tensor(
        _pil_nearest_index(img.shape[-2], nely), device=img.device
    )
    cols = torch.tensor(
        _pil_nearest_index(img.shape[-1], nelx), device=img.device
    )
    return img.index_select(-2, rows).index_select(-1, cols)


def project_for_fe(
    image: torch.Tensor,
    beta: float,
    nely: int,
    nelx: int,
    periodic: bool = False,
) -> torch.Tensor:
    """Chuỗi biến đổi khả vi khớp pipeline verify: [force_periodic] ->
    Heaviside(β, η=0,5) -> resize nearest về lưới FE. Dùng chung cho refine
    (tandem_lbfgs) và loss train (losses.real_physics_prior_loss) để 2 nơi
    không lệch nhau.

    Args:
        image: (B, 1, H, W) hoặc (B, H, W), mật độ trong [0,1].
        beta: độ dốc Heaviside.
        nely: số hàng lưới FE.
        nelx: số cột lưới FE.
        periodic: áp force_periodic trước projection.

    Returns:
        (B, 1, nely, nelx).
    """
    img = image if image.dim() == 4 else image.unsqueeze(1)
    if periodic:
        img = force_periodic_torch(img)
    img = heaviside_projection_torch(img, beta)
    return resize_nearest_torch(img, nely, nelx)
