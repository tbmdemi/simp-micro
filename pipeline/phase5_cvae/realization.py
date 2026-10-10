"""
Phase 5 - realization.py
=============================================
Hiện thực hóa dịch lệch lưới khả vi cho refine (plan.md mục N1).

Tensor đồng nhất hóa của ô tuần hoàn bất biến khi dịch ô. Nếu dịch thiết kế
lệch lưới FE một phần pixel rồi rời rạc hóa lại mà ν đổi nhiều, thiết kế đang
khai thác lỗi rời rạc hóa của lưới verify (chẩn đoán N1 bước 1: refine giảm
MAE ν12 68% trên lưới 50² nhưng chỉ 26% trên lưới mịn). Objective N1 lấy
trung bình sai số qua nhiều hiện thực hóa để không thưởng cho thiết kế chỉ
đúng ở đúng 1 cách đặt lưới.

R(s): ảnh mật độ → làm mượt Gauss tuần hoàn (σ cố định theo đơn vị vật lý)
→ lấy mẫu song tuyến tuần hoàn tại tâm phần tử lưới FE dịch s → Heaviside β.
Đổi ngưỡng chiếu cuối (eta ≠ 0,5) cho bản co/giãn của robust formulation
(plan.md mục N2-E1).
Bản numpy tương ứng (để đánh giá offline) nằm trong script chẩn đoán N1.
"""

import math
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    from .heaviside import heaviside_projection_torch
except ImportError:
    from heaviside import heaviside_projection_torch


def _gauss_blur_periodic(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Làm mượt Gauss tách biến với biên tuần hoàn.

    Args:
        x: (B, 1, H, W).
        sigma: độ lệch chuẩn theo pixel của chính x. ≤ 0 = không làm mượt.

    Returns:
        Tensor cùng shape.
    """
    if sigma <= 0:
        return x
    r = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="circular"), k.view(1, 1, 1, -1))
    return F.conv2d(
        F.pad(x, (0, 0, r, r), mode="circular"), k.view(1, 1, -1, 1)
    )


def _sample_periodic_1d(x: torch.Tensor, n: int, shift: float, dim: int):
    """Nội suy tuyến tính tuần hoàn x theo 1 trục tại tâm n phần tử dịch.

    Args:
        x: tensor nguồn.
        n: số phần tử lưới đích theo trục này.
        shift: độ dịch theo đơn vị phần tử lưới đích.
        dim: trục nội suy.

    Returns:
        Tensor với kích thước trục `dim` = n.
    """
    N = x.shape[dim]
    pos = torch.arange(n, dtype=x.dtype, device=x.device) + 0.5 + shift
    pos = pos * (N / n) - 0.5
    i0 = torch.floor(pos)
    w = pos - i0
    i0 = i0.long() % N
    i1 = (i0 + 1) % N
    shape = [1] * x.dim()
    shape[dim] = n
    w = w.view(shape)
    return x.index_select(dim, i0) * (1 - w) + x.index_select(dim, i1) * w


def realize_shifted(
    image: torch.Tensor,
    beta: float,
    nely: int,
    nelx: int,
    shifts: Sequence[Tuple[float, float]],
    sigma: float = 0.5,
    eta: float = 0.5,
) -> torch.Tensor:
    """Sinh K hiện thực hóa lệch lưới của 1 ảnh, sẵn sàng đưa vào FE.

    Args:
        image: (1, 1, H, W) hoặc (1, H, W), mật độ decoder trong [0,1].
        beta: độ dốc Heaviside (áp trước làm mượt và sau lấy mẫu, để ở β
            lớn mỗi hiện thực hóa gần nhị phân như verify).
        nely: số hàng lưới FE.
        nelx: số cột lưới FE.
        shifts: danh sách (dy, dx) theo đơn vị phần tử lưới FE.
        sigma: độ mượt theo đơn vị phần tử lưới FE (đổi sang pixel ảnh bên
            trong) - cố định vật lý, khớp chẩn đoán N1 (0,5).
        eta: ngưỡng chiếu cuối trên trường đã làm mượt. 0,5 = gốc; < 0,5 =
            giãn (dilated), > 0,5 = co (eroded) - robust formulation (Wang,
            Lazarov & Sigmund 2011). Với biên thẳng, biên dịch một đoạn
            σ·Φ⁻¹(eta) (Φ = CDF chuẩn), vd σ=1, eta=0,75 → co 0,67 phần tử.

    Returns:
        (K, 1, nely, nelx).
    """
    img = image if image.dim() == 4 else image.unsqueeze(1)
    x = heaviside_projection_torch(img, beta)
    phi = _gauss_blur_periodic(x, sigma * img.shape[-1] / nelx)
    out = []
    for dy, dx in shifts:
        v = _sample_periodic_1d(phi, nely, dy, dim=-2)
        v = _sample_periodic_1d(v, nelx, dx, dim=-1)
        out.append(heaviside_projection_torch(v, beta, eta))
    return torch.cat(out, dim=0)


def filtered_design(
    image: torch.Tensor,
    beta,
    nely: int,
    nelx: int,
    sigma: float,
    eta: float = 0.5,
) -> torch.Tensor:
    """Thiết kế vật lý qua bộ lọc (N2-E1′): lọc → lấy mẫu lưới FE → chiếu.

    Khác ``realize_shifted``: KHÔNG nhị phân hóa trước khi lọc - bộ lọc là
    một phần của tham số hóa thiết kế như density filter trong tối ưu topo
    (Wang, Lazarov & Sigmund 2011), nên bản gốc/co/giãn của robust
    formulation cùng sinh từ một trường đã lọc, và chi tiết nhỏ hơn ~σ tự
    biến mất thay vì thành đốm vật liệu "miễn phí" (lỗi của E1).

    Args:
        image: (B, 1, H, W) hoặc (B, H, W), mật độ decoder trong [0,1].
        beta: độ dốc Heaviside; None = ngưỡng cứng (dùng khi verify, β = ∞).
        nely: số hàng lưới FE.
        nelx: số cột lưới FE.
        sigma: độ mượt của bộ lọc, đơn vị phần tử lưới FE.
        eta: ngưỡng chiếu (0,5 gốc; < 0,5 giãn; > 0,5 co).

    Returns:
        (B, 1, nely, nelx) - nhị phân nếu ``beta`` là None.
    """
    img = image if image.dim() == 4 else image.unsqueeze(1)
    phi = _gauss_blur_periodic(img, sigma * img.shape[-1] / nelx)
    v = _sample_periodic_1d(phi, nely, 0.0, dim=-2)
    v = _sample_periodic_1d(v, nelx, 0.0, dim=-1)
    if beta is None:
        return (v > eta).to(img.dtype)
    return heaviside_projection_torch(v, beta, eta)
