"""
Phase 5 - manuf_penalty.py
=============================================
Phạt khả vi cho khả năng chế tạo, dùng trong refine nhận thức nhị phân hóa
(plan.md mục K1). Cùng ý tưởng với heaviside.py: refine phải tối ưu đúng
thứ verify đo. Verify chế tạo (manufacturability.check_connectivity) đo trên
ảnh nhị phân 64² - ở đây tính trên ảnh sau Heaviside β (gần nhị phân khi β
lớn) nên gradient chảy về z.

- corner_contact_penalty: cửa sổ 2×2 dạng chéo [[1,0],[0,1]] / [[0,1],[1,0]]
  = 2 khối chỉ chạm nhau tại 1 điểm. Với phần tử Q4 đó là bản lề 1 nút
  (artifact FE kinh điển, không chế tạo được); liên thông 8 hướng tính là
  "nối", 4 hướng thì không. Tính TUẦN HOÀN vì ô được lát: chạm góc qua
  đường nối giữa các ô cũng là bản lề thật.
- thin_feature_penalty: phần vật liệu bị mất sau phép mở hình thái
  (erosion rồi dilation) với phần tử chữ thập 3×3 - khớp
  ndimage.binary_erosion mặc định trong check_connectivity (biên ngoài coi
  là rỗng, border_value=0), để phạt đúng chỗ verify đánh trượt nét mảnh.
"""

import torch
import torch.nn.functional as F


def corner_contact_penalty(x: torch.Tensor) -> torch.Tensor:
    """Trung bình mức "chạm góc" trên mọi cửa sổ 2×2 tuần hoàn.

    Với ảnh nhị phân, giá trị = tỉ lệ cửa sổ 2×2 có dạng chéo (đúng 2 pixel
    rắn nằm chéo nhau). Với ảnh mềm, là đa thức khả vi theo x.

    Args:
        x: mật độ trong [0, 1], shape (..., H, W).

    Returns:
        Tensor vô hướng (trung bình trên mọi chiều batch và cửa sổ).
    """
    a = x
    b = torch.roll(x, shifts=-1, dims=-1)  # phải
    c = torch.roll(x, shifts=-1, dims=-2)  # dưới
    e = torch.roll(b, shifts=-1, dims=-2)  # chéo dưới-phải
    diag = a * e * (1 - b) * (1 - c) + b * c * (1 - a) * (1 - e)
    return diag.mean()


def _cross_pool(x: torch.Tensor, mode: str) -> torch.Tensor:
    """Min (erosion) hoặc max (dilation) trên lân cận chữ thập 3×3.

    Biên ngoài đệm 0 cho cả hai phép - khớp binary_erosion/binary_dilation
    của scipy với border_value=0 mặc định.

    Args:
        x: (..., H, W).
        mode: "min" hoặc "max".

    Returns:
        Tensor cùng shape.
    """
    p = F.pad(x, (1, 1, 1, 1), value=0.0)
    stack = torch.stack(
        [
            p[..., 1:-1, 1:-1],
            p[..., :-2, 1:-1],
            p[..., 2:, 1:-1],
            p[..., 1:-1, :-2],
            p[..., 1:-1, 2:],
        ]
    )
    return stack.amin(0) if mode == "min" else stack.amax(0)


def thin_feature_penalty(x: torch.Tensor) -> torch.Tensor:
    """Trung bình phần vật liệu mất đi sau phép mở hình thái chữ thập.

    Pixel thuộc nét rộng < 3 px (không chứa trọn chữ thập) bị opening xóa,
    nên x − opening(x) > 0 đúng ở các nét mảnh mà check_connectivity
    (erosion 1 lần) đánh trượt. Gradient qua min/max là subgradient.
    Tác dụng phụ: opening chữ thập cũng gọt pixel góc lồi của khối dày (4
    pixel/khối vuông) - phạt nhẹ, đẩy về bo góc, không hại chế tạo.

    Args:
        x: mật độ trong [0, 1], shape (..., H, W).

    Returns:
        Tensor vô hướng.
    """
    opened = _cross_pool(_cross_pool(x, "min"), "max")
    return (x - opened).clamp_min(0).mean()
