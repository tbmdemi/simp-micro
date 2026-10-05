"""
Vendored `efficient_kan` package (Blealtan/efficient-kan, MIT license).

Cung cấp `EfficientKANLinear` - lớp KAN (Kolmogorov-Arnold Network) hiệu quả
dùng B-spline. Được vendor trực tiếp vào repo vì gói `efficient-kan` KHÔNG có
trên PyPI (`pip install efficient-kan` thất bại), trong khi codebase
(`pipeline/phase5_cvae/model.py`) đã import `from efficient_kan import
EfficientKANLinear`.

API:
    EfficientKANLinear(in_features, out_features, grid_size=5, spline_order=3,
                       grid_eps=0.02, base_activation=nn.SiLU)

Tham số (learnable):
    base_weight   : (out_features, in_features) - nhánh tuyến tính cơ sở
    spline_weight : (out_features, in_features, grid_size + spline_order)
                  - hệ số B-spline
Buffer:
    grid          : (in_features, grid_size + 2*spline_order + 1)
                  - lưới B-spline mở rộng (spline_order điểm ngoài mỗi bên)

Forward:
    y = base_activation(x) @ base_weight^T + B(x) @ spline_weight^T
    với B(x) là các hàm cơ sở B-spline bậc spline_order đánh giá tại x.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rebuild_kan_grid(
    old_grid: torch.Tensor, new_in_features: int
) -> torch.Tensor:
    """Tái tạo lưới B-spline cho `in_features` mới từ một lưới cũ.

    Lưới chỉ phụ thuộc vào `in_features`, `grid_size` và `spline_order`
    (tất cả các hàng giống hệt nhau do `.expand()`), KHÔNG phải tham số
    học. Hàm này được `pipeline/phase5_cvae/train.py::resize_condition_dim_weights`
    dùng khi `--resume-from` một checkpoint KAN có `condition_dim` khác: vì
    `in_features = base_dim + condition_dim`, đổi `condition_dim` khiến cả
    `base_weight`/`spline_weight` lẫn buffer `grid` đổi shape.

    Args:
        old_grid: buffer grid cũ, shape (old_in, grid_size + 2*spline_order + 1).
        new_in_features: chiều input mới (base_dim + condition_dim mới).

    Returns:
        Grid mới shape (new_in_features, grid_size + 2*spline_order + 1).
    """
    row = old_grid[0]  # mọi hàng giống nhau, đọc hàng đầu là đủ
    h = float((row[1] - row[0]).item())
    # row[0] = -spline_order*h - 1 ; row[-1] = (grid_size+spline_order)*h - 1
    spline_order = int(round((-row[0].item() - 1.0) / h))
    grid_size = int(round((row[-1].item() + 1.0) / h)) - spline_order
    return (
        torch.arange(
            -spline_order,
            grid_size + spline_order + 1,
            dtype=old_grid.dtype,
            device=old_grid.device,
        )
        .mul(h)
        .add(-1.0)
        .expand(new_in_features, -1)
        .contiguous()
    )


class EfficientKANLinear(nn.Module):
    """Lớp KAN tuyến tính hiệu quả dựa trên B-spline.

    Mỗi chiều input `x_i` được biến đổi bởi một spline B-spline bậc
    `spline_order` cộng với nhánh tuyến tính `base_activation(x_i) * w_i`
    (paper "Kolmogorov-Arnold Networks", Liu et al. 2024). Đây là biến thể
    "efficient" dùng lưới B-spline cố định thay vì nội suy spline đắt đỏ.

    forward(x): x shape (..., in_features) -> (..., out_features).
    """

    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        grid_eps=0.02,
        base_activation=nn.SiLU,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        # `grid_eps` giữ lại chỉ để tương thích API với efficient-kan gốc;
        # lưới dưới đây phủ [-1, 1] (cộng spline_order điểm ngoài mỗi bên)
        # nên không cần tới nó.
        self.grid_eps = grid_eps

        # Lưới mở rộng: grid_size khoảng trên [-1, 1], thêm spline_order
        # điểm mỗi bên để đệ quy Cox-de Boor đánh giá cơ sở B-spline bậc
        # spline_order mà không cần xử lý biên riêng.
        h = 2.0 / grid_size
        grid = (
            torch.arange(
                -spline_order,
                grid_size + spline_order + 1,
                dtype=torch.float32,
            )
            .mul(h)
            .add(-1.0)
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        self.base_activation = (
            base_activation() if base_activation is not None else None
        )

        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5))

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Đánh giá cơ sở B-spline bậc `spline_order` tại `x`.

        Args:
            x: tensor shape (..., in_features).

        Returns:
            Tensor shape (..., in_features, grid_size + spline_order) - giá
            trị các hàm cơ sở B-spline cho từng chiều input.
        """
        grid = self.grid  # (in_features, grid_size + 2*spline_order + 1)
        x = x.unsqueeze(-1)  # (..., in_features, 1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            denom_left = grid[:, k:-1] - grid[:, :-(k+1)]
            denom_right = grid[:, k+1:] - grid[:, 1:-k]
            bases = (
                (x - grid[:, :-(k+1)]) / denom_left.clamp_min(1e-9) * bases[:, :, :-1]
                + (grid[:, k+1:] - x) / denom_right.clamp_min(1e-9) * bases[:, :, 1:]
            )
        return bases.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Ánh xạ x (..., in_features) -> (..., out_features)."""
        if self.base_activation is None:
            base_output = F.linear(x, self.base_weight)
        else:
            base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_bases = self.b_splines(x)  # (..., in, grid_size + spline_order)
        spline_output = F.linear(
            spline_bases.flatten(-2),
            self.spline_weight.reshape(self.out_features, -1),
        )
        # Ensure grid biên an toàn
        # assert torch.all(self.grid[:, 1:] > self.grid[:, :-1]), "Grid lỗi biên"
        return base_output + spline_output
