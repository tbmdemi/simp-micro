from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class SimpConfig:
    """Cấu hình SIMP dạng dataclass (chỉ hỗ trợ mục tiêu auxetic).

    Lưu ý: Đây là interface tham số phụ. Interface chính hiện tại là
    dict params (xem simp/run.py, pipeline/params.py). Hai interface này
    song song tồn tại và có thể bị lệch. Khi thêm tham số mới, cần cập nhật
    cả hai.

    Để thống nhất, khuyến nghị dùng dict params cho pipeline screening
    và runner (vì dễ serialize/deserialize). SimpConfig phù hợp cho
    single-run CLI nhập từ bàn phím.
    """
    nelx: int = 100
    nely: int = 100
    volfrac: float = 0.4
    penal: float = 3.0
    rmin: float = 3.0
    ft: int = 2
    objective_type: str = 'auxetic'
    max_iter: int = 200
    move: float = 0.2
    save_every: int = 1
    scale_factor: int = 1
    beta: float = 1.0
    E0: float = 199.0
    Emin: float = 1e-9
    nu: float = 0.3
    mu: float = 0.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        assert self.nelx > 0, 'nelx must be positive'
        assert self.nely > 0, 'nely must be positive'
        assert 0 < self.volfrac <= 1, 'volfrac must be in (0, 1]'
        assert self.penal >= 1, 'penal must be >= 1'
        assert self.rmin > 0, 'rmin must be positive'
        assert self.ft in (1, 2), 'ft must be 1 or 2'
        assert self.objective_type == 'auxetic', (
            'Chỉ hỗ trợ mục tiêu auxetic. objective_type must be "auxetic".'
        )
        assert self.max_iter > 0, 'max_iter must be positive'
        assert 0 < self.move <= 1, 'move must be in (0, 1]'
        assert self.save_every > 0, 'save_every must be positive'
        assert self.scale_factor >= 1, 'scale_factor must be >= 1'

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang dict."""

        return {
            'nelx': self.nelx,
            'nely': self.nely,
            'volfrac': self.volfrac,
            'penal': self.penal,
            'rmin': self.rmin,
            'ft': self.ft,
            'objective': self.objective_type,
            'max_iter': self.max_iter,
            'move': self.move,
            'save_every': self.save_every,
            'scale_factor': self.scale_factor,
            'beta': self.beta,
            'E0': self.E0,
            'Emin': self.Emin,
            'nu': self.nu,
            'void_size_frac': 0.4,
            'rotation_deg': 0.0,
        }
