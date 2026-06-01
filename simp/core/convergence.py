"""
Phát hiện hội tụ cho tối ưu hóa hình dạng SIMP.

Cung cấp các tiện ích kiểm tra hội tụ dựa trên:
    1. Thay đổi thiết kế (max |x_mới - x_cũ| < tol_change).
    2. Độ ổn định hàm mục tiêu (cửa sổ trượt các thay đổi tương đối).
"""

from typing import List, Optional


class ConvergenceChecker:
    """Theo dõi hội tụ của vòng lặp tối ưu hóa SIMP.

    Hỗ trợ hai tiêu chí hội tụ:
        - Thay đổi thiết kế: thay đổi tuyệt đối lớn nhất trong biến thiết kế.
        - Độ ổn định hàm mục tiêu: thay đổi tương đối dưới ngưỡng
          trong một số vòng lặp liên tiếp (cửa sổ trượt).

    Attributes:
        tol_change: Ngưỡng thay đổi biến thiết kế lớn nhất.
        tol_obj: Ngưỡng thay đổi tương đối hàm mục tiêu.
        window_size: Số vòng lặp liên tiếp để kiểm tra hàm mục tiêu.
        obj_changes: Cửa sổ trượt các thay đổi hàm mục tiêu gần đây.
        converged: Liệu đã đạt hội tụ hay chưa.
    """

    def __init__(
        self,
        tol_change: float = 0.01,
        tol_obj: float = 0.05,
        window_size: int = 20,
    ):
        """Khởi tạo bộ kiểm tra hội tụ.

        Args:
            tol_change: Ngưỡng thay đổi biến thiết kế lớn nhất.
            tol_obj: Ngưỡng thay đổi tương đối hàm mục tiêu.
            window_size: Số vòng lặp liên tiếp để kiểm tra hàm mục tiêu.
        """
        self.tol_change = tol_change
        self.tol_obj = tol_obj
        self.window_size = window_size
        self.obj_changes: List[float] = []
        self.converged = False

    def check_design_change(self, change: float) -> bool:
        """Kiểm tra nếu thay đổi thiết kế dưới ngưỡng.

        Args:
            change: Thay đổi tuyệt đối lớn nhất trong biến thiết kế.

        Returns:
            True nếu change <= tol_change (thiết kế đã hội tụ).
        """
        return change <= self.tol_change

    def check_objective_stability(self, obj: float, prev_obj: float) -> bool:
        """Kiểm tra độ ổn định hàm mục tiêu dùng cửa sổ trượt.

        Tính thay đổi tương đối của hàm mục tiêu và kiểm tra nếu
        `window_size` vòng lặp cuối đều có thay đổi dưới `tol_obj`.

        Args:
            obj: Giá trị hàm mục tiêu hiện tại.
            prev_obj: Giá trị hàm mục tiêu vòng lặp trước.

        Returns:
            True nếu hàm mục tiêu đã ổn định (đạt cửa sổ liên tiếp).
        """
        if prev_obj == float('inf'):
            return False

        change_in_obj = abs(obj - prev_obj) / max(abs(prev_obj), 1e-15)
        self.obj_changes.append(change_in_obj)

        # Chỉ giữ window_size mục gần nhất
        if len(self.obj_changes) > self.window_size:
            self.obj_changes.pop(0)

        # Kiểm tra nếu tất cả mục trong cửa sổ đều dưới ngưỡng
        if (len(self.obj_changes) == self.window_size
                and all(o < self.tol_obj for o in self.obj_changes)):
            self.converged = True
            return True

        return False

    def should_stop(
        self,
        change: float,
        obj: float,
        prev_obj: float,
        loop: int,
        max_iter: int,
    ) -> bool:
        """Xác định nếu vòng lặp tối ưu hóa nên dừng.

        Kiểm tra cả hai tiêu chí hội tụ và giới hạn vòng lặp.

        Args:
            change: Thay đổi tuyệt đối lớn nhất trong biến thiết kế.
            obj: Giá trị hàm mục tiêu hiện tại.
            prev_obj: Giá trị hàm mục tiêu vòng lặp trước.
            loop: Số vòng lặp hiện tại.
            max_iter: Số vòng lặp tối đa cho phép.

        Returns:
            True nếu vòng lặp nên dừng (đã hội tụ hoặc đạt max vòng lặp).
        """
        # Đạt số vòng lặp tối đa
        if loop >= max_iter:
            return True

        # Thay đổi thiết kế đã hội tụ
        if self.check_design_change(change):
            self.converged = True
            return True

        # Hàm mục tiêu đã ổn định
        if self.check_objective_stability(obj, prev_obj):
            return True

        return False

    def reset(self) -> None:
        """Đặt lại trạng thái của bộ kiểm tra hội tụ."""
        self.obj_changes.clear()
        self.converged = False
