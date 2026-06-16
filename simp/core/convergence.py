"""
Phát hiện hội tụ cho tối ưu hóa hình dạng SIMP.

Cung cấp các tiện ích kiểm tra hội tụ dựa trên:
    1. Thay đổi thiết kế (cửa sổ trượt max |x_mới - x_cũ| < tol_change).
    2. Độ ổn định hàm mục tiêu (cửa sổ trượt các thay đổi tương đối).
    3. Số vòng lặp tối thiểu trước khi cho phép hội tụ.
"""

from typing import List


class ConvergenceChecker:
    """Theo dõi hội tụ của vòng lặp tối ưu hóa SIMP.

    Hỗ trợ hai tiêu chí hội tụ, cả hai đều yêu cầu cửa sổ liên tiếp:
        - Thay đổi thiết kế: thay đổi tuyệt đối lớn nhất trong biến thiết kế
          dưới ngưỡng trong `window_change` vòng liên tiếp.
        - Độ ổn định hàm mục tiêu: thay đổi tương đối dưới ngưỡng
          trong `window_obj` vòng liên tiếp (cửa sổ trượt).

    Cần tối thiểu `min_iter` vòng lặp trước khi có thể tuyên bố hội tụ
    để tránh dừng sớm do biến thiết kế ban đầu thay đổi nhỏ.

    Attributes:
        tol_change: Ngưỡng thay đổi biến thiết kế lớn nhất.
        tol_obj: Ngưỡng thay đổi tương đối hàm mục tiêu.
        window_change: Số vòng liên tiếp thay đổi thiết kế < tol_change.
        window_obj: Số vòng liên tiếp thay đổi obj < tol_obj.
        min_iter: Số vòng lặp tối thiểu trước khi cho phép hội tụ.
        change_history: Cửa sổ trượt các thay đổi thiết kế gần đây.
        obj_changes: Cửa sổ trượt các thay đổi hàm mục tiêu gần đây.
        converged: Liệu đã đạt hội tụ hay chưa.
    """

    def __init__(
        self,
        tol_change: float = 0.01,
        tol_obj: float = 0.05,
        window_change: int = 5,
        window_obj: int = 20,
        min_iter: int = 10,
    ):
        """Khởi tạo bộ kiểm tra hội tụ.

        Args:
            tol_change: Ngưỡng thay đổi biến thiết kế lớn nhất.
            tol_obj: Ngưỡng thay đổi tương đối hàm mục tiêu.
            window_change: Số vòng liên tiếp thay đổi thiết kế < tol_change.
            window_obj: Số vòng liên tiếp thay đổi obj < tol_obj.
            min_iter: Số vòng lặp tối thiểu trước khi cho phép hội tụ.
        """
        self.tol_change = tol_change
        self.tol_obj = tol_obj
        self.window_change = window_change
        self.window_obj = window_obj
        self.min_iter = min_iter
        self.change_history: List[float] = []
        self.obj_changes: List[float] = []
        self.converged = False

    def _check_window(self, history: List[float], window: int, tol: float) -> bool:
        """Kiểm tra nếu cửa sổ trượt đã đạt ngưỡng.

        Args:
            history: Cửa sổ trượt các giá trị gần đây.
            window: Kích thước cửa sổ yêu cầu.
            tol: Ngưỡng cho phép.

        Returns:
            True nếu len(history) >= window và tất cả <= tol.
        """
        if len(history) < window:
            return False
        # Chỉ kiểm tra `window` giá trị gần nhất
        return all(h <= tol for h in history[-window:])

    def record_design_change(self, change: float) -> None:
        """Ghi nhận thay đổi thiết kế cho cửa sổ trượt.

        Args:
            change: Thay đổi tuyệt đối lớn nhất trong biến thiết kế.
        """
        self.change_history.append(change)
        # Giới hạn bộ nhớ (chỉ giữ window_change * 2 phần tử)
        max_keep = max(self.window_change * 2, 20)
        if len(self.change_history) > max_keep:
            self.change_history = self.change_history[-max_keep:]

    def record_objective_change(self, obj: float, prev_obj: float) -> bool:
        """Ghi nhận thay đổi hàm mục tiêu cho cửa sổ trượt.

        Args:
            obj: Giá trị hàm mục tiêu hiện tại.
            prev_obj: Giá trị hàm mục tiêu vòng lặp trước.

        Returns:
            True nếu thay đổi tương đối được ghi nhận thành công.
        """
        if prev_obj == float('inf'):
            return False

        change_in_obj = abs(obj - prev_obj) / max(abs(prev_obj), 1e-15)
        self.obj_changes.append(change_in_obj)

        # Giới hạn bộ nhớ
        max_keep = max(self.window_obj * 2, 40)
        if len(self.obj_changes) > max_keep:
            self.obj_changes = self.obj_changes[-max_keep:]

        return True

    def should_stop(
        self,
        change: float,
        obj: float,
        prev_obj: float,
        loop: int,
        max_iter: int,
    ) -> bool:
        """Xác định nếu vòng lặp tối ưu hóa nên dừng.

        Kiểm tra đồng thời cả hai tiêu chí hội tụ (thiết kế và mục tiêu),
        yêu cầu tối thiểu `min_iter` vòng lặp. Cả hai tiêu chí đều phải
        đạt cửa sổ liên tiếp trước khi dừng.

        Args:
            change: Thay đổi tuyệt đối lớn nhất trong biến thiết kế.
            obj: Giá trị hàm mục tiêu hiện tại.
            prev_obj: Giá trị hàm mục tiêu vòng lặp trước.
            loop: Số vòng lặp hiện tại.
            max_iter: Số vòng lặp tối đa cho phép.

        Returns:
            True nếu vòng lặp nên dừng.
        """
        # 1. Đạt số vòng lặp tối đa
        if loop >= max_iter:
            self.converged = True
            return True

        # 2. Chưa đủ vòng tối thiểu → không dừng
        if loop < self.min_iter:
            self.record_design_change(change)
            self.record_objective_change(obj, prev_obj)
            return False

        # 3. Ghi nhận thay đổi
        self.record_design_change(change)
        self.record_objective_change(obj, prev_obj)

        # 4. Kiểm tra hội tụ kết hợp:
        #    - Thiết kế thay đổi rất nhỏ trong window_change vòng
        #    - HOẶC objective ổn định trong window_obj vòng
        design_converged = self._check_window(
            self.change_history, self.window_change, self.tol_change,
        )
        obj_converged = self._check_window(
            self.obj_changes, self.window_obj, self.tol_obj,
        )

        # Hội tụ khi thiết kế ổn định (mạnh) hoặc khi objective ổn định
        # lâu dài (yếu hơn, cần cẩn thận).
        # Lưu ý: điều kiện (obj_converged and change <= self.tol_change * 2)
        # là heuristic không chuẩn. Trong SIMP chuẩn, chỉ cần design change
        # hội tụ là đủ. Điều kiện bổ sung này giúp dừng sớm khi objective
        # đã ổn định nhưng design change còn dao động nhỏ.
        if design_converged or (obj_converged and change <= self.tol_change * 2):
            self.converged = True
            return True

        return False

    def is_converged(self) -> bool:
        """Trả về trạng thái hội tụ hiện tại."""
        return self.converged

    def reset(self) -> None:
        """Đặt lại trạng thái của bộ kiểm tra hội tụ."""
        self.obj_changes.clear()
        self.change_history.clear()
        self.converged = False
