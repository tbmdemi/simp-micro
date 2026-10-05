"""
Tests for analysis/pareto/frontier.py::is_pareto_efficient().

Bug đã sửa 2026-08-20 (LIMITATIONS.md, EXPERIMENT_LOG.md): bản trước luôn
đánh dấu SAI phần tử CUỐI CÙNG của mảng đầu vào là không-Pareto-efficient,
bất kể nó có thực sự bị lấn át hay không (kể cả điểm tốt nhất tuyệt đối,
hoặc mảng chỉ có 1 điểm). Chưa từng có test nào cho module này trước đây -
bug bị bỏ sót vì analysis/pareto/ chưa từng chạy thật cho kết quả nào đã
công bố (không có output/pareto/* trên đĩa).
"""

import numpy as np

from analysis.pareto.frontier import is_pareto_efficient


class TestIsParetoEfficient:
    def test_single_point_is_always_efficient(self):
        """Regression cho bug chính: 1 điểm luôn Pareto-efficient tầm
        thường (không có ai để so sánh) - bản cũ trả về False."""
        result = is_pareto_efficient(np.array([[1.0, 2.0]]), maximize=True)
        assert result.tolist() == [True]

    def test_best_point_as_last_row_is_still_efficient(self):
        """Regression cho bug chính: điểm tốt nhất tuyệt đối đặt ở HÀNG
        CUỐI của mảng vẫn phải được đánh dấu True - bản cũ luôn trả False
        cho hàng cuối bất kể giá trị thật."""
        costs = np.array([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]])
        result = is_pareto_efficient(costs, maximize=True)
        assert result.tolist() == [False, False, True]

    def test_best_point_as_middle_row_is_efficient(self):
        costs = np.array([[5.0, 5.0], [1.0, 1.0], [3.0, 3.0]])
        result = is_pareto_efficient(costs, maximize=True)
        assert result.tolist() == [True, False, False]

    def test_dominant_point_as_last_row_is_efficient(self):
        """2 điểm, điểm SAU lấn át điểm trước - trường hợp trực tiếp gây
        vòng lặp vô hạn ở Nhóm 3.2 (non-domination ranking lặp lại)."""
        costs = np.array([[1.0, 1.0], [2.0, 2.0]])
        result = is_pareto_efficient(costs, maximize=True)
        assert result.tolist() == [False, True]

    def test_non_dominated_pair_both_efficient(self):
        """2 điểm không lấn át nhau (mỗi điểm tốt hơn ở 1 trục) - cả 2
        đều phải là Pareto-efficient."""
        costs = np.array([[1.0, 5.0], [5.0, 1.0]])
        result = is_pareto_efficient(costs, maximize=True)
        assert result.tolist() == [True, True]

    def test_minimize_mode(self):
        costs = np.array([[1.0, 1.0], [2.0, 2.0]])
        result = is_pareto_efficient(costs, maximize=False)
        assert result.tolist() == [True, False]

    def test_identical_points_both_efficient(self):
        """2 điểm giống hệt nhau - không ai lấn át ai thật sự (cần lấn át
        NGHIÊM NGẶT ở ít nhất 1 trục), cả 2 đều Pareto-efficient."""
        costs = np.array([[2.0, 2.0], [2.0, 2.0]])
        result = is_pareto_efficient(costs, maximize=True)
        assert result.tolist() == [True, True]

    def test_repeated_ranking_terminates_and_covers_all_points(self):
        """Nhóm 3.2 (PROJECT_PLAN.md): dùng is_pareto_efficient() LẶP LẠI
        để xếp hạng tầng Pareto (loại tầng 1, tìm tầng 2 trên phần còn
        lại, v.v.) - trước khi sửa, điểm cuối cùng còn lại ở mỗi tầng
        không bao giờ được xếp hạng, gây vòng lặp vô hạn. Test này lặp lại
        đúng pattern đó trên dữ liệu ngẫu nhiên cỡ thật (n=30) và xác nhận
        vòng lặp DỪNG với mọi điểm được xếp hạng."""
        rng = np.random.default_rng(0)
        costs = rng.random((30, 3))

        n = costs.shape[0]
        ranks = np.zeros(n, dtype=int)
        remaining = np.arange(n)
        rank = 1
        max_iters = (
            n + 1
        )  # khong the vuot qua n tang trong truong hop xau nhat
        iters = 0
        while len(remaining) > 0:
            iters += 1
            assert iters <= max_iters, "vòng lặp không hội tụ (regression)"
            mask = is_pareto_efficient(costs[remaining], maximize=True)
            assert (
                mask.any()
            ), "is_pareto_efficient trả về toàn False (regression)"
            ranks[remaining[mask]] = rank
            remaining = remaining[~mask]
            rank += 1

        assert (ranks > 0).all()
