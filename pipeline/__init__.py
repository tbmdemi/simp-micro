"""
Pipeline tối ưu hóa tham số SIMP (3 Phase).

Phase 1: LHS Screening — Tìm tham số ảnh hưởng nhất trên lưới 50×50
Phase 2: Bayesian Optimization — Tối ưu cục bộ trên lưới 80×80
Phase 3: Validation — Xác nhận kết quả trên lưới 100×100
"""

__version__ = '0.1.0'