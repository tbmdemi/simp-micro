"""
Pipeline tối ưu hóa tham số SIMP (3 Phase).

Phase 1: LHS Screening - Tìm tham số ảnh hưởng nhất trên lưới 50×50
Phase 2: Tuning với Optimization Algorithms - DE, SHGO, Basinhopping, L-BFGS-B
Phase 3: Validation - Xác nhận kết quả trên lưới 100×100
"""

__version__ = '0.1.0'