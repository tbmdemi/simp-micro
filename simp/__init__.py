"""
SIMP - Tối ưu hóa hình dạng (Topology Optimization) cho thiết kế vi cấu trúc vật liệu tuần hoàn.

Bản Python của phương pháp SIMP (Solid Isotropic Material with Penalization)
dùng để tối ưu hóa hình dạng các ô cơ sở tuần hoàn nhằm đạt tính chất auxetic.

Các gói con:
    core:           Phân tích FEM, lọc, PBC, solver, OC, phát hiện hội tụ.
    materials:      Tính chất vật liệu và ma trận độ cứng phần tử.
    objectives:     Hàm mục tiêu (loại thứ nhất và thứ hai).
    seeds:          Bộ sinh mẫu lỗ rỗng ban đầu.
    homogenization: Tính toán ten-xơ độ cứng đồng nhất hóa.
    io:             Công cụ ghi log và trực quan hóa.
"""

__version__ = '1.4.0'
__author__ = 'AuxForge Team'
__license__ = 'MIT'
