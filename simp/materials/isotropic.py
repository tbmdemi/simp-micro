"""
Định nghĩa vật liệu đẳng hướng cho tối ưu hóa hình dạng SIMP.

Cung cấp lớp Material chứa ma trận độ cứng phần tử (KE)
cho bài toán ứng suất phẳng (plane stress).
"""

import numpy as np


class Material:
    """Vật liệu đẳng hướng đàn hồi tuyến tính.

    Tính toán ma trận độ cứng phần tử cho phần tử tứ giác 4 nút
    trong điều kiện **ứng suất phẳng (plane stress)**.

    Attributes:
        E0: Modul đàn hồi Young của vật liệu đặc.
        Emin: Modul đàn hồi Young của lỗ rỗng.
        nu: Hệ số Poisson.
        KE: Ma trận độ cứng phần tử (8×8).
    """

    def __init__(self, E0: float = 199.0, Emin: float = 1e-9, nu: float = 0.3):
        """Khởi tạo vật liệu và tính ma trận độ cứng phần tử.

        Args:
            E0: Modul đàn hồi Young của vật liệu đặc (mặc định 199.0).
            Emin: Modul đàn hồi Young của lỗ rỗng (mặc định 1e-9).
            nu: Hệ số Poisson (mặc định 0.3).
        """
        self.E0 = E0
        self.Emin = Emin
        self.nu = nu
        self.KE = self._compute_element_stiffness()

    def _compute_element_stiffness(self) -> np.ndarray:
        """Tính ma trận độ cứng phần tử cho phần tử tứ giác 4 nút.

        Sử dụng tích phân số Gauss 2×2 cho bài toán **ứng suất phẳng (plane stress)**.
        Ma trận đàn hồi D:
            D = E/(1-ν²) × [[1, ν, 0], [ν, 1, 0], [0, 0, (1-ν)/2]]

        Returns:
            Ma trận độ cứng phần tử (8×8).
        """
        # Điểm Gauss và trọng số
        gauss_points = np.array([
            [-1 / np.sqrt(3), -1 / np.sqrt(3)],
            [1 / np.sqrt(3), -1 / np.sqrt(3)],
            [1 / np.sqrt(3), 1 / np.sqrt(3)],
            [-1 / np.sqrt(3), 1 / np.sqrt(3)],
        ])
        weights = np.ones(4)

        # Ma trận đàn hồi D cho ứng suất phẳng (plane stress)
        D = self.E0 / (1 - self.nu ** 2) * np.array([
            [1, self.nu, 0],
            [self.nu, 1, 0],
            [0, 0, (1 - self.nu) / 2],
        ])

        # Tọa độ nút phần tử (phần tử tham chiếu)
        node_coords = np.array([
            [-1, -1],
            [1, -1],
            [1, 1],
            [-1, 1],
        ])

        KE = np.zeros((8, 8))

        for gp, w in zip(gauss_points, weights):
            xi, eta = gp

            # Đạo hàm hàm dạng
            dN_dxi = 0.25 * np.array([
                [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)],
            ])

            # Ma trận Jacobi
            J = dN_dxi @ node_coords
            detJ = np.linalg.det(J)
            invJ = np.linalg.inv(J)

            # Đạo hàm hàm dạng trong tọa độ vật lý
            dN_dx = invJ @ dN_dxi

            # Ma trận B (biến dạng-chuyển vị)
            B = np.zeros((3, 8))
            for i in range(4):
                B[0, 2 * i] = dN_dx[0, i]
                B[1, 2 * i + 1] = dN_dx[1, i]
                B[2, 2 * i] = dN_dx[1, i]
                B[2, 2 * i + 1] = dN_dx[0, i]

            # Tích hợp
            KE += B.T @ D @ B * detJ * w

        return KE