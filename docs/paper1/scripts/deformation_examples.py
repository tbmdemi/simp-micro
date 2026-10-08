"""Trường biến dạng dưới kéo đơn trục cho hình cơ chế auxetic (Bài #1,
review mục 3).

Với mỗi thiết kế ví dụ: giải homogenization (lưới verify 50², penal 3),
lấy ma trận mềm S = Q⁻¹, biến dạng vĩ mô dưới ứng suất đơn trục
σ = [1, 0, 0]ᵀ là ε = S·σ, rồi trường chuyển vị nút
    u(X) = ε_xx·u⁰₁ + ε_yy·u⁰₂ + γ_xy·u⁰₃ + (χ₁ ε_xx + χ₂ ε_yy + χ₃ γ_xy)
(u⁰_j: trường affine đơn vị, χ_j: dao động tuần hoàn của load case j).
Lưu tọa độ nút, phần dao động, ε vĩ mô và ảnh nhị phân để make_figures.py
vẽ lưới lát đã biến dạng mà không cần giải lại FE.

Kiểm chứng tự động: −ε_yy/ε_xx phải bằng ν12 verified (định nghĩa, sai số
máy) - nếu không, quy ước Voigt/chỉ số đã sai.

Thiết kế lấy từ pipeline cuối không force_periodic (quyết định C5,
2026-10-07): best-of-30 IN100 `p1_7_c5_in100_bo30_nofp.json` và OOD đổi dấu
`p1_9_r7_signflip_sym_nofp.json`.

Chạy từ gốc repo:
    python docs/paper1/scripts/deformation_examples.py
Output: outputs/phase5/plan_v3/p1_6c_deformation_examples.npz
"""

import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline", "phase5_cvae"))

from verify_fe import FE_PARAMS, resize_to_fe_grid  # noqa: E402

from simp.core.fem import build_dof_mesh  # noqa: E402
from simp.core.pbc import build_pbc  # noqa: E402
from simp.core.solver import solve_fe  # noqa: E402
from simp.homogenization.compute import (  # noqa: E402
    compute_homogenized_tensor,
)
from simp.materials.isotropic import Material  # noqa: E402
from simp.objectives.auxetic import compute_nu12  # noqa: E402

PLAN_V3 = os.path.join(REPO_ROOT, "outputs", "phase5", "plan_v3")


def _guarded_image(row: dict) -> np.ndarray:
    """Ảnh nhị phân 64² của thiết kế cuối chế độ guarded."""
    key = "refined" if row["guarded_accept"] else "baseline"
    return np.asarray(row[f"{key}_image"], dtype=np.float32)


def uniaxial_deformation(design50: np.ndarray) -> dict:
    """Biến dạng vĩ mô + trường dao động dưới σ = [1, 0, 0]ᵀ.

    Args:
        design50: ảnh nhị phân (50, 50), hàng = chỉ số y của phần tử.

    Returns:
        dict: eps (3,), nu12, fluct (nnode_y, nnode_x, 2) phần dao động
        tuần hoàn của chuyển vị nút (đơn vị: cùng ε), design.
    """
    nely, nelx = design50.shape
    p = FE_PARAMS
    mat = Material(E0=p["E0"], Emin=p["Emin"], nu=p["nu"])
    nodenrs, _, edofMat, iK, jK = build_dof_mesh(nelx, nely)
    pbc = build_pbc(nelx, nely, nodenrs)
    U, U0 = solve_fe(
        design50, mat.KE, iK, jK, pbc, p["penal"], p["E0"], p["Emin"]
    )
    Q, _, _ = compute_homogenized_tensor(
        U0 + U, U0, design50, mat.KE, edofMat, p["penal"], p["E0"], p["Emin"]
    )
    S = np.linalg.inv(Q)
    eps = S @ np.array([1.0, 0.0, 0.0])
    nu12 = compute_nu12(Q)
    # Kiểm chứng quy ước: tỉ số co ngang dưới kéo x phải là ν12.
    assert abs(-eps[1] / eps[0] - nu12) < 1e-9, "quy ước Voigt sai"
    chi = U @ eps  # (ndof,) dao động tuần hoàn
    # Nút (j, i): dof u = 2*(node-1), v = 2*(node-1)+1 (solver.py).
    node0 = nodenrs - 1
    fluct = np.stack([chi[2 * node0], chi[2 * node0 + 1]], axis=-1)
    return {
        "eps": eps,
        "nu12": float(nu12),
        "fluct": fluct,
        "design": design50,
    }


def main():
    """Tính và lưu trường biến dạng cho 2 thiết kế ví dụ."""
    with open(os.path.join(PLAN_V3, "p1_7_c5_in100_bo30_nofp.json")) as f:
        cv = json.load(f)["per_condition"]
    with open(os.path.join(PLAN_V3, "p1_9_r7_signflip_sym_nofp.json")) as f:
        sf = json.load(f)["per_condition"]
    # Auxetic mạnh nhất trong IN100 và mục tiêu ν*=0,5 (dấu chưa thấy).
    i_aux = int(np.argmin([c["target_v12"] for c in cv]))
    i_pos = int(np.argmin([abs(c["target_v12"] - 0.5) for c in sf]))
    out, meta = {}, []
    for name, row in (("aux", cv[i_aux]), ("pos", sf[i_pos])):
        # Cùng đường verify của pipeline cuối (C5, không force_periodic):
        # ngưỡng 0,5 → resize nearest.
        img = resize_to_fe_grid(
            (_guarded_image(row) > 0.5).astype(np.float32),
            50,
            50,
        )
        r = uniaxial_deformation(img)
        for k in ("eps", "fluct", "design"):
            out[f"{name}_{k}"] = r[k]
        meta.append(
            {
                "name": name,
                "target": [row["target_v12"], row["target_v21"]],
                "nu12": r["nu12"],
            }
        )
        print(name, meta[-1], "eps", r["eps"])
    np.savez_compressed(
        os.path.join(PLAN_V3, "p1_6c_deformation_examples.npz"), **out
    )
    with open(
        os.path.join(PLAN_V3, "p1_6c_deformation_examples.json"), "w"
    ) as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
