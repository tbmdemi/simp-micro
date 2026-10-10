"""
Phase 5 - benchmark_simp_baseline.py
============================================================
Baseline "SIMP chạy từ đầu" cho Bài #1 (plan.md v3, P1.6a): với MỖI target
(ν12*, ν21*) giải bài toán inverse homogenization cổ điển (Sigmund 1994)

    min_x  ½‖ν(x̂) − ν*‖²
    s.t.   mean(x̂) ≤ volfrac_max,  Q11, Q22 ≥ δ (= stiffness_delta)

bằng MMA (nlopt.LD_MMA), x̂ = H_β(filter(x)) với Heaviside continuation
β ∈ {1, 2, …, 64} (Wang, Lazarov & Sigmund 2011). Không có projection
thì SIMP dừng ở lời giải xám - đúng failure mode bài báo phê phán ở refine
liên tục - nên so sánh với cVAE sẽ không công bằng.

Kiểm chứng GIỐNG HỆT pipeline cVAE: ngưỡng 0,5 trên x̂ cuối → FE độc lập
(`verify_fe.evaluate_density_field`, penal 3) → (ν12, ν21) verified. Ô SIMP
vốn tuần hoàn (PBC), không cần force_periodic hay resize.

Multi-start từ các seed topology của dataset (mặc định 4 seed phổ biến
nhất); "best-of-seeds" chọn theo sai số verified (FE oracle, giống
best-of-N của cVAE) và tính chi phí FE = TỔNG mọi seed. 2 ngân sách:
  * `full`   - tối đa 43 eval/mức β (≤ 301 FE-solve/seed): SIMP hội tụ.
  * `matched`- tối đa 15 eval/mức β (≤ 105/seed): xấp xỉ ngân sách
               best-of-30 + refine của cVAE (107 FE/condition).

Target đọc từ JSON kết quả cVAE (mặc định P1.1e-b, IN100) để so sánh ghép
cặp theo condition, cùng thứ tự.

Cách chạy:
    python pipeline/phase5_cvae/benchmark_simp_baseline.py \
        --n-conditions 100 --workers 10 \
        --out outputs/phase5/plan_v3/p1_6a_simp_baseline_in100.json
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

# FE scipy sparse solve nhanh nhất ở 1 thread/worker khi chạy song song
# nhiều worker - đặt TRƯỚC import numpy.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import nlopt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from manufacturability import check_manufacturability  # noqa: E402
from verify_fe import FE_PARAMS, evaluate_density_field  # noqa: E402

from simp.core.fem import build_dof_mesh  # noqa: E402
from simp.core.filter import (  # noqa: E402
    apply_heaviside_projection,
    build_filter,
    heaviside_projection_derivative,
)
from simp.core.oc import X_MIN  # noqa: E402
from simp.core.pbc import build_pbc  # noqa: E402
from simp.core.solver import solve_fe  # noqa: E402
from simp.homogenization.compute import (  # noqa: E402
    compute_homogenized_tensor,
)
from simp.materials.isotropic import Material  # noqa: E402
from simp.objectives.auxetic import stiffness_delta  # noqa: E402
from simp.runner import SEED_MAP  # noqa: E402

PLAN_V3_DIR = os.path.join(REPO_ROOT, "outputs", "phase5", "plan_v3")
DEFAULT_TARGETS_JSON = os.path.join(
    PLAN_V3_DIR, "p1_1e_b_v2_finetuned_in100_bo30.json"
)
# 4 seed nhiều mẫu nhất trong dataset_64.npz (hourglass 7088, hexagonal 2119,
# reentrant_bowtie 1143, circle 456) - SIMP được cùng "tri thức miền" về
# điểm khởi tạo như dữ liệu cVAE học, so sánh không thiên vị cVAE.
DEFAULT_SEEDS = ("hourglass", "hexagonal", "reentrant_bowtie", "circle")
# Trần β=64 giống refine P1.1e: smoke-test với trần 32 cho thấy SIMP CŨNG
# khai thác vật liệu xám (reentrant_bowtie: ν21 xám −0,636 → nhị phân −1,15).
BETAS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
BUDGETS = {"full": 43, "matched": 15}
# Tham số SIMP = trung vị dataset (volfrac 0,537, penal 3, rmin 1,5,
# void_size_frac 0,315); volfrac làm cận TRÊN để target cần ít vật liệu
# hơn vẫn khả thi.
SIMP_PARAMS = {
    "volfrac_max": 0.55,
    "penal": 3.0,
    "rmin": 1.5,
    "void_size_frac": 0.315,
}


class TargetProblem:
    """Bài toán inverse homogenization ½‖ν − ν*‖² trên lưới FE 50×50.

    Đóng gói mesh/filter (dựng 1 lần) và hàm đánh giá có cache theo x để
    callback objective + 3 ràng buộc của nlopt dùng chung 1 lần giải FE.

    Args:
        target: (ν12*, ν21*).
        nelx, nely: kích thước lưới FE (mặc định = FE_PARAMS, 50×50).
        params: tham số SIMP (xem SIMP_PARAMS).
    """

    def __init__(
        self,
        target,
        nelx: int = 50,
        nely: int = 50,
        params: dict = SIMP_PARAMS,
    ):
        self.target = np.asarray(target, dtype=np.float64)
        self.nelx, self.nely = nelx, nely
        self.penal = params["penal"]
        self.volfrac_max = params["volfrac_max"]
        self.E0, self.Emin = FE_PARAMS["E0"], FE_PARAMS["Emin"]
        self.material = Material(
            E0=self.E0, Emin=self.Emin, nu=FE_PARAMS["nu"]
        )
        nodenrs, _, self.edofMat, self.iK, self.jK = build_dof_mesh(nelx, nely)
        self.pbc = build_pbc(nelx, nely, nodenrs)
        self.H, self.Hs = build_filter(nelx, nely, params["rmin"])
        self.delta = stiffness_delta(self.volfrac_max, self.E0)
        self.beta = BETAS[0]
        self.n_evals = 0
        self._cache_key = None
        self._cache = None

    def physical(self, x_flat: np.ndarray):
        """x (thiết kế, flatten 'F') → (x̃ đã lọc, x̂ đã chiếu), (nely, nelx)."""
        x_tilde = np.reshape(
            self.H @ x_flat / self.Hs, (self.nely, self.nelx), order="F"
        )
        return x_tilde, apply_heaviside_projection(x_tilde, self.beta)

    def _chain(self, d_xhat: np.ndarray, x_tilde: np.ndarray) -> np.ndarray:
        """Lan truyền ngược d/dx̂ → d/dx qua projection rồi filter (H đối
        xứng nên Hᵀ = H; chia Hs trước khi nhân, đúng chuẩn density filter).
        """
        d_tilde = d_xhat * heaviside_projection_derivative(x_tilde, self.beta)
        return self.H @ (d_tilde.flatten("F") / self.Hs)

    def evaluate(self, x_flat: np.ndarray) -> dict:
        """1 lần giải FE + homogenization tại x, kèm mọi gradient theo x.

        Returns:
            dict: c, dc (objective), g11/dg11, g22/dg22 (ràng buộc
            δ − Q_ii ≤ 0), vol/dvol (mean(x̂) − volfrac_max ≤ 0), v12, v21,
            xhat.
        """
        key = x_flat.tobytes()
        if key == self._cache_key:
            return self._cache
        x_tilde, xhat = self.physical(x_flat)
        U, U0 = solve_fe(
            xhat,
            self.material.KE,
            self.iK,
            self.jK,
            self.pbc,
            self.penal,
            self.E0,
            self.Emin,
        )
        Q, dQ, _ = compute_homogenized_tensor(
            U0 + U,
            U0,
            xhat,
            self.material.KE,
            self.edofMat,
            self.penal,
            self.E0,
            self.Emin,
        )
        self.n_evals += 1
        # ν và dν/dx̂ qua S = Q⁻¹ (cùng công thức real_physics.py, 1 nguồn
        # đạo hàm đã kiểm finite-difference ở đó).
        S = np.linalg.inv(Q)
        dS = -np.einsum("ik,klpq,lm->impq", S, dQ, S)
        v12, v21 = -S[0, 1] / S[0, 0], -S[0, 1] / S[1, 1]
        dv12 = -(dS[0, 1] * S[0, 0] - S[0, 1] * dS[0, 0]) / S[0, 0] ** 2
        dv21 = -(dS[0, 1] * S[1, 1] - S[0, 1] * dS[1, 1]) / S[1, 1] ** 2
        r12, r21 = v12 - self.target[0], v21 - self.target[1]
        nele = self.nelx * self.nely
        out = {
            "c": 0.5 * (r12**2 + r21**2),
            "dc": self._chain(r12 * dv12 + r21 * dv21, x_tilde),
            "g11": self.delta - Q[0, 0],
            "dg11": self._chain(-dQ[0, 0], x_tilde),
            "g22": self.delta - Q[1, 1],
            "dg22": self._chain(-dQ[1, 1], x_tilde),
            "vol": float(xhat.mean()) - self.volfrac_max,
            "dvol": self._chain(np.full_like(xhat, 1.0 / nele), x_tilde),
            "v12": float(v12),
            "v21": float(v21),
            "xhat": xhat,
        }
        self._cache_key, self._cache = key, out
        return out


def initial_design(
    seed_name: str, nelx: int, nely: int, params: dict = SIMP_PARAMS
) -> np.ndarray:
    """Mật độ khởi tạo từ seed topology của dataset (flatten 'F').

    Co mật độ về đúng volfrac_max nếu seed thô đặc hơn (circle/hexagonal ở
    void_size 0,315 có thể tích ~0,92): x0 vi phạm ràng buộc thể tích khiến
    MMA thoát sau vài eval về ô gần đặc (ν ≈ ν0) - smoke-test 2026-10-06.
    Clip về [X_MIN, 1] vì nlopt từ chối x0 vi phạm bound (xem mma_runner.py).

    Args:
        seed_name: khóa trong SEED_MAP.
        nelx, nely: lưới FE.
        params: tham số SIMP (volfrac_max, void_size_frac).

    Returns:
        np.ndarray (nelx*nely,) mật độ khởi tạo khả thi.
    """
    fn = SEED_MAP[seed_name]
    if seed_name == "hourglass":
        x0 = fn(nelx, nely, params["volfrac_max"], 0.0)
    else:
        x0 = fn(nelx, nely, params["void_size_frac"], 0.0)
    if x0.mean() > params["volfrac_max"]:
        x0 = x0 * params["volfrac_max"] / x0.mean()
    return np.clip(x0, X_MIN, 1.0).flatten("F")


def run_simp_target(
    target,
    seed_name: str,
    evals_per_beta: int,
    nelx: int = 50,
    nely: int = 50,
    x0: np.ndarray = None,
    betas: tuple = None,
) -> dict:
    """Giải 1 target từ 1 seed với continuation β, rồi verify độc lập.

    Args:
        target: (ν12*, ν21*).
        seed_name: khóa trong simp.runner.SEED_MAP (chỉ là nhãn khi có x0).
        evals_per_beta: số FE-eval tối đa mỗi mức β (ngân sách).
        nelx, nely: lưới FE.
        x0: điểm khởi tạo (nelx*nely,) flatten 'F', đã khả thi - dùng cho
            thí nghiệm lai cVAE → SIMP (P1.7). None = seed topology.
        betas: các mức β continuation. None = BETAS (đọc lúc gọi, để test
            monkeypatch được).

    Returns:
        dict: v12/v21 verified (ngưỡng 0,5 + FE độc lập, None nếu FE lỗi),
        v12/v21 của x̂ xám cuối (raw), n_fe (số FE-solve tối ưu + 1 verify),
        thời gian, manufacturability của ảnh nhị phân, ảnh nhị phân (list).
    """
    prob = TargetProblem(target, nelx, nely)
    x = initial_design(seed_name, nelx, nely) if x0 is None else x0.copy()
    t0 = time.time()

    def _wrap(key, dkey):
        def cb(x_flat, grad):
            r = prob.evaluate(x_flat)
            if grad.size > 0:
                grad[:] = r[dkey]
            return float(r[key])

        return cb

    for beta in BETAS if betas is None else betas:
        # Mỗi mức β là 1 bài toán con mới (projection đổi) → khởi động lại
        # MMA, giống restart L-BFGS mỗi mức β của refine P1.1e.
        prob.beta = beta
        prob._cache_key = None
        opt = nlopt.opt(nlopt.LD_MMA, x.size)
        opt.set_lower_bounds(X_MIN)
        opt.set_upper_bounds(1.0)
        opt.set_min_objective(_wrap("c", "dc"))
        opt.add_inequality_constraint(_wrap("vol", "dvol"), 1e-6)
        opt.add_inequality_constraint(_wrap("g11", "dg11"), 1e-3)
        opt.add_inequality_constraint(_wrap("g22", "dg22"), 1e-3)
        opt.set_maxeval(evals_per_beta)
        opt.set_ftol_rel(1e-6)
        try:
            x = opt.optimize(x)
        except Exception:
            # RoundoffLimited/FE suy biến: giữ điểm tốt cuối cùng đã có
            # (nlopt không trả x khi raise) - cùng quy ước mma_runner.py.
            pass

    raw = prob.evaluate(x)
    elapsed = time.time() - t0
    img_bin = (raw["xhat"] > 0.5).astype(np.float64)
    try:
        v12, v21, _ = evaluate_density_field(img_bin, FE_PARAMS)
        v12, v21 = float(v12), float(v21)
        if not (np.isfinite(v12) and np.isfinite(v21)):
            v12 = v21 = None
    except Exception:
        v12 = v21 = None
    manuf = check_manufacturability(img_bin)
    return {
        "seed": seed_name,
        "v12": v12,
        "v21": v21,
        "raw_v12": raw["v12"],
        "raw_v21": raw["v21"],
        "raw_objective": float(raw["c"]),
        "volume": float(img_bin.mean()),
        "n_fe": prob.n_evals + 1,
        "elapsed_s": elapsed,
        "manufacturable": manuf["passes_all"],
        "image": img_bin.astype(np.uint8).tolist(),
    }


def _job(args):
    """Wrapper cho Pool.imap: (cond_idx, budget, seed, target) → kết quả."""
    cond_idx, budget, seed_name, target = args
    res = run_simp_target(target, seed_name, BUDGETS[budget])
    return cond_idx, budget, res


def _pair_err(v12, v21, target) -> float:
    """|Δν12| + |Δν21| (inf nếu FE verify lỗi) - cùng tiêu chí chọn với
    chế độ guarded của refine P1.1e."""
    if v12 is None:
        return float("inf")
    return abs(v12 - target[0]) + abs(v21 - target[1])


def load_targets(path: str, n_conditions: int):
    """Đọc n_conditions target đầu tiên (cùng thứ tự IN100) từ JSON cVAE."""
    with open(path) as f:
        pc = json.load(f)["per_condition"]
    return [(c["target_v12"], c["target_v21"]) for c in pc[:n_conditions]]


def run_benchmark(
    targets,
    seeds=DEFAULT_SEEDS,
    budgets=("full", "matched"),
    workers: int = 10,
    save_images: bool = False,
) -> list:
    """Chạy mọi (target × ngân sách × seed) song song, gom theo condition.

    Returns:
        list per-condition: target + {budget: {"per_seed": [...],
        "best": kết quả seed có sai số verified nhỏ nhất, "n_fe_total"}}.
    """
    jobs = [
        (i, b, s, t)
        for i, t in enumerate(targets)
        for b in budgets
        for s in seeds
    ]
    per_condition = [{"target_v12": t[0], "target_v21": t[1]} for t in targets]
    done = 0
    with Pool(workers) as pool:
        for i, b, res in pool.imap_unordered(_job, jobs):
            if not save_images:
                res.pop("image")
            per_condition[i].setdefault(b, {"per_seed": []})
            per_condition[i][b]["per_seed"].append(res)
            done += 1
            if done % max(1, len(jobs) // 20) == 0:
                print(f"  {done}/{len(jobs)} job xong", flush=True)
    for c in per_condition:
        tgt = (c["target_v12"], c["target_v21"])
        for b in budgets:
            runs = c[b]["per_seed"]
            c[b]["best"] = min(
                runs, key=lambda r: _pair_err(r["v12"], r["v21"], tgt)
            )
            c[b]["n_fe_total"] = int(sum(r["n_fe"] for r in runs))
            c[b]["elapsed_total_s"] = float(sum(r["elapsed_s"] for r in runs))
    return per_condition


def main():
    """CLI: chạy baseline và ghi JSON (per-condition + config)."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    p.add_argument("--targets-json", default=DEFAULT_TARGETS_JSON)
    p.add_argument("--n-conditions", type=int, default=100)
    p.add_argument("--seeds", nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument(
        "--budgets",
        nargs="+",
        default=["full", "matched"],
        choices=list(BUDGETS),
    )
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--save-images", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    targets = load_targets(args.targets_json, args.n_conditions)
    print(
        f"SIMP baseline: {len(targets)} target × {len(args.seeds)} seed × "
        f"{args.budgets}, {args.workers} worker",
        flush=True,
    )
    t0 = time.time()
    per_condition = run_benchmark(
        targets,
        tuple(args.seeds),
        tuple(args.budgets),
        args.workers,
        args.save_images,
    )
    out = {
        "config": {
            "targets_json": os.path.relpath(args.targets_json, REPO_ROOT),
            "n_conditions": len(targets),
            "seeds": args.seeds,
            "budgets": {b: BUDGETS[b] for b in args.budgets},
            "betas": list(BETAS),
            "simp_params": SIMP_PARAMS,
            "fe_params": FE_PARAMS,
            "workers": args.workers,
            "wall_time_s": time.time() - t0,
        },
        "per_condition": per_condition,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"Xong sau {out['config']['wall_time_s']:.0f} s → {args.out}")


if __name__ == "__main__":
    main()
