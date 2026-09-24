"""
Phase 5 - benchmark_physics_guided_refinement.py
============================================================
So sánh 2 chiến lược lấy latent z cho 1 cVAE đã train sẵn (frozen), trên
CÙNG condition test set (seed=123, test.npz - khớp quy ước
self_play.verify_round/best_of_n_eval.py để so sánh apples-to-apples):

  (a) baseline: z ~ N(0, I) ngẫu nhiên (đúng những gì model.generate() làm
      hôm nay) -> binarize -> FE thật.
  (b) physics-guided refinement (plan.md Phần B): CÙNG z khởi tạo như (a),
      tối ưu z bằng tandem_lbfgs.tandem_inverse_design_lbfgs(
      guidance_source="real_physics") - gradient GIẢI TÍCH từ chính bộ giải
      FE thật (real_physics.py), KHÔNG qua surrogate CNN xấp xỉ (đã biết là
      "exploitable" - xem losses.py cảnh báo đầu file) - rồi mới binarize +
      FE thật để đánh giá (verify bằng FE THẬT độc lập với loss dùng để tối
      ưu, không tự chấm điểm bằng chính hàm mình vừa minimize).

Đây là benchmark 1-mẫu-1-lần-sinh (KHÔNG best-of-N) để cô lập đúng tác dụng
của refinement, không trộn với hiệu ứng lọc-nhiều-mẫu (đã đo riêng ở
best_of_n_eval.py) - là số liệu quyết định có nên đầu tư tiếp vào latent
diffusion prior (Giai đoạn 2, plan.md) hay không.

Cách chạy:
    python3 pipeline/phase5_cvae/benchmark_physics_guided_refinement.py \\
        --cvae-ckpt outputs/phase5/cvae_kan_realphysics_v2.pt \\
        --n-conditions 24 --steps 30
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from adversarial_dataset import load_cvae  # noqa: E402
from dataset import CVAEDataset, condition_flags_from_dim  # noqa: E402
from tandem_lbfgs import tandem_inverse_design_lbfgs  # noqa: E402
from verify_fe import (  # noqa: E402
    FE_PARAMS,
    evaluate_density_field,
    resize_to_fe_grid,
)

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
PHASE3_DIR = os.path.join(REPO_ROOT, "outputs", "phase3")
PHASE5_DIR = os.path.join(REPO_ROOT, "outputs", "phase5")


def _verify_v12_v21(image_2d: np.ndarray, fe_params: dict):
    """Binarize (threshold 0.5) + resize về lưới FE + giải FE thật.

    Args:
        image_2d: ảnh mật độ (H, W) trong [0,1].
        fe_params: dict FE params (xem verify_fe.FE_PARAMS).

    Returns:
        (v12, v21) hoặc None nếu FE-solve lỗi (Q suy biến/không hữu hạn -
        cùng hành vi "bỏ qua mẫu lỗi" như best_of_n_eval.py).
    """
    img_bin = (image_2d > 0.5).astype(np.float32)
    img_fe = resize_to_fe_grid(img_bin, fe_params["nely"], fe_params["nelx"])
    try:
        v12, v21, _ = evaluate_density_field(img_fe, fe_params)
    except Exception:
        return None
    return float(v12), float(v21)


def _r2(preds, targets) -> float:
    """R2 = 1 - SS_res/SS_tot. NaN nếu targets không có phương sai (SS_tot=0)."""
    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    ss_res = np.sum((preds - targets) ** 2)
    return float(1.0 - ss_res / ss_tot)


def run_benchmark(
    cvae_ckpt_path: str,
    n_conditions: int = 24,
    steps: int = 30,
    seed: int = 123,
    data_dir: str = None,
    device: str = "cpu",
    learning_rate: float = 0.5,
    real_physics_subsample: int = None,
) -> dict:
    """Chạy benchmark baseline-vs-refined trên `n_conditions` target lấy từ
    test.npz (cùng seed=123 với best_of_n_eval.py/self_play.verify_round).

    Args:
        cvae_ckpt_path: đường dẫn checkpoint cVAE đã train (frozen).
        n_conditions: số condition target lấy ngẫu nhiên từ test.npz.
        steps: số bước L-BFGS tối ưu z cho mỗi condition (guidance_source=
            "real_physics").
        seed: seed chọn condition VÀ khởi tạo z ngẫu nhiên (tái lập được).
        data_dir: thư mục chứa test.npz. None = outputs/phase3/.
        device: "cpu" hoặc "cuda".
        learning_rate: bước L-BFGS (tandem_inverse_design_lbfgs).
        real_physics_subsample: xem real_physics_loss - bỏ qua có ý nghĩa vì
            mỗi lần refine chỉ có batch=1 mẫu.

    Returns:
        dict {"summary": {...}, "per_condition": [...], "config": {...}}.
    """
    torch.manual_seed(seed)
    ckpt_meta = torch.load(
        cvae_ckpt_path, map_location="cpu", weights_only=False
    )
    condition_dim = ckpt_meta.get("condition_dim", 2)
    latent_dim = ckpt_meta["latent_dim"]
    del ckpt_meta
    extended_condition, include_nu0 = condition_flags_from_dim(condition_dim)

    model = load_cvae(cvae_ckpt_path, device)

    test_ds = CVAEDataset(
        os.path.join(data_dir or PHASE3_DIR, "test.npz"),
        extended_condition=extended_condition,
        include_nu0=include_nu0,
    )
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(test_ds), size=n_conditions, replace=False)
    conditions = [test_ds[i][1].numpy() for i in idxs]

    per_condition = []
    for cond in conditions:
        cond_t = torch.tensor(cond, dtype=torch.float32, device=device)
        target_v12, target_v21 = float(cond[0]), float(cond[1])

        z0 = torch.randn(1, latent_dim, device=device)
        with torch.no_grad():
            image_baseline = model.decoder(z0, cond_t.unsqueeze(0))
        baseline_pred = _verify_v12_v21(
            image_baseline.squeeze().cpu().numpy(), FE_PARAMS
        )

        result = tandem_inverse_design_lbfgs(
            target_poisson=cond_t[:2],
            generator_model=model,
            surrogate_model=None,
            steps=steps,
            condition=cond_t,
            initial_z=z0,
            learning_rate=learning_rate,
            guidance_source="real_physics",
            fe_params=FE_PARAMS,
            subsample=real_physics_subsample,
        )
        refined_pred = _verify_v12_v21(
            result["image"].squeeze().cpu().numpy(), FE_PARAMS
        )

        per_condition.append(
            {
                "target_v12": target_v12,
                "target_v21": target_v21,
                "baseline_v12": (
                    None if baseline_pred is None else baseline_pred[0]
                ),
                "baseline_v21": (
                    None if baseline_pred is None else baseline_pred[1]
                ),
                "refined_v12": (
                    None if refined_pred is None else refined_pred[0]
                ),
                "refined_v21": (
                    None if refined_pred is None else refined_pred[1]
                ),
                "refinement_loss_history": result["history"],
            }
        )

    valid_baseline = [
        c for c in per_condition if c["baseline_v12"] is not None
    ]
    valid_refined = [c for c in per_condition if c["refined_v12"] is not None]

    summary = {
        "n_conditions": len(conditions),
        "n_valid_baseline": len(valid_baseline),
        "n_valid_refined": len(valid_refined),
        "r2_v12_baseline": _r2(
            [c["baseline_v12"] for c in valid_baseline],
            [c["target_v12"] for c in valid_baseline],
        ),
        "r2_v12_refined": _r2(
            [c["refined_v12"] for c in valid_refined],
            [c["target_v12"] for c in valid_refined],
        ),
        "mean_abs_err_v12_baseline": (
            float(
                np.mean(
                    [
                        abs(c["baseline_v12"] - c["target_v12"])
                        for c in valid_baseline
                    ]
                )
            )
            if valid_baseline
            else float("nan")
        ),
        "mean_abs_err_v12_refined": (
            float(
                np.mean(
                    [
                        abs(c["refined_v12"] - c["target_v12"])
                        for c in valid_refined
                    ]
                )
            )
            if valid_refined
            else float("nan")
        ),
    }
    return {
        "summary": summary,
        "per_condition": per_condition,
        "config": {
            "cvae_ckpt_path": cvae_ckpt_path,
            "n_conditions": n_conditions,
            "steps": steps,
            "seed": seed,
            "learning_rate": learning_rate,
            "real_physics_subsample": real_physics_subsample,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cvae-ckpt",
        default=os.path.join(PHASE5_DIR, "cvae_kan_realphysics_v2.pt"),
    )
    parser.add_argument("--n-conditions", type=int, default=24)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--real-physics-subsample", type=int, default=None)
    parser.add_argument(
        "--out",
        default=os.path.join(
            PHASE5_DIR, "physics_guided_refinement_benchmark.json"
        ),
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = run_benchmark(
        args.cvae_ckpt,
        args.n_conditions,
        args.steps,
        args.seed,
        args.data_dir,
        device,
        args.learning_rate,
        args.real_physics_subsample,
    )
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    s = result["summary"]
    print(
        f"N condition: {s['n_conditions']} "
        f"(valid FE: baseline={s['n_valid_baseline']}, "
        f"refined={s['n_valid_refined']})"
    )
    print(
        f"R2(v12) baseline={s['r2_v12_baseline']:.4f}  "
        f"refined={s['r2_v12_refined']:.4f}"
    )
    print(
        f"MAE(v12) baseline={s['mean_abs_err_v12_baseline']:.4f}  "
        f"refined={s['mean_abs_err_v12_refined']:.4f}"
    )
    print(f"Đã lưu: {args.out}")


if __name__ == "__main__":
    main()
