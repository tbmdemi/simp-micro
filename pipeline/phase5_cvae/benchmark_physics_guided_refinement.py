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

Mở rộng cho plan.md v3 (P1.0) - 3 chế độ, mặc định giữ nguyên hành vi cũ:

  * --targets ν1 ν2 ...: target tự chọn (v12=ν*) thay vì lấy từ test.npz -
    dùng cho quét OOD (P1.1d, dải train v12 ∈ [-1,95; 0]). --target-v21 cố
    định v21 cho mọi target (dị hướng); bỏ trống = v21=ν* (đối xứng). LƯU Ý
    vật lý: vật liệu trực hướng 2D cần ν12·ν21 < 1 (độ cứng xác định dương)
    - target đối xứng ν*≤-1 KHÔNG tồn tại lời giải; mẫu cực trị trong dữ
    liệu train là dị hướng mạnh (v12=-1,95 đi với v21=-0,04).
  * --n-samples N>1: baseline = best-of-N (chọn z có |Δv12|+|Δv21| FE nhỏ
    nhất trong N mẫu), refinement warm-start từ CHÍNH z thắng đó - đo tác
    dụng cộng thêm của refinement lên pipeline best-of-N (P1.1b).
  * --force-periodic: áp manufacturability.force_periodic() trước khi
    binarize + FE, khớp pipeline production (best_of_n_eval.py mặc định bật).

  * --projection-betas 1 4 16 64: refine nhận thức nhị phân hóa (P1.1e) -
    objective áp [force_periodic] -> Heaviside projection -> resize nearest
    giống verify, thay vì tối ưu ảnh liên tục (khai thác vật liệu xám).

Summary luôn kèm chế độ "guarded": mỗi condition giữ refined chỉ khi FE
verify tốt hơn baseline (|Δv12|+|Δv21|), không tốn thêm FE nào vì cả 2 đã
được verify - đây là cách triển khai thực tế (thiết kế cuối luôn được FE
kiểm chứng trước khi dùng).

Mọi chế độ đều xuất CI 95% paired bootstrap cho mức giảm MAE
(bootstrap_ci.bootstrap_paired_mae_reduction) - tiêu chí "thắng" của plan.md
v3 mục 2.4.

Cách chạy:
    python3 pipeline/phase5_cvae/benchmark_physics_guided_refinement.py \\
        --cvae-ckpt outputs/phase5/cvae_kan_realphysics_v2.pt \\
        --n-conditions 24 --steps 30

    # quét OOD, best-of-30, force_periodic
    python3 pipeline/phase5_cvae/benchmark_physics_guided_refinement.py \\
        --cvae-ckpt outputs/phase5/cvae_realphysics.pt \\
        --targets -1.5 -2.0 --target-v21 -0.05 --n-samples 30 \\
        --force-periodic \\
        --out outputs/phase5/refinement_ood.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from adversarial_dataset import load_cvae  # noqa: E402
from bootstrap_ci import bootstrap_paired_mae_reduction  # noqa: E402
from dataset import (  # noqa: E402
    CVAEDataset,
    build_condition_vector,
    condition_flags_from_dim,
)
from manufacturability import (  # noqa: E402
    check_connectivity,
    check_manufacturability,
    count_corner_contacts,
    force_periodic,
)
from realization import filtered_design  # noqa: E402
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


def _verify_v12_v21(
    image_2d: np.ndarray,
    fe_params: dict,
    apply_force_periodic: bool = False,
    design_sigma: float = None,
):
    """(Tùy chọn force_periodic) + binarize (threshold 0.5) + resize về lưới
    FE + giải FE thật.

    Args:
        image_2d: ảnh mật độ (H, W) trong [0,1].
        fe_params: dict FE params (xem verify_fe.FE_PARAMS).
        apply_force_periodic: True -> ép biên tuần hoàn trước khi binarize,
            đúng thứ tự best_of_n_eval.py (force_periodic trên ảnh liên tục).
        design_sigma: N2-E1′ - đặt thì thiết kế = ảnh đã lọc
            (``_binary_design``) thay cho ngưỡng + resize nearest.

    Returns:
        (v12, v21) hoặc None nếu FE-solve lỗi (Q suy biến/không hữu hạn -
        cùng hành vi "bỏ qua mẫu lỗi" như best_of_n_eval.py).
    """
    img_bin = _binary_design(
        image_2d, apply_force_periodic, design_sigma, fe_params
    ).astype(np.float32)
    img_fe = resize_to_fe_grid(img_bin, fe_params["nely"], fe_params["nelx"])
    try:
        v12, v21, _ = evaluate_density_field(img_fe, fe_params)
    except Exception:
        return None
    return float(v12), float(v21)


def _binary_design(
    image_2d: np.ndarray,
    apply_force_periodic: bool = False,
    design_sigma: float = None,
    fe_params: dict = FE_PARAMS,
    eta: float = 0.5,
):
    """Ảnh nhị phân của thiết kế - đúng ảnh verify FE và kiểm chế tạo.

    Mặc định: ảnh 64×64 (tùy chọn force_periodic → ngưỡng 0,5), cùng ảnh
    best_of_n_eval.py dùng để chấm manufacturability. Với ``design_sigma``
    (N2-E1′): ảnh trên lưới FE từ ``realization.filtered_design`` (lọc Gauss
    → lấy mẫu → ngưỡng ``eta``), để FE, chế tạo và hình vẽ cùng 1 ảnh.

    Args:
        image_2d: ảnh mật độ (H, W) trong [0,1].
        apply_force_periodic: ép biên tuần hoàn trước khi ngưỡng.
        design_sigma: độ mượt bộ lọc (phần tử FE); None = hành vi cũ.
        fe_params: lưới FE đích khi dùng ``design_sigma``.
        eta: ngưỡng (0,5 gốc; khác 0,5 = bản co/giãn cho guard robust).

    Returns:
        np.ndarray uint8 0/1 - (H, W) hoặc (nely, nelx).
    """
    if design_sigma is not None:
        t = torch.as_tensor(image_2d, dtype=torch.float32)[None, None]
        d = filtered_design(
            t, None, fe_params["nely"], fe_params["nelx"], design_sigma, eta
        )
        return d[0, 0].numpy().astype(np.uint8)
    if apply_force_periodic:
        image_2d = force_periodic(image_2d)
    return (image_2d > eta).astype(np.uint8)


def _robust_score(
    image_2d: np.ndarray,
    nominal_pred,
    target_v12: float,
    target_v21: float,
    design_sigma: float,
    etas: list,
    fe_params: dict = FE_PARAMS,
) -> float:
    """Loss robust đo FE ở β = ∞, cùng dạng objective E1′ - dùng làm guard
    (tối ưu gì thì chọn bằng nấy): 0,5·MSE(gốc) + 0,5·trung bình MSE(co/giãn).

    Args:
        image_2d: ảnh decoder (H, W).
        nominal_pred: (v12, v21) của bản gốc đã tính (tránh giải lại).
        target_v12, target_v21: mục tiêu.
        design_sigma: độ mượt bộ lọc thiết kế.
        etas: ngưỡng bản co/giãn.
        fe_params: lưới FE.

    Returns:
        Giá trị loss; ``inf`` nếu có FE lỗi.
    """

    def mse(pred):
        return 0.5 * (
            (pred[0] - target_v12) ** 2 + (pred[1] - target_v21) ** 2
        )

    if nominal_pred is None:
        return float("inf")
    pert = []
    for eta in etas:
        d = _binary_design(image_2d, False, design_sigma, fe_params, eta)
        try:
            v12, v21, _ = evaluate_density_field(
                d.astype(np.float32), fe_params
            )
        except Exception:
            return float("inf")
        pert.append(mse((v12, v21)))
    return 0.5 * mse(nominal_pred) + 0.5 * float(np.mean(pert))


def _r2(preds, targets) -> float:
    """R2 = 1 - SS_res/SS_tot. NaN nếu targets không có phương sai (SS_tot=0)."""
    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    ss_res = np.sum((preds - targets) ** 2)
    return float(1.0 - ss_res / ss_tot)


def _pair_err(pred, target_v12: float, target_v21: float):
    """Sai số tuyệt đối (|Δv12|, |Δv21|) của 1 dự đoán FE; None nếu FE lỗi."""
    if pred is None:
        return None
    return abs(pred[0] - target_v12), abs(pred[1] - target_v21)


def _mean_or_nan(values) -> float:
    """Trung bình, NaN nếu rỗng (tránh RuntimeWarning của np.mean([]))."""
    return float(np.mean(values)) if len(values) else float("nan")


def _summarize(per_condition: list, n_boot: int) -> dict:
    """Tổng hợp per_condition thành summary.

    Giữ nguyên các khóa cũ (r2_v12_*, mean_abs_err_v12_*, tính riêng trên
    từng tập valid như bản 2026-09-23) để số mới so trực tiếp được với số
    cũ; thêm v21, sai số tương đối và CI paired (chỉ trên condition mà CẢ
    baseline lẫn refined đều FE-valid - paired bootstrap cần đủ cặp).

    Args:
        per_condition: list dict do run_benchmark() sinh ra.
        n_boot: số lần resample bootstrap.

    Returns:
        dict summary.
    """
    vb = [c for c in per_condition if c["baseline_v12"] is not None]
    vr = [c for c in per_condition if c["refined_v12"] is not None]
    paired = [
        c
        for c in per_condition
        if c["baseline_v12"] is not None and c["refined_v12"] is not None
    ]

    def col(rows, key):
        return [c[key] for c in rows]

    def abs_err(rows, which, comp):
        return [abs(c[f"{which}_{comp}"] - c[f"target_{comp}"]) for c in rows]

    def rel_err(rows, which, comps=("v12", "v21")):
        # Sai số tương đối trung bình các thành phần `comps` - thước đo của
        # DoN OOD (≤ 8%, plan.md v3 P1.1d, tính trên v12 vì target OOD dị
        # hướng có v21 ~ -0,05: tỉ lệ trên mẫu số nhỏ vô nghĩa); bỏ thành
        # phần có target ~0 (không định nghĩa được tỉ lệ).
        out = []
        for c in rows:
            parts = [
                abs(c[f"{which}_{k}"] - c[f"target_{k}"])
                / abs(c[f"target_{k}"])
                for k in comps
                if abs(c[f"target_{k}"]) > 1e-6
            ]
            if parts:
                out.append(float(np.mean(parts)))
        return out

    # Guarded: lấy refined chỉ khi verify tốt hơn baseline (cờ tính sẵn
    # trong run_benchmark) - so paired với baseline trên cùng tập condition.
    guarded = [
        {
            **c,
            "guarded_v12": (
                c["refined_v12"] if c["guarded_accept"] else c["baseline_v12"]
            ),
            "guarded_v21": (
                c["refined_v21"] if c["guarded_accept"] else c["baseline_v21"]
            ),
        }
        for c in paired
    ]

    both_b = [
        0.5 * (a + b)
        for a, b in zip(
            abs_err(paired, "baseline", "v12"),
            abs_err(paired, "baseline", "v21"),
        )
    ]
    both_r = [
        0.5 * (a + b)
        for a, b in zip(
            abs_err(paired, "refined", "v12"),
            abs_err(paired, "refined", "v21"),
        )
    ]
    return {
        "n_conditions": len(per_condition),
        "n_valid_baseline": len(vb),
        "n_valid_refined": len(vr),
        "n_paired": len(paired),
        "r2_v12_baseline": _r2(col(vb, "baseline_v12"), col(vb, "target_v12")),
        "r2_v12_refined": _r2(col(vr, "refined_v12"), col(vr, "target_v12")),
        "r2_v21_baseline": _r2(col(vb, "baseline_v21"), col(vb, "target_v21")),
        "r2_v21_refined": _r2(col(vr, "refined_v21"), col(vr, "target_v21")),
        "mean_abs_err_v12_baseline": _mean_or_nan(
            abs_err(vb, "baseline", "v12")
        ),
        "mean_abs_err_v12_refined": _mean_or_nan(
            abs_err(vr, "refined", "v12")
        ),
        "mean_abs_err_v21_baseline": _mean_or_nan(
            abs_err(vb, "baseline", "v21")
        ),
        "mean_abs_err_v21_refined": _mean_or_nan(
            abs_err(vr, "refined", "v21")
        ),
        "mean_rel_err_baseline": _mean_or_nan(rel_err(vb, "baseline")),
        "mean_rel_err_refined": _mean_or_nan(rel_err(vr, "refined")),
        "mean_rel_err_v12_baseline": _mean_or_nan(
            rel_err(vb, "baseline", ("v12",))
        ),
        "mean_rel_err_v12_refined": _mean_or_nan(
            rel_err(vr, "refined", ("v12",))
        ),
        "paired_v12": bootstrap_paired_mae_reduction(
            abs_err(paired, "baseline", "v12"),
            abs_err(paired, "refined", "v12"),
            n_boot=n_boot,
        ),
        "paired_v12_v21": bootstrap_paired_mae_reduction(
            both_b, both_r, n_boot=n_boot
        ),
        "r2_v12_guarded": _r2(
            col(guarded, "guarded_v12"), col(guarded, "target_v12")
        ),
        "mean_rel_err_v12_guarded": _mean_or_nan(
            rel_err(guarded, "guarded", ("v12",))
        ),
        "frac_guarded_accept": _mean_or_nan(
            [float(c["guarded_accept"]) for c in paired]
        ),
        "paired_v12_guarded": bootstrap_paired_mae_reduction(
            abs_err(guarded, "baseline", "v12"),
            abs_err(guarded, "guarded", "v12"),
            n_boot=n_boot,
        ),
        "n_fe_calls_baseline": int(
            sum(c["n_fe_calls_baseline"] for c in per_condition)
        ),
        "n_fe_calls_refine": int(
            sum(c["n_fe_calls_refine"] for c in per_condition)
        ),
    }


def run_benchmark(
    cvae_ckpt_path: str,
    n_conditions: int = 24,
    steps: int = 30,
    seed: int = 123,
    data_dir: str = None,
    device: str = "cpu",
    learning_rate: float = 0.5,
    real_physics_subsample: int = None,
    targets: list = None,
    target_v21: float = None,
    n_samples: int = 1,
    apply_force_periodic: bool = False,
    n_boot: int = 10000,
    projection_betas: list = None,
    save_images: bool = False,
    corner_weight: float = 0.0,
    thin_weight: float = 0.0,
    realization_shifts: list = None,
    realization_sigma: float = 0.5,
    fe_upsample: int = 1,
    n_workers: int = 0,
    fe_upsample_last_only: bool = False,
    robust_etas: list = None,
    robust_sigma: float = 1.0,
    design_filter_sigma: float = None,
) -> dict:
    """Chạy benchmark baseline-vs-refined.

    Condition lấy từ `targets` (nếu có) hoặc `n_conditions` mẫu ngẫu nhiên
    của test.npz (cùng seed=123 với best_of_n_eval.py/self_play.verify_round).

    Args:
        cvae_ckpt_path: đường dẫn checkpoint cVAE đã train (frozen).
        n_conditions: số condition target lấy ngẫu nhiên từ test.npz (bỏ qua
            khi có `targets`).
        steps: số bước L-BFGS tối ưu z cho mỗi condition (guidance_source=
            "real_physics").
        seed: seed chọn condition VÀ khởi tạo z ngẫu nhiên (tái lập được).
        data_dir: thư mục chứa test.npz. None = outputs/phase3/.
        device: "cpu" hoặc "cuda".
        learning_rate: bước L-BFGS (tandem_inverse_design_lbfgs).
        real_physics_subsample: xem real_physics_loss - bỏ qua có ý nghĩa vì
            mỗi lần refine chỉ có batch=1 mẫu.
        targets: list ν* - mỗi phần tử thành 1 condition v12=ν* (các
            trường optional để mask=0). None = dùng test.npz.
        target_v21: v21 cố định cho mọi target (dị hướng). None = v21=ν*
            (đối xứng) - chỉ khả thi vật lý khi ν*² < 1.
        n_samples: số z ngẫu nhiên mỗi condition. 1 = hành vi cũ (tái lập
            đúng số 2026-09-23: cùng thứ tự rút RNG). >1 = best-of-N theo
            |Δv12|+|Δv21| FE, refine từ z thắng.
        apply_force_periodic: áp force_periodic() trước FE verify (cả
            baseline lẫn refined, để so sánh công bằng).
        n_boot: số lần resample cho CI paired.
        projection_betas: None = refine trên ảnh liên tục (hành vi cũ). Có
            giá trị = refine nhận thức nhị phân hóa, xem
            tandem_lbfgs.tandem_inverse_design_lbfgs(projection_betas=...);
            force_periodic trong objective bật theo apply_force_periodic.
        save_images: lưu ảnh nhị phân 64×64 của thiết kế baseline và refined
            vào per_condition (cho hình + đo chế tạo lại, P1.6c). Cờ
            manufacturable của cả 2 luôn được ghi (rẻ, không cần FE).
        corner_weight, thin_weight: hệ số phạt chế tạo trong refine (K1,
            xem tandem_lbfgs). 0 = hành vi cũ.
        realization_shifts, realization_sigma, fe_upsample: objective N1 và
            đối chứng lưới mịn, xem tandem_lbfgs. None/1 = hành vi cũ.
        n_workers: số tiến trình FE song song trong refine (0 = tuần tự).
        fe_upsample_last_only, robust_etas, robust_sigma: N2-E2 (lưới mịn chỉ
            ở mức β cuối) và N2-E1 (robust formulation co/giãn), xem
            tandem_lbfgs. Mặc định = hành vi cũ.
        design_filter_sigma: N2-E1′ - thiết kế = ảnh đã lọc trên lưới FE
            cho MỌI bước (chọn best-of-N, refine, verify, chế tạo, ảnh lưu);
            nếu có thêm ``robust_etas`` thì guard theo loss robust.

    Returns:
        dict {"summary": {...}, "per_condition": [...], "config": {...}}.
    """
    if n_samples < 1:
        raise ValueError("n_samples phải >= 1")
    # manual_seed TRƯỚC load_cvae: khởi tạo module tiêu thụ RNG, giữ đúng
    # thứ tự của bản gốc để n_samples=1 tái lập bit-for-bit z0 cũ.
    torch.manual_seed(seed)
    ckpt_meta = torch.load(
        cvae_ckpt_path, map_location="cpu", weights_only=False
    )
    condition_dim = ckpt_meta.get("condition_dim", 2)
    latent_dim = ckpt_meta["latent_dim"]
    del ckpt_meta
    extended_condition, include_nu0 = condition_flags_from_dim(condition_dim)

    model = load_cvae(cvae_ckpt_path, device)

    if targets:
        conditions = [
            build_condition_vector(
                float(t),
                float(t) if target_v21 is None else float(target_v21),
                condition_dim,
            )
            for t in targets
        ]
    else:
        test_ds = CVAEDataset(
            os.path.join(data_dir or PHASE3_DIR, "test.npz"),
            extended_condition=extended_condition,
            include_nu0=include_nu0,
        )
        rng = np.random.default_rng(seed)
        idxs = rng.choice(len(test_ds), size=n_conditions, replace=False)
        conditions = [test_ds[i][1].numpy() for i in idxs]

    per_condition = []
    t_start = time.time()
    for ci, cond in enumerate(conditions):
        cond_t = torch.tensor(cond, dtype=torch.float32, device=device)
        target_v12, target_v21 = float(cond[0]), float(cond[1])

        # randn(N, d) với N=1 rút đúng cùng số như randn(1, d) bản cũ.
        z_all = torch.randn(n_samples, latent_dim, device=device)
        with torch.no_grad():
            images = model.decoder(
                z_all, cond_t.unsqueeze(0).expand(n_samples, -1)
            )
        preds = [
            _verify_v12_v21(
                images[i].squeeze().cpu().numpy(),
                FE_PARAMS,
                apply_force_periodic,
                design_sigma=design_filter_sigma,
            )
            for i in range(n_samples)
        ]
        # Chọn theo tổng sai số 2 thành phần: real_physics_loss tối ưu cả
        # v12 lẫn v21, chọn chỉ theo v12 sẽ lệch mục tiêu với refinement.
        errs = [
            (sum(e), i)
            for i, e in enumerate(
                _pair_err(p, target_v12, target_v21) for p in preds
            )
            if e is not None
        ]
        best_i = min(errs)[1] if errs else 0
        baseline_pred = preds[best_i]
        z0 = z_all[best_i : best_i + 1]

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
            projection_betas=projection_betas,
            periodic=apply_force_periodic,
            corner_weight=corner_weight,
            thin_weight=thin_weight,
            realization_shifts=realization_shifts,
            realization_sigma=realization_sigma,
            fe_upsample=fe_upsample,
            n_workers=n_workers,
            fe_upsample_last_only=fe_upsample_last_only,
            robust_etas=robust_etas,
            robust_sigma=robust_sigma,
            design_filter_sigma=design_filter_sigma,
        )
        refined_img = result["image"].squeeze().cpu().numpy()
        refined_pred = _verify_v12_v21(
            refined_img,
            FE_PARAMS,
            apply_force_periodic,
            design_sigma=design_filter_sigma,
        )
        # Chế tạo được đo trên đúng ảnh nhị phân của 2 thiết kế được so
        # sánh - refine có thể đổi tô-pô (đứt nét mảnh) dù ν khớp hơn.
        baseline_img = images[best_i].squeeze().cpu().numpy()
        baseline_bin = _binary_design(
            baseline_img, apply_force_periodic, design_filter_sigma
        )
        refined_bin = _binary_design(
            refined_img, apply_force_periodic, design_filter_sigma
        )
        robust_guard = design_filter_sigma is not None and bool(robust_etas)
        if robust_guard:
            # E1′: guard theo chính loss robust (4 FE thêm: co/giãn × 2).
            score_base = _robust_score(
                baseline_img,
                baseline_pred,
                target_v12,
                target_v21,
                design_filter_sigma,
                robust_etas,
            )
            score_ref = _robust_score(
                refined_img,
                refined_pred,
                target_v12,
                target_v21,
                design_filter_sigma,
                robust_etas,
            )
            accept = score_ref < score_base
        else:
            accept = (
                refined_pred is not None
                and baseline_pred is not None
                and sum(_pair_err(refined_pred, target_v12, target_v21))
                < sum(_pair_err(baseline_pred, target_v12, target_v21))
            )
        manuf = {
            "baseline_manufacturable": check_manufacturability(baseline_bin)[
                "passes_all"
            ],
            "refined_manufacturable": check_manufacturability(refined_bin)[
                "passes_all"
            ],
        }
        # Định nghĩa K1 (plan.md): liên thông 4 hướng + nét tối thiểu, và số
        # điểm chạm góc tuần hoàn (bản lề 1 nút).
        for key, img in (("baseline", baseline_bin), ("refined", refined_bin)):
            manuf[f"{key}_manuf4"] = check_connectivity(img, connectivity=4)[
                "manufacturable"
            ]
            manuf[f"{key}_corners"] = count_corner_contacts(img)
        if save_images:
            manuf["baseline_image"] = baseline_bin.tolist()
            manuf["refined_image"] = refined_bin.tolist()

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
                "baseline_sample_index": int(best_i),
                "guarded_accept": bool(accept),
                # Chi phí FE: N lần verify baseline; refine = mỗi closure
                # L-BFGS 1 FE (forward+adjoint) + 1 FE dự đoán cuối trong
                # tandem + 1 FE verify độc lập.
                "n_fe_calls_baseline": n_samples,
                "n_fe_calls_refine": len(result["history"])
                + 2
                + (2 * len(robust_etas) if robust_guard else 0),
                "refinement_loss_history": result["history"],
                **manuf,
            }
        )
        _print_progress(ci, len(conditions), per_condition[-1], t_start)

    summary = _summarize(per_condition, n_boot)
    for key in ("baseline", "refined"):
        summary[f"frac_manufacturable_{key}"] = float(
            np.mean([c[f"{key}_manufacturable"] for c in per_condition])
        )
    # Thiết kế cuối guarded (cái được báo trong bài) theo định nghĩa K1.
    final = [
        "refined" if c["guarded_accept"] else "baseline" for c in per_condition
    ]
    summary["frac_manuf4_guarded"] = float(
        np.mean([c[f"{k}_manuf4"] for c, k in zip(per_condition, final)])
    )
    summary["frac_corner_guarded"] = float(
        np.mean([c[f"{k}_corners"] > 0 for c, k in zip(per_condition, final)])
    )
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
            "targets": list(targets) if targets else None,
            "target_v21": target_v21,
            "n_samples": n_samples,
            "apply_force_periodic": apply_force_periodic,
            "n_boot": n_boot,
            "projection_betas": (
                list(projection_betas) if projection_betas else None
            ),
            "corner_weight": corner_weight,
            "thin_weight": thin_weight,
            "realization_shifts": realization_shifts,
            "realization_sigma": realization_sigma,
            "fe_upsample": fe_upsample,
            "fe_upsample_last_only": fe_upsample_last_only,
            "robust_etas": list(robust_etas) if robust_etas else None,
            "robust_sigma": robust_sigma,
            "design_filter_sigma": design_filter_sigma,
        },
    }


def _print_progress(ci: int, n: int, c: dict, t_start: float) -> None:
    """In 1 dòng tiến độ sau mỗi condition (flush ngay để theo dõi log).

    Args:
        ci: chỉ số condition vừa xong (0-based).
        n: tổng số condition.
        c: bản ghi per_condition của condition đó.
        t_start: time.time() lúc bắt đầu vòng lặp (để ước ETA).
    """

    def err(key):
        if c[f"{key}_v12"] is None:
            return float("nan")
        return abs(c[f"{key}_v12"] - c["target_v12"]) + abs(
            c[f"{key}_v21"] - c["target_v21"]
        )

    el = time.time() - t_start
    eta = el / (ci + 1) * (n - ci - 1)
    fin = "refined" if c["guarded_accept"] else "baseline"
    print(
        f"[{ci + 1:3d}/{n}] target=({c['target_v12']:+.3f},"
        f"{c['target_v21']:+.3f})  err_cặp {err('baseline'):.4f}->"
        f"{err('refined'):.4f} {'nhận' if c['guarded_accept'] else 'giữ'}"
        f"  chế_tạo4 {int(c['baseline_manuf4'])}->{int(c[f'{fin}_manuf4'])}"
        f"  góc {c['baseline_corners']}->{c[f'{fin}_corners']}"
        f"  | {el / 60:.1f} phút, ETA {eta / 60:.1f} phút",
        flush=True,
    )


def main():
    """CLI: chạy benchmark và lưu JSON (xem docstring module)."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
        "--targets",
        type=float,
        nargs="+",
        default=None,
        help="ν* tự chọn cho v12, thay cho test.npz.",
    )
    parser.add_argument(
        "--target-v21",
        type=float,
        default=None,
        help="v21 cố định cho mọi --targets (bỏ trống = v21=ν*).",
    )
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--force-periodic", action="store_true")
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Lưu ảnh nhị phân baseline/refined vào JSON (P1.6c).",
    )
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument(
        "--projection-betas",
        type=float,
        nargs="+",
        default=None,
        help="β Heaviside tăng dần cho refine nhận thức nhị phân hóa.",
    )
    parser.add_argument(
        "--corner-weight",
        type=float,
        default=0.0,
        help="Hệ số phạt chạm góc pixel trong refine (K1).",
    )
    parser.add_argument(
        "--thin-weight",
        type=float,
        default=0.0,
        help="Hệ số phạt nét mảnh < 3 px trong refine (K1).",
    )
    parser.add_argument(
        "--realization-shifts",
        type=float,
        nargs="+",
        default=None,
        help="N1: cặp dy dx (phần tử FE) nối tiếp, vd 0 0 0 0.5 0.5 0.",
    )
    parser.add_argument("--realization-sigma", type=float, default=0.5)
    parser.add_argument(
        "--fe-upsample",
        type=int,
        default=1,
        help="Đối chứng N1: refine với FE trên lưới mịn hơn k lần.",
    )
    parser.add_argument("--n-workers", type=int, default=0)
    parser.add_argument(
        "--fe-upsample-last-only",
        action="store_true",
        help="N2-E2: chỉ dùng lưới mịn (--fe-upsample) ở mức β cuối.",
    )
    parser.add_argument(
        "--robust-etas",
        type=float,
        nargs="+",
        default=None,
        help="N2-E1: ngưỡng chiếu bản giãn/co, vd 0.25 0.75.",
    )
    parser.add_argument("--robust-sigma", type=float, default=1.0)
    parser.add_argument(
        "--design-filter-sigma",
        type=float,
        default=None,
        help="N2-E1′: thiết kế = ảnh lọc Gauss σ (phần tử FE) trên lưới FE.",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(
            PHASE5_DIR, "physics_guided_refinement_benchmark.json"
        ),
    )
    args = parser.parse_args()

    shifts = args.realization_shifts
    if shifts is not None:
        if len(shifts) % 2:
            parser.error("--realization-shifts cần số giá trị chẵn (dy dx)")
        shifts = [list(shifts[i : i + 2]) for i in range(0, len(shifts), 2)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = run_benchmark(
        args.cvae_ckpt,
        n_conditions=args.n_conditions,
        steps=args.steps,
        seed=args.seed,
        data_dir=args.data_dir,
        device=device,
        learning_rate=args.learning_rate,
        real_physics_subsample=args.real_physics_subsample,
        targets=args.targets,
        target_v21=args.target_v21,
        n_samples=args.n_samples,
        apply_force_periodic=args.force_periodic,
        n_boot=args.n_boot,
        projection_betas=args.projection_betas,
        save_images=args.save_images,
        corner_weight=args.corner_weight,
        thin_weight=args.thin_weight,
        realization_shifts=shifts,
        realization_sigma=args.realization_sigma,
        fe_upsample=args.fe_upsample,
        n_workers=args.n_workers,
        fe_upsample_last_only=args.fe_upsample_last_only,
        robust_etas=args.robust_etas,
        robust_sigma=args.robust_sigma,
        design_filter_sigma=args.design_filter_sigma,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    s = result["summary"]
    p = s["paired_v12"]
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
    print(
        f"Giảm MAE(v12) paired: {p['rel_reduction']:.1%} "
        f"[CI95 {p['rel_reduction_ci95_lo']:.1%}, "
        f"{p['rel_reduction_ci95_hi']:.1%}]"
    )
    print(
        f"Sai số tương đối: baseline={s['mean_rel_err_baseline']:.1%}  "
        f"refined={s['mean_rel_err_refined']:.1%}  "
        f"(v12: {s['mean_rel_err_v12_baseline']:.1%} -> "
        f"{s['mean_rel_err_v12_refined']:.1%})"
    )
    print(
        f"FE calls: baseline={s['n_fe_calls_baseline']}  "
        f"refine={s['n_fe_calls_refine']}"
    )
    g = s["paired_v12_guarded"]
    print(
        f"Guarded (nhận refine {s['frac_guarded_accept']:.0%}): "
        f"R2(v12)={s['r2_v12_guarded']:.4f}  giảm MAE(v12) "
        f"{g['rel_reduction']:.1%} [CI95 {g['rel_reduction_ci95_lo']:.1%}, "
        f"{g['rel_reduction_ci95_hi']:.1%}]"
    )
    print(
        f"Chế tạo (K1: 4 hướng + nét tối thiểu, guarded): "
        f"{s['frac_manuf4_guarded']:.2f}  có chạm góc: "
        f"{s['frac_corner_guarded']:.2f}"
    )
    print(f"Đã lưu: {args.out}")


if __name__ == "__main__":
    main()
