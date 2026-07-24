"""
So sánh cvae_gamma20.pt (baseline, shuffle đều) vs cvae_gamma20_weighted.pt
(--weighted-sampling, xem train.py/dataset.compute_v12_sample_weights) -
yêu cầu advisor 2026-07-24: "train lại vae [với weighted sampling theo phổ
auxetic] xem thế nào".

Không dùng property_accuracy() qua surrogate (README cảnh báo không đáng
tin - surrogate exploitation, xem evaluate.py) mà dùng FE THẬT
(verify_fe.py, cùng cơ chế train.py::real_fe_r2) trên các condition target
CHỌN TRƯỚC trải đều qua toàn phổ v12 - kể cả các bin RẤT THƯA mẫu (auxetic
mạnh <-0.6, hoặc gần 0/dương) mà val.npz tự nhiên gần như không có mẫu nào
rơi vào (n=0 khi lọc theo trọng số >1.5x mean, xem phân tích trước khi viết
script này) - nên không thể đánh giá bằng cách lọc val set, phải tự đặt
condition mục tiêu.

condition = (v12, v21); vì thiếu 1 giá trị v21 "thật" tương ứng cho mỗi
v12 tuỳ ý, dùng xấp xỉ v21=v12 (dữ liệu thật có tương quan mạnh, xem
outputs/phase3/manifest.csv describe()) - đủ cho mục đích so sánh TƯƠNG ĐỐI
2 checkpoint, không phải đo R2 tuyệt đối.

Cách chạy:
    python3 analysis/scripts/compare_weighted_sampling.py

Output: in bảng so sánh MAE(v12 FE thật) theo vùng (core/tail) cho 2
checkpoint, lưu outputs/phase5/weighted_sampling_comparison.json.
"""
import os
import sys
import json

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline", "phase5_cvae"))

from model import CVAE                                                      # noqa: E402
from verify_fe import FE_PARAMS, resize_to_fe_grid, evaluate_density_field   # noqa: E402

PHASE5_DIR = os.path.join(REPO_ROOT, "outputs", "phase5")
CKPTS = {
    "baseline (shuffle đều)": os.path.join(PHASE5_DIR, "cvae_gamma20.pt"),
    "weighted-sampling": os.path.join(PHASE5_DIR, "cvae_gamma20_weighted.pt"),
}
# Vùng advisor hỏi ("-0.5 tới 0.3"): mode/core dày mẫu ~[-0.45,-0.25), 2 đuôi
# thưa mẫu - auxetic mạnh (<-0.6) và gần 0/dương (>0.05) - xem
# analyze_auxetic_distribution.py để tái tạo đúng số liệu phần trăm.
REGIONS = {
    "core (mode, dày mẫu)": [-0.45, -0.40, -0.35, -0.30],
    "tail_strong_auxetic (thưa mẫu)": [-0.75, -0.70, -0.65, -0.60],
    "tail_near_zero_pos (thưa mẫu)": [0.05, 0.10, 0.15, 0.20],
}
N_SAMPLES_PER_CONDITION = 10
SEED = 123


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = CVAE(condition_dim=ckpt["condition_dim"], latent_dim=ckpt["latent_dim"],
                 resolution=ckpt["resolution"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def eval_condition(model, v12_target, device, n_samples):
    """Sinh n_samples ảnh (single-shot decoder, không lọc), chạy FE thật,
    trả về list v12 đo được (bỏ qua mẫu FE solve lỗi)."""
    cond = torch.tensor([v12_target, v12_target], dtype=torch.float32, device=device)
    preds = []
    with torch.no_grad():
        imgs = model.generate(cond, n_samples=n_samples, device=device)
    for img in imgs:
        img64 = img.squeeze().cpu().numpy().astype(np.float32)
        img_bin = (img64 > 0.5).astype(np.float32)
        img_fe = resize_to_fe_grid(img_bin, FE_PARAMS["nely"], FE_PARAMS["nelx"])
        try:
            v12_fe, _v21_fe, _ = evaluate_density_field(img_fe, FE_PARAMS)
            preds.append(v12_fe)
        except Exception:
            continue
    return preds


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)

    for path in CKPTS.values():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy checkpoint: {path}")

    report = {}
    for ckpt_name, ckpt_path in CKPTS.items():
        model = load_model(ckpt_path, device)
        report[ckpt_name] = {}
        print(f"\n=== {ckpt_name} ({ckpt_path}) ===")
        for region_name, targets in REGIONS.items():
            all_abs_err = []
            for v12_t in targets:
                preds = eval_condition(model, v12_t, device, N_SAMPLES_PER_CONDITION)
                if not preds:
                    continue
                abs_err = [abs(p - v12_t) for p in preds]
                all_abs_err.extend(abs_err)
            mae = float(np.mean(all_abs_err)) if all_abs_err else float("nan")
            std = float(np.std(all_abs_err)) if all_abs_err else float("nan")
            report[ckpt_name][region_name] = {
                "mae_v12_fe": mae, "std": std, "n": len(all_abs_err),
                "targets": targets,
            }
            print(f"  {region_name:32s} MAE(v12,FE thật)={mae:.4f} (std={std:.4f}, n={len(all_abs_err)})")

    print("\n=== So sánh trực tiếp (MAE thấp hơn = tốt hơn) ===")
    names = list(CKPTS.keys())
    for region_name in REGIONS:
        m0 = report[names[0]][region_name]["mae_v12_fe"]
        m1 = report[names[1]][region_name]["mae_v12_fe"]
        delta = m1 - m0
        better = names[1] if delta < 0 else names[0]
        print(f"  {region_name:32s} {names[0]}={m0:.4f}  {names[1]}={m1:.4f}  "
              f"delta={delta:+.4f} ({'tốt hơn: ' + better if abs(delta) > 1e-6 else 'ngang nhau'})")

    out_path = os.path.join(PHASE5_DIR, "weighted_sampling_comparison.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nĐã lưu: {out_path}")


if __name__ == "__main__":
    main()
