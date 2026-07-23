"""
Phase 5 - coverage_eval.py
============================================================
Roadmap 7.4: đo "coverage" của best-of-N trên 1 LƯỚI target v12 trải đều
khắp miền train (~[-0.81, 0.37], xem sample.py LƯU Ý), thay vì 24 condition
NGẪU NHIÊN lấy từ test.npz mà best_of_n_eval.py dùng mặc định. Mục đích:
phát hiện "vùng chết" - target mà ngay cả best-of-N (N mẫu, chọn bằng FE
thật) cũng không tìm được ứng viên đúng dấu Poisson ratio mong muốn - để có
căn cứ quyết định 7.5 (có đáng đầu tư active-learning thật, thu thập thêm
dữ liệu SIMP có mục tiêu, hay best-of-N hiện tại đã đủ tốt khắp miền).

v21 = v12 trong lưới này (đơn giản hoá 1 chiều, giống ví dụ dùng trong
sample.py --v12 -0.6 --v21 -0.6) - không phải coverage 2D đầy đủ của
(v12, v21) độc lập, nhưng đủ để trả lời câu hỏi "có vùng nào dọc trục target
auxetic chính mà generator không với tới được".

hit (chỉ tính cho target auxetic, v12<0, giống định nghĩa hit_rate trong
best_of_n_eval.py/self_play.py): best_v12 (ứng viên gần target nhất trong N
mẫu, chấm bằng FE thật) có <0 hay không. Target không-auxetic (v12>=0)
không tính hit/miss, chỉ báo cáo abs_error để tham khảo.

Cách chạy:
    python3 pipeline/phase5_cvae/coverage_eval.py \\
        --cvae-ckpt outputs/phase5/cvae_gamma20.pt --n-samples 15 --grid-size 8
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from verify_fe import FE_PARAMS, resize_to_fe_grid, evaluate_density_field  # noqa: E402
from self_play import load_cvae                                            # noqa: E402
from manufacturability import check_manufacturability                      # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHASE5_DIR = os.path.join(REPO_ROOT, "outputs", "phase5")

V12_TRAIN_RANGE = (-0.81, 0.37)


def coverage_eval(cvae_ckpt_path: str, n_samples: int, grid_size: int,
                   device: str, seed: int = 123,
                   v12_range=V12_TRAIN_RANGE, check_manuf: bool = False):
    torch.manual_seed(seed)
    model = load_cvae(cvae_ckpt_path, device)
    grid = np.linspace(v12_range[0], v12_range[1], grid_size)

    per_target = []
    for v12_target in grid:
        v12_target = float(v12_target)
        cond_t = torch.tensor([v12_target, v12_target], dtype=torch.float32, device=device)
        is_auxetic_target = v12_target < 0

        imgs = []
        for _ in range(n_samples):
            with torch.no_grad():
                img = model.generate(cond_t, n_samples=1, device=device)
            imgs.append(img.squeeze().cpu().numpy().astype(np.float32))

        v12_reals = []
        n_manufacturable = 0
        for img in imgs:
            img_bin = (img > 0.5).astype(np.float32)
            if check_manuf and check_manufacturability(img_bin)["passes_all"]:
                n_manufacturable += 1
            img_fe = resize_to_fe_grid(img_bin, FE_PARAMS["nely"], FE_PARAMS["nelx"])
            try:
                v12_fe, _v21_fe, _ = evaluate_density_field(img_fe, FE_PARAMS)
            except Exception:
                continue
            v12_reals.append(v12_fe)

        entry = {
            "target_v12": v12_target,
            "is_auxetic_target": bool(is_auxetic_target),
            "n_valid_samples": len(v12_reals),
            "n_manufacturable": n_manufacturable,
            "frac_manufacturable": n_manufacturable / n_samples if check_manuf else None,
        }
        if not v12_reals:
            entry.update({"best_v12": None, "abs_error": None, "hit": None})
            per_target.append(entry)
            continue

        v12_reals = np.array(v12_reals)
        best_idx = int(np.argmin(np.abs(v12_reals - v12_target)))
        best_v12 = float(v12_reals[best_idx])
        hit = bool(best_v12 < 0) if is_auxetic_target else None
        entry.update({
            "best_v12": best_v12,
            "abs_error": float(abs(best_v12 - v12_target)),
            "hit": hit,
        })
        per_target.append(entry)

    auxetic_rows = [t for t in per_target if t["is_auxetic_target"] and t["hit"] is not None]
    hit_rate = float(np.mean([t["hit"] for t in auxetic_rows])) if auxetic_rows else float("nan")
    dead_zones = [t["target_v12"] for t in auxetic_rows if not t["hit"]]
    valid_errors = [t["abs_error"] for t in per_target if t["abs_error"] is not None]
    mean_abs_error = float(np.mean(valid_errors)) if valid_errors else float("nan")

    return {
        "grid_size": grid_size,
        "n_samples_per_target": n_samples,
        "v12_range": list(v12_range),
        "n_auxetic_targets": len(auxetic_rows),
        "hit_rate": hit_rate,
        "mean_abs_error": mean_abs_error,
        "dead_zone_targets": dead_zones,
        "per_target": per_target,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cvae-ckpt", type=str,
                         default=os.path.join(PHASE5_DIR, "cvae_gamma20.pt"))
    parser.add_argument("--n-samples", type=int, default=15,
                         help="Số ứng viên sinh ra MỖI target trên lưới, chấm FE thật trên tất cả (oracle).")
    parser.add_argument("--grid-size", type=int, default=8,
                         help="Số điểm target v12 trải đều trong --v12-range.")
    parser.add_argument("--v12-min", type=float, default=V12_TRAIN_RANGE[0])
    parser.add_argument("--v12-max", type=float, default=V12_TRAIN_RANGE[1])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--check-manufacturability", action="store_true",
                         help="Ghi thêm frac_manufacturable mỗi target (roadmap 6.2/6.3).")
    parser.add_argument("--out", type=str,
                         default=os.path.join(PHASE5_DIR, "self_play", "coverage_result.json"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = coverage_eval(
        args.cvae_ckpt, args.n_samples, args.grid_size, device, args.seed,
        v12_range=(args.v12_min, args.v12_max), check_manuf=args.check_manufacturability,
    )

    print(f"Checkpoint: {args.cvae_ckpt}")
    print(f"Lưới {result['grid_size']} target v12 trong {result['v12_range']} "
          f"({result['n_auxetic_targets']} target auxetic) x {result['n_samples_per_target']} mẫu/target")
    print(f"  hit_rate (auxetic targets, best-of-N oracle) = {result['hit_rate']:.3f}")
    print(f"  mean_abs_error (v12, tất cả target)          = {result['mean_abs_error']:.4f}")
    if result["dead_zone_targets"]:
        print(f"  VÙNG CHẾT (target auxetic không hit): {result['dead_zone_targets']}")
    else:
        print("  Không có vùng chết trong lưới đã thử.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nĐã lưu: {args.out}")


if __name__ == "__main__":
    main()
