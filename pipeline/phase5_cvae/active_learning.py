"""
Phase 5 - active_learning.py
============================================================
Roadmap 7.1-7.3 + 7.5: vòng lặp active-learning THẬT (khác self_play.py -
xem docstring bên dưới), đóng vòng còn thiếu sau 7.4 (coverage_eval.py).

Mỗi vòng k:
  1. find_weak_targets(): chạy coverage_eval() hiện có (roadmap 7.4) trên
     checkpoint cVAE hiện tại, lấy N target auxetic có abs_error lớn nhất
     (gồm cả dead_zone_targets, abs_error=inf) - đây là điểm khác self_play
     (sinh mẫu NGẪU NHIÊN từ train.npz để "đánh lừa" surrogate); ở đây ta
     CHỦ ĐỘNG nhắm vào vùng property-space còn thưa/sai nhiều.
  2. generate_and_verify_candidates(): sinh --n-samples ứng viên cho MỖI
     target yếu, chấm FE THẬT (evaluate_density_field) - MỌI mẫu FE-eval
     thành công đều lấy nhãn thật, không chỉ mẫu "trúng" target (surrogate
     cần nhãn thật đáng tin ở vùng thưa, không chỉ mẫu thắng).
  3. filter_good_candidates(): giữ mẫu FE thành công VÀ
     check_manufacturability().passes_all - "tốt" = nhãn đáng tin + sản
     xuất được, không phải "đúng target tuyệt đối". Lưu .npz cùng schema
     adversarial_dataset.py (images, v12, v21, volfrac_achieved,
     seed_onehot, seed_classes) - seed_onehot gán bằng cách xoay vòng
     seed_classes (đúng quy ước adversarial_dataset.py:100-101, chỉ để
     khớp schema AuxeticDataset, không mang ý nghĩa seed thật).
  4. Fine-tune surrogate (subprocess phase4_surrogate/train.py
     --adversarial-npz <good.npz> --init-from <surrogate vòng trước>) rồi
     export_for_phase5.export_surrogate() - CÙNG cơ chế self_play.py dùng,
     chỉ khác NỘI DUNG npz (mẫu tốt/nhắm-vùng-yếu, không phải đối kháng).
  5. Fine-tune cVAE (subprocess phase5_cvae/train.py --resume-from
     --surrogate-path <surrogate mới> --select-by fe_r2
     --lambda-real-physics) - cVAE KHÔNG được train lại trên chính ảnh mới
     sinh (không có cờ tương đương --adversarial-npz ở phase5/train.py,
     self_play.py cũng không làm việc này) - hưởng lợi gián tiếp qua
     gradient property-consistency/real-physics của surrogate mới.
  6. coverage_eval() lại trên checkpoint mới, so sánh hit_rate/
     dead_zone_targets/mean_abs_error với vòng trước, ghi
     outputs/phase5/active_learning/summary.json.

7.5 - quyết định dừng loop (should_stop_loop()): dừng khi (a) số
dead_zone_targets KHÔNG giảm so với vòng trước, HOẶC (b) hit_rate cải
thiện <2 điểm % so với vòng trước, HOẶC (c) đã chạy đủ --max-rounds (mặc
định 3) - lấy điều kiện nào xảy ra trước. Không đặt cược >3 vòng ngay từ
đầu - self_play.py từng chạy 8 vòng và không hội tụ ổn định (R2 phẳng/
nhiễu, xem EXPERIMENT_LOG.md), tránh lặp lại sai lầm "cứ chạy thêm vòng
là sẽ tốt hơn" mà không có bằng chứng.

Cách chạy:
    python3 pipeline/phase5_cvae/active_learning.py --max-rounds 3

QUAN TRỌNG: đây là vòng lặp training thật (không phải demo/mock) - mỗi
vòng tốn thời gian thật (sinh+verify ứng viên bằng FE thật, fine-tune
surrogate, fine-tune cVAE). Dùng --n-samples/--ft-epochs/--cvae-epochs/
--grid-size nhỏ để smoke-test trước khi chạy full-size.
"""
import os
import sys
import json
import argparse
import subprocess
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from coverage_eval import coverage_eval, V12_TRAIN_RANGE          # noqa: E402
from self_play import load_cvae                                   # noqa: E402
from verify_fe import FE_PARAMS, resize_to_fe_grid, evaluate_density_field  # noqa: E402
from manufacturability import check_manufacturability, force_periodic       # noqa: E402

PHASE3_DIR = os.path.join(REPO_ROOT, "outputs", "phase3")
PHASE4_DIR = os.path.join(REPO_ROOT, "outputs", "phase4")
PHASE5_DIR = os.path.join(REPO_ROOT, "outputs", "phase5")
AL_DIR = os.path.join(PHASE5_DIR, "active_learning")

PHASE4_TRAIN = os.path.join(REPO_ROOT, "pipeline", "phase4_surrogate", "train.py")
PHASE4_EXPORT = os.path.join(REPO_ROOT, "pipeline", "phase4_surrogate", "export_for_phase5.py")
PHASE5_TRAIN = os.path.join(REPO_ROOT, "pipeline", "phase5_cvae", "train.py")


def _import_export_surrogate():
    """Cùng lý do self_play.py._import_export_surrogate(): phase4_surrogate
    và phase5_cvae đều có dataset.py/model.py trùng tên - import theo
    đường dẫn tuyệt đối để tránh đụng module cache sai trong sys.modules."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("phase4_export_for_phase5", PHASE4_EXPORT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.export_surrogate


export_surrogate = _import_export_surrogate()


def find_weak_targets(cvae_ckpt_path: str, n_samples: int, grid_size: int,
                       device: str, n_weak: int = 10, seed: int = 123,
                       v12_range=V12_TRAIN_RANGE):
    """Chạy coverage_eval() (roadmap 7.4), xếp hạng target auxetic theo
    abs_error giảm dần (dead_zone -> abs_error=None -> coi là tệ nhất,
    xếp lên đầu), trả về tối đa n_weak target v12 cần tập trung sinh thêm
    dữ liệu. Trả kèm result đầy đủ để ghi vào summary.json."""
    result = coverage_eval(cvae_ckpt_path, n_samples, grid_size, device,
                            seed=seed, v12_range=v12_range,
                            check_manuf=False, apply_force_periodic=True)

    auxetic = [t for t in result["per_target"] if t["is_auxetic_target"]]
    # None (FE toàn lỗi ở target đó) coi là tệ nhất -> +inf để lên đầu khi sort giảm dần
    auxetic_sorted = sorted(
        auxetic, key=lambda t: t["abs_error"] if t["abs_error"] is not None else float("inf"),
        reverse=True,
    )
    weak_targets = [t["target_v12"] for t in auxetic_sorted[:n_weak]]
    return weak_targets, result


def generate_and_verify_candidates(cvae_ckpt_path: str, weak_targets, n_samples: int,
                                    device: str, seed_classes, seed: int = 123):
    """Sinh n_samples ứng viên MỖI target yếu, chấm FE thật trên TẤT CẢ
    (không chỉ mẫu trúng) - trả về list dict {image, v12, v21, volfrac,
    seed_onehot} cho MỌI mẫu FE-eval thành công (lọc "tốt"/manufacturable
    là bước riêng, xem filter_good_candidates)."""
    torch.manual_seed(seed)
    model = load_cvae(cvae_ckpt_path, device)
    n_seeds = len(seed_classes)

    candidates = []
    for i, v12_target in enumerate(weak_targets):
        cond_t = torch.tensor([v12_target, v12_target], dtype=torch.float32, device=device)
        for j in range(n_samples):
            with torch.no_grad():
                img = model.generate(cond_t, n_samples=1, device=device)
            img64 = img.squeeze().cpu().numpy().astype(np.float32)
            img64 = force_periodic(img64)
            img_bin = (img64 > 0.5).astype(np.float32)
            img_fe = resize_to_fe_grid(img_bin, FE_PARAMS["nely"], FE_PARAMS["nelx"])
            try:
                v12_real, v21_real, _ = evaluate_density_field(img_fe, FE_PARAMS)
            except Exception:
                continue
            onehot = np.zeros(n_seeds, dtype=np.float32)
            onehot[(i * n_samples + j) % n_seeds] = 1.0
            candidates.append({
                "image": img64,
                "img_bin": img_bin,
                "v12": v12_real,
                "v21": v21_real,
                "volfrac": float(img_fe.mean()),
                "seed_onehot": onehot,
            })
    return candidates


def filter_good_candidates(candidates, min_feature_px: int = 2, periodicity_tol: float = 0.1):
    """Giữ mẫu manufacturable (connectivity + min-feature + periodicity) -
    "tốt" = nhãn đáng tin/sản xuất được, không phải "trúng target"."""
    good = []
    for c in candidates:
        manuf = check_manufacturability(c["img_bin"], min_feature_px=min_feature_px,
                                         periodicity_tol=periodicity_tol)
        if manuf["passes_all"]:
            good.append(c)
    return good


def save_good_candidates_npz(good_candidates, seed_classes, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(
        out_path,
        images=np.stack([c["image"] for c in good_candidates]),
        v12=np.array([c["v12"] for c in good_candidates], dtype=np.float32),
        v21=np.array([c["v21"] for c in good_candidates], dtype=np.float32),
        volfrac_achieved=np.array([c["volfrac"] for c in good_candidates], dtype=np.float32),
        seed_onehot=np.stack([c["seed_onehot"] for c in good_candidates]),
        seed_classes=seed_classes,
    )
    return out_path


def should_stop_loop(summary, round_idx: int, max_rounds: int,
                      min_hit_rate_gain: float = 0.02):
    """7.5: dừng khi (a) dead_zone không giảm, (b) hit_rate cải thiện <
    min_hit_rate_gain so với vòng trước, hoặc (c) đã tới max_rounds - điều
    kiện nào xảy ra trước. Trả về (stop: bool, reason: str|None)."""
    if round_idx >= max_rounds:
        return True, f"đã chạy đủ max_rounds={max_rounds}"
    if len(summary) < 2:
        return False, None

    prev, curr = summary[-2], summary[-1]
    prev_dead = len(prev["dead_zone_targets"])
    curr_dead = len(curr["dead_zone_targets"])
    if curr_dead >= prev_dead:
        return True, (f"dead_zone_targets không giảm ({prev_dead}->{curr_dead})")

    prev_hr = prev["hit_rate"] if prev["hit_rate"] == prev["hit_rate"] else 0.0  # NaN-safe
    curr_hr = curr["hit_rate"] if curr["hit_rate"] == curr["hit_rate"] else 0.0
    gain = curr_hr - prev_hr
    if gain < min_hit_rate_gain:
        return True, (f"hit_rate cải thiện {gain:.3f} < ngưỡng {min_hit_rate_gain}")

    return False, None


def run(args):
    os.makedirs(AL_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    surrogate_ckpt = args.start_surrogate_ckpt
    cvae_ckpt = args.start_cvae_ckpt

    train_raw = np.load(os.path.join(PHASE3_DIR, "train.npz"), allow_pickle=True)
    seed_classes = train_raw["seed_classes"]

    summary_path = os.path.join(AL_DIR, "summary.json")
    summary = []
    if args.start_round > 1 and os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        print(f"=== Tiếp tục active-learning từ round {args.start_round} "
              f"({len(summary)} round trước đó trong {summary_path}) ===")
    else:
        print("=== Round 0 (baseline, trước active-learning) ===")
        weak0, cov0 = find_weak_targets(cvae_ckpt, args.n_samples, args.grid_size,
                                         device, n_weak=args.n_weak_targets)
        print(f"  hit_rate={cov0['hit_rate']:.3f} mean_abs_error={cov0['mean_abs_error']:.4f} "
              f"dead_zone={len(cov0['dead_zone_targets'])}")
        summary = [{
            "round": 0, "cvae_ckpt": cvae_ckpt, "surrogate_ckpt": surrogate_ckpt,
            "hit_rate": cov0["hit_rate"], "mean_abs_error": cov0["mean_abs_error"],
            "dead_zone_targets": cov0["dead_zone_targets"], "n_good_candidates": 0,
        }]
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    round_idx = args.start_round - 1
    while True:
        round_idx += 1
        print(f"\n=== Round {round_idx} (max {args.max_rounds}) ===")
        round_dir = os.path.join(AL_DIR, f"round{round_idx}")
        os.makedirs(round_dir, exist_ok=True)

        # 1. Tìm vùng yếu
        weak_targets, _cov = find_weak_targets(
            cvae_ckpt, args.n_samples, args.grid_size, device,
            n_weak=args.n_weak_targets, seed=args.seed,
        )
        print(f"  Vùng yếu (top {len(weak_targets)}): "
              f"{[round(t, 3) for t in weak_targets]}")

        # 2. Sinh + chấm FE thật
        candidates = generate_and_verify_candidates(
            cvae_ckpt, weak_targets, args.n_samples, device, seed_classes,
            seed=args.seed,
        )
        # 3. Lọc "tốt" (manufacturable)
        good = filter_good_candidates(candidates)
        print(f"  Sinh {len(candidates)} ứng viên, {len(good)} manufacturable "
              f"({len(good) / len(candidates) * 100:.1f}%)" if candidates else
              "  Không sinh được ứng viên nào (mọi FE solve lỗi)")

        if not good:
            print("  Không có mẫu tốt nào - dừng loop (không có gì để thêm vào dataset).")
            break

        good_npz = os.path.join(round_dir, "good_candidates.npz")
        save_good_candidates_npz(good, seed_classes, good_npz)

        # 4. Fine-tune surrogate
        surrogate_name = f"surrogate_al_round{round_idx}.pt"
        cmd = [
            sys.executable, PHASE4_TRAIN,
            "--adversarial-npz", good_npz,
            "--adversarial-oversample", str(args.al_oversample),
            "--init-from", surrogate_ckpt,
            "--epochs", str(args.ft_epochs),
            "--patience", str(max(3, args.ft_epochs // 3)),
            "--output-name", surrogate_name,
        ]
        print("  $", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
        surrogate_ckpt = os.path.join(PHASE4_DIR, surrogate_name)

        surrogate_export = os.path.join(round_dir, "surrogate_for_phase5.pt")
        export_surrogate(surrogate_ckpt, surrogate_export)

        # 5. Fine-tune cVAE
        cvae_name = f"cvae_al_round{round_idx}.pt"
        cmd = [
            sys.executable, PHASE5_TRAIN,
            "--resume-from", cvae_ckpt,
            "--surrogate-path", surrogate_export,
            "--lambda-real-physics", str(args.lambda_real_physics),
            "--real-physics-subsample", str(args.real_physics_subsample),
            "--real-physics-every", str(args.real_physics_every),
            "--real-physics-workers", str(args.real_physics_workers),
            "--epochs", str(args.cvae_epochs),
            "--patience", str(max(3, args.cvae_epochs // 3)),
            "--fe-eval-every", str(args.fe_eval_every),
            "--select-by", "fe_r2",
            "--output-name", cvae_name,
        ]
        print("  $", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
        cvae_ckpt = os.path.join(PHASE5_DIR, cvae_name)

        # 6. Đo lại coverage
        _weak, cov = find_weak_targets(cvae_ckpt, args.n_samples, args.grid_size,
                                        device, n_weak=args.n_weak_targets, seed=args.seed)
        print(f"  --> hit_rate={cov['hit_rate']:.3f} mean_abs_error={cov['mean_abs_error']:.4f} "
              f"dead_zone={len(cov['dead_zone_targets'])}")
        summary.append({
            "round": round_idx, "cvae_ckpt": cvae_ckpt, "surrogate_ckpt": surrogate_ckpt,
            "hit_rate": cov["hit_rate"], "mean_abs_error": cov["mean_abs_error"],
            "dead_zone_targets": cov["dead_zone_targets"], "n_good_candidates": len(good),
        })
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # 7.5 - quyết định dừng
        stop, reason = should_stop_loop(summary, round_idx, args.max_rounds,
                                         args.min_hit_rate_gain)
        if stop:
            print(f"\n=== Dừng loop sau round {round_idx}: {reason} ===")
            break

    print("\n=== Tổng kết active-learning ===")
    print(f"{'round':>5} | {'hit_rate':>8} | {'mean_abs_err':>12} | {'dead_zone':>9} | {'n_good':>6}")
    for row in summary:
        print(f"{row['round']:>5} | {row['hit_rate']:>8.3f} | {row['mean_abs_error']:>12.4f} | "
              f"{len(row['dead_zone_targets']):>9} | {row['n_good_candidates']:>6}")
    print(f"\nChi tiết: {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=15,
                         help="Số mẫu sinh/chấm FE mỗi target (dùng cả cho coverage_eval "
                              "lẫn sinh ứng viên ở vùng yếu).")
    parser.add_argument("--grid-size", type=int, default=8,
                         help="Số điểm target v12 trải đều để đo coverage (roadmap 7.4).")
    parser.add_argument("--n-weak-targets", type=int, default=10,
                         help="Số target yếu nhất (abs_error cao nhất) để tập trung sinh thêm.")
    parser.add_argument("--al-oversample", type=int, default=40,
                         help="Số lần lặp mẫu tốt khi fine-tune surrogate - cùng lý do "
                              "self_play.py --adv-oversample: batch nhỏ cần trọng số đủ lớn "
                              "để có gradient signal đáng kể so với ~57k mẫu thật.")
    parser.add_argument("--ft-epochs", type=int, default=10,
                         help="Số epoch fine-tune surrogate mỗi vòng.")
    parser.add_argument("--cvae-epochs", type=int, default=20,
                         help="Số epoch fine-tune cVAE mỗi vòng.")
    parser.add_argument("--fe-eval-every", type=int, default=5)
    parser.add_argument("--lambda-real-physics", type=float, default=20.0,
                         help="Khớp cấu hình cvae_realphysics.pt đã kiểm chứng (README).")
    parser.add_argument("--real-physics-subsample", type=int, default=8,
                         help="Số mẫu/batch được FE-solve thật (KHÔNG để None/full-batch - "
                              "README §5 dùng đúng giá trị này cho fine-tune 35 epoch "
                              "~15-18 phút; bỏ qua cờ này khiến train.py FE-solve TOÀN BỘ "
                              "batch mỗi bước, biến 1 epoch từ ~30s thành hàng chục phút.")
    parser.add_argument("--real-physics-every", type=int, default=2,
                         help="Chỉ tính real-physics loss mỗi N batch (README §5 dùng 2).")
    parser.add_argument("--real-physics-workers", type=int, default=8,
                         help="Số worker song song hoá FE-solve (README §5 dùng 8).")
    parser.add_argument("--max-rounds", type=int, default=3,
                         help="Ngân sách tối đa (roadmap 7.5) - self_play.py từng chạy 8 vòng "
                              "không hội tụ ổn định, không đặt cược cao ngay từ đầu.")
    parser.add_argument("--min-hit-rate-gain", type=float, default=0.02,
                         help="Ngưỡng cải thiện hit_rate tối thiểu để tiếp tục vòng kế (roadmap 7.5).")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--start-round", type=int, default=1)
    parser.add_argument(
        "--start-cvae-ckpt", type=str,
        default=os.path.join(PHASE5_DIR, "cvae_realphysics.pt"),
    )
    parser.add_argument(
        "--start-surrogate-ckpt", type=str,
        default=os.path.join(PHASE4_DIR, "surrogate_for_phase5_v2.pt"),
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
