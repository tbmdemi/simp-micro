"""
Phase 5 - active_learning.py
============================================================
Roadmap 7.1-7.3 + 7.5: active-learning loop thật (khác self_play.py, vốn
mining mẫu đối kháng chứ không phải mẫu tốt ở vùng yếu). Mỗi vòng: tìm
target yếu qua coverage_eval() -> sinh+FE-verify ứng viên -> lọc
manufacturable -> fine-tune surrogate rồi cVAE -> đo lại coverage -> quyết
định dừng. Chi tiết thiết kế + kết quả chạy thật: xem EXPERIMENT_LOG.md
mục 2026-07-31.

Cách chạy:
    python3 pipeline/phase5_cvae/active_learning.py --max-rounds 3
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
    """Import theo đường dẫn tuyệt đối, không phải sys.path bare import
    (tránh đụng module trùng tên phase4/phase5 - xem tests/conftest.py)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("phase4_export_for_phase5", PHASE4_EXPORT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.export_surrogate


export_surrogate = _import_export_surrogate()


def find_weak_targets(cvae_ckpt_path: str, n_samples: int, grid_size: int,
                       device: str, n_weak: int = 10, seed: int = 123,
                       v12_range=V12_TRAIN_RANGE):
    """Xếp hạng target auxetic của coverage_eval() theo abs_error giảm
    dần, trả về top n_weak target v12 cần tập trung sinh thêm dữ liệu."""
    result = coverage_eval(cvae_ckpt_path, n_samples, grid_size, device,
                            seed=seed, v12_range=v12_range,
                            check_manuf=False, apply_force_periodic=True)

    auxetic = [t for t in result["per_target"] if t["is_auxetic_target"]]
    # target FE-solve toàn lỗi (abs_error=None) coi là tệ nhất -> +inf
    auxetic_sorted = sorted(
        auxetic, key=lambda t: t["abs_error"] if t["abs_error"] is not None else float("inf"),
        reverse=True,
    )
    weak_targets = [t["target_v12"] for t in auxetic_sorted[:n_weak]]
    return weak_targets, result


def generate_and_verify_candidates(cvae_ckpt_path: str, weak_targets, n_samples: int,
                                    device: str, seed_classes, seed: int = 123):
    """Sinh n_samples ứng viên/target, chấm FE thật trên TẤT CẢ (không chỉ
    mẫu trúng target) - lọc "tốt" là bước riêng, xem filter_good_candidates."""
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
    """Roadmap 7.5. Trả về (stop: bool, reason: str|None)."""
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

        weak_targets, _cov = find_weak_targets(
            cvae_ckpt, args.n_samples, args.grid_size, device,
            n_weak=args.n_weak_targets, seed=args.seed,
        )
        print(f"  Vùng yếu (top {len(weak_targets)}): "
              f"{[round(t, 3) for t in weak_targets]}")

        candidates = generate_and_verify_candidates(
            cvae_ckpt, weak_targets, args.n_samples, device, seed_classes,
            seed=args.seed,
        )
        good = filter_good_candidates(candidates)
        print(f"  Sinh {len(candidates)} ứng viên, {len(good)} manufacturable "
              f"({len(good) / len(candidates) * 100:.1f}%)" if candidates else
              "  Không sinh được ứng viên nào (mọi FE solve lỗi)")

        if not good:
            print("  Không có mẫu tốt nào - dừng loop (không có gì để thêm vào dataset).")
            break

        good_npz = os.path.join(round_dir, "good_candidates.npz")
        save_good_candidates_npz(good, seed_classes, good_npz)

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
    parser.add_argument("--n-samples", type=int, default=15)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--n-weak-targets", type=int, default=10)
    parser.add_argument("--al-oversample", type=int, default=40)
    parser.add_argument("--ft-epochs", type=int, default=10)
    parser.add_argument("--cvae-epochs", type=int, default=20)
    parser.add_argument("--fe-eval-every", type=int, default=5)
    parser.add_argument("--lambda-real-physics", type=float, default=20.0)
    parser.add_argument("--real-physics-subsample", type=int, default=8,
                         help="KHÔNG bỏ trống/None - full-batch real-physics "
                              "biến 1 epoch từ ~30s thành hàng chục phút.")
    parser.add_argument("--real-physics-every", type=int, default=2)
    parser.add_argument("--real-physics-workers", type=int, default=8)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--min-hit-rate-gain", type=float, default=0.02)
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
