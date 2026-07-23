"""
Phase 5 - self_play.py
============================================================
Điều phối vòng lặp self-play giữa surrogate (Phase 4) và cVAE (Phase 5), để
giải quyết vấn đề đã XÁC NHẬN ở outputs/phase5/fe_verification_report.json:
surrogate đông cứng bị decoder "đánh lừa" - R2 đo qua surrogate leo thang
nhưng R2 đo bằng FE thật âm nặng ở mọi gamma.

Mỗi vòng k:
  1. adversarial_dataset.generate_adversarial_npz(): sinh ảnh từ cVAE hiện
     tại ở nhiều condition, chấm điểm THẬT bằng FE -> outputs/phase5/
     self_play/round{k}/adversarial.npz
  2. Fine-tune surrogate trên train.npz + adversarial.npz (subprocess gọi
     phase4_surrogate/train.py --adversarial-npz ... --init-from <surrogate
     vòng trước> --output-name surrogate_round{k}.pt)
  3. export_for_phase5.export_surrogate(): đóng gói checkpoint vừa fine-tune
     -> round{k}/surrogate_for_phase5.pt
  4. Train tiếp cVAE với surrogate mới (subprocess gọi phase5_cvae/train.py
     --resume-from <cvae vòng trước> --surrogate-path round{k}/
     surrogate_for_phase5.pt --fe-eval-every 5 --select-by fe_r2
     --output-name cvae_round{k}.pt)
  5. Verify vòng này bằng FE thật trên tập test giữ riêng (độc lập với
     conditions dùng để sinh adversarial.npz ở bước 1) -> ghi vào
     outputs/phase5/self_play/summary.json

Cách chạy:
    python3 pipeline/phase5_cvae/self_play.py --rounds 2

QUAN TRỌNG: đây là vòng lặp training thật (không phải demo/mock) - mỗi vòng
tốn thời gian thật (fine-tune surrogate + train cVAE + nhiều lần FE solve).
Giảm --ft-epochs/--cvae-epochs/--n-conditions nếu cần chạy nhanh để test.
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
from adversarial_dataset import generate_adversarial_npz, load_cvae  # noqa: E402
from verify_fe import FE_PARAMS, resize_to_fe_grid, evaluate_density_field  # noqa: E402
from dataset import CVAEDataset                                     # noqa: E402

PHASE3_DIR = os.path.join(REPO_ROOT, "outputs", "phase3")
PHASE4_DIR = os.path.join(REPO_ROOT, "outputs", "phase4")
PHASE5_DIR = os.path.join(REPO_ROOT, "outputs", "phase5")
SELF_PLAY_DIR = os.path.join(PHASE5_DIR, "self_play")


def _import_export_surrogate():
    """Import bằng importlib theo đường dẫn tuyệt đối, không phải sys.path +
    `import`, vì phase4_surrogate cũng có dataset.py/model.py trùng tên với
    phase5_cvae - tránh đụng module đã cache sai trong sys.modules (cùng lý
    do losses.py._import_surrogate_cnn() làm vậy)."""
    import importlib.util
    path = os.path.join(REPO_ROOT, "pipeline", "phase4_surrogate", "export_for_phase5.py")
    spec = importlib.util.spec_from_file_location("phase4_export_for_phase5", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.export_surrogate


export_surrogate = _import_export_surrogate()

PHASE4_TRAIN = os.path.join(REPO_ROOT, "pipeline", "phase4_surrogate", "train.py")
PHASE5_TRAIN = os.path.join(REPO_ROOT, "pipeline", "phase5_cvae", "train.py")


def verify_round(cvae_ckpt_path: str, n_conditions: int, n_per_condition: int,
                  device: str, seed: int = 123):
    """Chấm điểm 1 checkpoint cVAE bằng FE thật trên test.npz. `seed` PHẢI cố
    định giống nhau ở mọi lần gọi (mọi round) để so sánh round-với-round là
    apples-to-apples trên CÙNG 1 tập condition - test.npz vốn đã tách biệt
    hoàn toàn với train.npz (nguồn condition của adversarial_dataset.py), nên
    không cần đổi seed mỗi round để "tránh trùng" như bản trước từng làm
    (đó là lỗi: mỗi round vô tình bị chấm trên 1 tập con test khác nhau,
    khiến kết quả round-over-round bị nhiễu bởi độ khó tập test, không chỉ
    bởi chất lượng checkpoint). Cũng cố định torch.manual_seed(seed) trước
    khi generate - model.generate() lấy mẫu z ~ N(0,1) từ RNG toàn cục
    (model.py không tự seed), nên KHÔNG cố định sẽ khiến chấm cùng 1
    checkpoint 2 lần ra 2 kết quả khác nhau (thêm 1 trục nhiễu độc lập với
    trục "tập condition" ở trên) - cả 2 đều phải cố định để so sánh
    round-over-round có ý nghĩa."""
    torch.manual_seed(seed)
    test_ds = CVAEDataset(os.path.join(PHASE3_DIR, "test.npz"))
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(test_ds), size=n_conditions, replace=False)
    conditions = [test_ds[i][1].numpy() for i in idxs]

    model = load_cvae(cvae_ckpt_path, device)
    targets, preds = [], []
    n_auxetic_targets = 0
    n_hits = 0
    for cond in conditions:
        cond_t = torch.tensor(cond, dtype=torch.float32, device=device)
        is_auxetic_target = cond[0] < 0
        n_auxetic_targets += n_per_condition if is_auxetic_target else 0
        for _ in range(n_per_condition):
            with torch.no_grad():
                img = model.generate(cond_t, n_samples=1, device=device)
            img64 = img.squeeze().cpu().numpy().astype(np.float32)
            img_bin = (img64 > 0.5).astype(np.float32)
            img_fe = resize_to_fe_grid(img_bin, FE_PARAMS["nely"], FE_PARAMS["nelx"])
            try:
                v12_fe, _v21_fe, _ = evaluate_density_field(img_fe, FE_PARAMS)
            except Exception:
                continue
            targets.append(cond[0])
            preds.append(v12_fe)
            if is_auxetic_target and v12_fe < 0:
                n_hits += 1

    targets = np.array(targets)
    preds = np.array(preds)
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    hit_rate = n_hits / n_auxetic_targets if n_auxetic_targets else float("nan")
    return {
        "r2_fe_v12": r2,
        "hit_rate": hit_rate,
        "n_auxetic_targets": n_auxetic_targets,
        "n_samples": len(targets),
    }


def run(args):
    os.makedirs(SELF_PLAY_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    surrogate_ckpt = args.start_surrogate_ckpt
    cvae_ckpt = args.start_cvae_ckpt

    summary_path = os.path.join(SELF_PLAY_DIR, "summary.json")
    summary = []
    if args.start_round > 1 and os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        print(f"=== Tiếp tục self-play từ round {args.start_round} "
              f"({len(summary)} round trước đó đã có trong {summary_path}) ===")
        last = summary[-1]
        print(f"  Round cuối trước đó ({last['round']}): "
              f"R2(FE)={last['r2_fe_v12']:.4f} hit_rate={last['hit_rate']:.3f}")
    else:
        print("=== Round 0 (baseline, trước self-play) ===")
        baseline = verify_round(cvae_ckpt, args.n_conditions, args.n_per_condition, device)
        print(f"  R2(FE)={baseline['r2_fe_v12']:.4f} hit_rate={baseline['hit_rate']:.3f} "
              f"({baseline['n_auxetic_targets']} auxetic targets)")
        summary = [{"round": 0, "cvae_ckpt": cvae_ckpt, "surrogate_ckpt": surrogate_ckpt,
                    **baseline}]

    end_round = args.start_round + args.rounds - 1
    for k in range(args.start_round, end_round + 1):
        print(f"\n=== Round {k}/{end_round} ===")
        round_dir = os.path.join(SELF_PLAY_DIR, f"round{k}")
        os.makedirs(round_dir, exist_ok=True)

        # 1. Sinh mẫu đối kháng từ cVAE hiện tại
        adv_npz = os.path.join(round_dir, "adversarial.npz")
        generate_adversarial_npz(
            cvae_ckpt, adv_npz, args.n_conditions, args.seeds_per_condition,
            device, seed=k,
        )

        # 2. Fine-tune surrogate trên train.npz + adversarial.npz
        surrogate_name = f"surrogate_round{k}.pt"
        cmd = [
            sys.executable, PHASE4_TRAIN,
            "--adversarial-npz", adv_npz,
            "--adversarial-oversample", str(args.adv_oversample),
            "--init-from", surrogate_ckpt,
            "--epochs", str(args.ft_epochs),
            "--patience", str(max(3, args.ft_epochs // 3)),
            "--output-name", surrogate_name,
        ]
        print("  $", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
        surrogate_ckpt = os.path.join(PHASE4_DIR, surrogate_name)

        # 3. Export cho Phase 5
        surrogate_export = os.path.join(round_dir, "surrogate_for_phase5.pt")
        export_surrogate(surrogate_ckpt, surrogate_export)

        # 4. Train tiếp cVAE với surrogate mới, chọn checkpoint theo R2(FE) thật
        cvae_name = f"cvae_round{k}.pt"
        cmd = [
            sys.executable, PHASE5_TRAIN,
            "--resume-from", cvae_ckpt,
            "--surrogate-path", surrogate_export,
            "--gamma", str(args.gamma),
            "--epochs", str(args.cvae_epochs),
            "--patience", str(max(3, args.cvae_epochs // 3)),
            "--fe-eval-every", str(args.fe_eval_every),
            "--select-by", "fe_r2",
            "--lambda-tv", str(args.lambda_tv),
            "--lambda-bin", str(args.lambda_bin),
            "--output-name", cvae_name,
        ]
        print("  $", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
        cvae_ckpt = os.path.join(PHASE5_DIR, cvae_name)

        # 5. Verify vòng này bằng FE thật - CÙNG 1 tập condition cố định mọi
        # round (seed mặc định của verify_round) để so sánh round-over-round
        # không bị nhiễu bởi tập test khác nhau (xem docstring verify_round).
        result = verify_round(cvae_ckpt, args.n_conditions, args.n_per_condition, device)
        print(f"  --> R2(FE)={result['r2_fe_v12']:.4f} hit_rate={result['hit_rate']:.3f}")
        summary.append({"round": k, "cvae_ckpt": cvae_ckpt, "surrogate_ckpt": surrogate_ckpt,
                         **result})

        with open(os.path.join(SELF_PLAY_DIR, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    print("\n=== Tổng kết self-play (R2 đo bằng FE thật, KHÔNG qua surrogate) ===")
    print(f"{'round':>5} | {'R2(FE)':>8} | {'hit_rate':>8}")
    for row in summary:
        print(f"{row['round']:>5} | {row['r2_fe_v12']:>8.4f} | {row['hit_rate']:>8.3f}")
    print(f"\nChi tiết: {os.path.join(SELF_PLAY_DIR, 'summary.json')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--n-conditions", type=int, default=8)
    parser.add_argument("--n-per-condition", type=int, default=3,
                         help="Số mẫu sinh ra mỗi condition lúc VERIFY (bước 5) - "
                              "khác --seeds-per-condition (dùng lúc sinh dữ liệu train "
                              "surrogate ở bước 1).")
    parser.add_argument("--seeds-per-condition", type=int, default=2)
    parser.add_argument("--adv-oversample", type=int, default=40,
                         help="Số lần lặp mỗi mẫu đối kháng khi fine-tune surrogate "
                              "(xem --adversarial-oversample của phase4 train.py). "
                              "Mặc định trước đây là 1 lần (~0.1%% batch, vô hình) - "
                              "40 đưa tỉ trọng lên mức đáng kể mà không lấn át "
                              "hoàn toàn 33k mẫu thật.")
    parser.add_argument("--ft-epochs", type=int, default=10,
                         help="Số epoch fine-tune surrogate mỗi vòng.")
    parser.add_argument("--cvae-epochs", type=int, default=15,
                         help="Số epoch train tiếp cVAE mỗi vòng.")
    parser.add_argument("--fe-eval-every", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=20.0,
                         help="gamma=20 được chọn vì có hit-rate FE thật tốt nhất "
                              "(12/24) trong gamma-sweep gốc, xem README §5.")
    parser.add_argument("--lambda-tv", type=float, default=0.0,
                         help="Trọng số total-variation regularizer (bổ sung cho "
                              "self-play, xem README - chưa từng kết hợp trước đây).")
    parser.add_argument("--lambda-bin", type=float, default=0.0,
                         help="Trọng số binarization regularizer (bổ sung cho "
                              "self-play, xem README - chưa từng kết hợp trước đây).")
    parser.add_argument("--start-round", type=int, default=1,
                         help="Số vòng bắt đầu đánh số (>1 để tiếp tục 1 lần chạy "
                              "self-play trước đó thay vì ghi đè round1.. - sẽ nạp "
                              "summary.json cũ và nối thêm thay vì tính lại round 0).")
    parser.add_argument(
        "--start-cvae-ckpt", type=str,
        default=os.path.join(PHASE5_DIR, "cvae_gamma20.pt"),
    )
    parser.add_argument(
        "--start-surrogate-ckpt", type=str,
        default=os.path.join(PHASE4_DIR, "surrogate_best.pt"),
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
