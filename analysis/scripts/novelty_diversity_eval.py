"""
Novelty / diversity eval cho cVAE (Phase 5)
============================================================
Trả lời trực tiếp câu hỏi #6 trong AUDIT_REPORT_SCIENTIFIC_REVIEW_2026-08-02.md:
"cVAE có thật sự sinh ra design mới, hay chỉ đang tạo lại mẫu gần với phân
phối training?" Không có metric này trong pipeline hiện có (evaluate.py::
diversity_check chỉ đo qua surrogate, không so với training set thật).

Hai chỉ số, đo trên density field thô [0,1] (không binarize - so sánh đúng
không gian mà cVAE thực sự sinh ra):

1. Novelty: với mỗi ảnh cVAE sinh ra cho 1 target, khoảng cách L2 (pixel
   space, 64x64=4096 chiều) tới ảnh training GẦN NHẤT (toàn bộ 57.216 mẫu
   train.npz, không subsample). Khoảng cách nhỏ => nghi ngờ "gần như sao
   chép" 1 mẫu train.
2. Diversity: với K mẫu sinh ra CÙNG 1 target, khoảng cách L2 trung bình
   giữa các cặp mẫu - K mẫu giống hệt nhau (mode collapse theo condition)
   sẽ cho diversity ~0 dù novelty có thể vẫn cao.

Để 2 con số trên có ý nghĩa (không phải đơn vị đo mơ hồ), tính thêm 1
đường tham chiếu: khoảng cách NN thật-tới-thật (lấy ngẫu nhiên 300 mẫu
train, tìm hàng xóm gần nhất trong PHẦN CÒN LẠI của train set, loại bỏ
chính nó). Nếu novelty của mẫu sinh ra thấp hơn hẳn phân phối tham chiếu
này, đó là bằng chứng cụ thể cho "đang sao chép" thay vì chỉ là cảm tính.

Báo cáo PHÂN PHỐI (percentile), không chỉ trung bình - đúng yêu cầu audit
report mục 4.3 ("dùng thống kê chứ không chỉ số trung bình").

Cách chạy:
    python3 analysis/scripts/novelty_diversity_eval.py \\
        --cvae-ckpt outputs/phase5/cvae_v2_finetuned.pt \\
        --n-conditions 24 --n-samples-per-condition 20

Output: outputs/phase5/reports/novelty_diversity_<ckpt-stem>.json
"""
import os
import sys
import json
import argparse

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline", "phase5_cvae"))

from dataset import CVAEDataset          # noqa: E402
from adversarial_dataset import load_cvae  # noqa: E402
from manufacturability import force_periodic  # noqa: E402

PHASE3_DIR = os.path.join(REPO_ROOT, "outputs", "phase3")
PHASE5_DIR = os.path.join(REPO_ROOT, "outputs", "phase5")


def _load_train_flat() -> np.ndarray:
    data = np.load(os.path.join(PHASE3_DIR, "train.npz"), allow_pickle=True)
    imgs = data["images"].astype(np.float32)
    return imgs.reshape(imgs.shape[0], -1)


def min_dist_to_train(query: np.ndarray, train_flat: np.ndarray, chunk: int = 10000) -> np.ndarray:
    """query: (Nq, D) float32. Trả (Nq,) khoảng cách L2 tới hàng xóm gần
    nhất trong train_flat, chunk theo train để tránh cấp phát ma trận
    (Nq x Ntrain) một lần (57.216 mẫu x D=4096 quá lớn nếu Nq lớn)."""
    q_norm = (query ** 2).sum(axis=1)
    min_d2 = np.full(query.shape[0], np.inf, dtype=np.float64)
    for start in range(0, train_flat.shape[0], chunk):
        t = train_flat[start:start + chunk]
        t_norm = (t ** 2).sum(axis=1)
        d2 = q_norm[:, None] + t_norm[None, :] - 2.0 * query.astype(np.float64) @ t.T.astype(np.float64)
        np.maximum(d2, 0.0, out=d2)
        min_d2 = np.minimum(min_d2, d2.min(axis=1))
    return np.sqrt(min_d2)


def real_to_real_reference(train_flat: np.ndarray, n_query: int = 300, seed: int = 0) -> np.ndarray:
    """NN thật-tới-thật (loại bỏ chính nó) trên n_query mẫu train ngẫu
    nhiên, tìm trong TOÀN BỘ train set - đường tham chiếu diễn giải novelty."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(train_flat.shape[0], size=min(n_query, train_flat.shape[0]), replace=False)
    q = train_flat[idx].astype(np.float64)
    t_norm_full = (train_flat.astype(np.float64) ** 2).sum(axis=1)
    q_norm = (q ** 2).sum(axis=1)
    d2 = q_norm[:, None] + t_norm_full[None, :] - 2.0 * q @ train_flat.T.astype(np.float64)
    np.maximum(d2, 0.0, out=d2)
    d2[np.arange(len(idx)), idx] = np.inf  # loại chính nó
    return np.sqrt(d2.min(axis=1))


def percentiles(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    qs = [0, 5, 10, 25, 50, 75, 90, 100]
    vals = np.percentile(x, qs)
    return {f"p{q}": float(v) for q, v in zip(qs, vals)} | {"mean": float(x.mean()), "n": int(len(x))}


def run(cvae_ckpt: str, n_conditions: int, n_samples_per_condition: int,
        device: str, seed: int = 123, apply_force_periodic: bool = True) -> dict:
    train_flat = _load_train_flat()
    ref_dist = real_to_real_reference(train_flat, n_query=300, seed=0)

    test_ds = CVAEDataset(os.path.join(PHASE3_DIR, "test.npz"))
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(test_ds), size=n_conditions, replace=False)
    conditions = [test_ds[i][1].numpy() for i in idxs]

    model = load_cvae(cvae_ckpt, device)
    torch.manual_seed(seed)

    all_novelty = []
    all_diversity = []
    per_condition = []
    for cond in conditions:
        cond_t = torch.tensor(cond, dtype=torch.float32, device=device)
        imgs = []
        for _ in range(n_samples_per_condition):
            with torch.no_grad():
                img = model.generate(cond_t, n_samples=1, device=device)
            imgs.append(img.squeeze().cpu().numpy().astype(np.float32))
        if apply_force_periodic:
            imgs = [force_periodic(img) for img in imgs]
        imgs_flat = np.stack(imgs).reshape(len(imgs), -1)

        novelty = min_dist_to_train(imgs_flat, train_flat)
        all_novelty.extend(novelty.tolist())

        # diversity: trung bình khoảng cách cặp trong K mẫu cùng 1 condition
        k = imgs_flat.shape[0]
        pair_d = []
        for i in range(k):
            for j in range(i + 1, k):
                pair_d.append(float(np.linalg.norm(imgs_flat[i] - imgs_flat[j])))
        mean_pair_d = float(np.mean(pair_d)) if pair_d else float("nan")
        all_diversity.append(mean_pair_d)

        per_condition.append({
            "target_v12": float(cond[0]),
            "target_v21": float(cond[1]),
            "novelty_mean": float(novelty.mean()),
            "novelty_min": float(novelty.min()),
            "diversity_mean_pairwise": mean_pair_d,
        })

    novelty_arr = np.array(all_novelty)
    ref_p5 = np.percentile(ref_dist, 5)
    frac_near_duplicate = float((novelty_arr < ref_p5).mean())

    return {
        "cvae_ckpt": cvae_ckpt,
        "n_conditions": n_conditions,
        "n_samples_per_condition": n_samples_per_condition,
        "apply_force_periodic": apply_force_periodic,
        "novelty_generated": percentiles(novelty_arr),
        "diversity_within_condition": percentiles(np.array(all_diversity)),
        "reference_real_to_real_nn": percentiles(ref_dist),
        "frac_generated_below_ref_p5": frac_near_duplicate,
        "interpretation": (
            "frac_generated_below_ref_p5 = ti le mau sinh ra co novelty thap hon "
            "ca P5 cua khoang cach hang-xom-gan-nhat GIUA CAC MAU THAT trong "
            "training set - neu ti le nay cao, day la bang chung cu the cho "
            "'gan nhu sao chep' thay vi suy dien cam tinh."
        ),
        "per_condition": per_condition,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cvae-ckpt", type=str,
                         default=os.path.join(PHASE5_DIR, "cvae_v2_finetuned.pt"))
    parser.add_argument("--n-conditions", type=int, default=24)
    parser.add_argument("--n-samples-per-condition", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no-force-periodic", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = run(args.cvae_ckpt, args.n_conditions, args.n_samples_per_condition,
                 device, seed=args.seed, apply_force_periodic=not args.no_force_periodic)

    ckpt_stem = os.path.splitext(os.path.basename(args.cvae_ckpt))[0]
    out_path = args.out or os.path.join(PHASE5_DIR, "reports", f"novelty_diversity_{ckpt_stem}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    n = result["novelty_generated"]
    d = result["diversity_within_condition"]
    r = result["reference_real_to_real_nn"]
    print(f"Checkpoint: {args.cvae_ckpt}")
    print(f"N condition={args.n_conditions} x {args.n_samples_per_condition} mau/condition")
    print(f"  Novelty (sinh ra -> train gan nhat)   : median={n['p50']:.3f}  p5={n['p5']:.3f}  p95={n['p90']:.3f}")
    print(f"  Reference (that -> that gan nhat)     : median={r['p50']:.3f}  p5={r['p5']:.3f}  p95={r['p90']:.3f}")
    print(f"  Diversity (trong cung 1 condition)    : median={d['p50']:.3f}  p5={d['p5']:.3f}  p95={d['p90']:.3f}")
    print(f"  Frac mau sinh ra co novelty < P5(ref) = {result['frac_generated_below_ref_p5']:.3f}"
          f"  (ti le nghi ngo 'gan nhu sao chep')")
    print(f"\nDa luu: {out_path}")


if __name__ == "__main__":
    main()
