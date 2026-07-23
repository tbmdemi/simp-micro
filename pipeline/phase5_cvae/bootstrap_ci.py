"""
Phase 5 - bootstrap_ci.py
============================================================
best_of_n_eval.py bao cao R2(FE that) va hit_rate nhu 1 diem uoc luong don
(point estimate) tren n_conditions rat nho (24, trong do 19 auxetic) - khong
co khoang tin cay di kem. Voi n nho nhu vay, R2=+0.60 va R2=+0.30 co the
khong khac nhau co y nghia thong ke.

Script nay KHONG chay lai FE (khong ton thoi gian/tai nguyen) - chi resample
lai per_condition da luu san trong JSON ket qua cua best_of_n_eval.py:

- R2(v12, FE that): percentile bootstrap tren cac dieu kien (resample dieu
  kien co hoan lai, tinh lai R2 moi lan, lay khoang 2.5%-97.5%). R2 la thong
  ke lien tuc nen bootstrap thong thuong phu hop.
- hit_rate: dung Wilson score interval (KHONG dung bootstrap thuong) - vi
  hit_rate thuong = 1.0 (100%) tren tap 19-24 dieu kien, va bootstrap percentile
  tren 1 ty le nam dung o bien 0/1 se cho khoang suy bien [1.0, 1.0] (moi
  lan resample lai deu ra toan 1), trong khi that ra "19/19 dung" van con
  bat dinh dang ke o n nho. Wilson score la khoang chuan cho ty le nhi phan,
  khong suy bien o p=1.

Cach chay:
    python3 pipeline/phase5_cvae/bootstrap_ci.py \\
        outputs/phase5/self_play/best_of_n_result.json \\
        outputs/phase5/self_play/best_of_n_k10_result.json \\
        outputs/phase5/self_play/best_of_n_manuf_n1500.json
"""
import json
import math
import argparse
import numpy as np


def r2_score(targets: np.ndarray, preds: np.ndarray) -> float:
    """Cung cong thuc voi best_of_n_eval.py::best_of_n (khop 1-1 voi
    r2_fe_v12_best_of_n da luu trong JSON khi tinh tren toan bo per_condition).

    Dung nguong dung sai (khong phai `== 0` tuyet doi) cho targets gan-nhu-
    hang-so: khi bootstrap resample trung lap nhieu lan cung 1 dieu kien (de
    xay ra o n nho nhu 3), `targets - targets.mean()` khong ve dung 0.0 vi
    sai so lam tron dau phay dong (vd 3 gia tri -0.4 het nhau van cho
    ss_tot ~ 1e-33 thay vi 0 hoan toan) - neu so sanh `== 0` se lot ss_tot
    gan-0 nay qua, chia ra 1 R2 khong lo gia thay vi bi loc thanh NaN."""
    ss_res = ((targets - preds) ** 2).sum()
    if np.ptp(targets) < 1e-9:
        return float("nan")
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot)


def bootstrap_r2(per_condition: list, n_boot: int = 10000, seed: int = 0) -> dict:
    targets = np.array([c["target_v12"] for c in per_condition])
    preds = np.array([c["v12_best"] for c in per_condition])
    n = len(targets)
    point = r2_score(targets, preds)

    rng = np.random.default_rng(seed)
    boot_r2 = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_r2[b] = r2_score(targets[idx], preds[idx])
    valid = boot_r2[~np.isnan(boot_r2)]
    lo, hi = np.percentile(valid, [2.5, 97.5]) if len(valid) else (float("nan"), float("nan"))

    return {
        "n_conditions": int(n),
        "r2_point_estimate": point,
        "r2_ci95_lo": float(lo),
        "r2_ci95_hi": float(hi),
        "n_boot_valid": int(len(valid)),
        "n_boot_total": int(n_boot),
    }


def wilson_ci(hits: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval cho 1 ty le nhi phan - khong suy bien o p=0 hoac
    p=1 nhu bootstrap percentile thuong, phu hop cho hit_rate tren n nho."""
    if n == 0:
        return float("nan"), float("nan")
    phat = hits / n
    denom = 1 + z ** 2 / n
    center = (phat + z ** 2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))
    return max(0.0, center - margin), min(1.0, center + margin)


def hit_rate_ci(per_condition: list) -> dict:
    aux = [c for c in per_condition if c["is_auxetic_target"]]
    n = len(aux)
    hits = sum(1 for c in aux if c["v12_best"] < 0)
    point = hits / n if n else float("nan")
    lo, hi = wilson_ci(hits, n)
    return {
        "n_auxetic_conditions": int(n),
        "hit_rate_point_estimate": float(point),
        "hit_rate_wilson_ci95_lo": float(lo),
        "hit_rate_wilson_ci95_hi": float(hi),
    }


def analyze_file(path: str, n_boot: int = 10000, seed: int = 0) -> dict:
    with open(path) as f:
        result = json.load(f)
    per_condition = result["per_condition"]
    stats = bootstrap_r2(per_condition, n_boot=n_boot, seed=seed)
    stats.update(hit_rate_ci(per_condition))
    return stats


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+",
                         help="1+ file JSON ket qua tu best_of_n_eval.py")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None,
                         help="Neu set, ghi toan bo ket qua (tat ca file) ra 1 JSON tong hop.")
    args = parser.parse_args()

    all_results = {}
    for path in args.files:
        stats = analyze_file(path, n_boot=args.n_boot, seed=args.seed)
        all_results[path] = stats
        print(f"\n{path}")
        print(f"  n_conditions = {stats['n_conditions']}, n_auxetic = {stats['n_auxetic_conditions']}")
        print(f"  R2(FE, best-of-N) = {stats['r2_point_estimate']:.4f}   "
              f"95% CI (bootstrap, n={stats['n_boot_valid']}/{stats['n_boot_total']}) = "
              f"[{stats['r2_ci95_lo']:.4f}, {stats['r2_ci95_hi']:.4f}]")
        print(f"  hit_rate          = {stats['hit_rate_point_estimate']:.4f}   "
              f"95% CI (Wilson) = [{stats['hit_rate_wilson_ci95_lo']:.4f}, "
              f"{stats['hit_rate_wilson_ci95_hi']:.4f}]")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nDa ghi {args.out}")


if __name__ == "__main__":
    main()
