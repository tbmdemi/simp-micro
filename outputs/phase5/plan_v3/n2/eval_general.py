"""Chấm điểm tổng quát nhiều nhánh refine trên bảng điểm N2 (plan.md mục N2).

Mỗi nhánh = 1 file JSON của benchmark_physics_guided_refinement.py (có
--save-images). Với thiết kế guarded và refined của mỗi condition:
- e_verify: |Δν12| trên ảnh FE 50² (chuỗi verify của chính nhánh đó).
- e_mesh: cùng ảnh 50² giải trên lưới 200² (kron 4×).
- e_real05 / e_real10: hiện thực hóa làm mượt σ = 0,5 / 1,0 trên lưới 200²
  (thước đo chính từ 2026-10-10).
- pair_real05: |Δν12| + |Δν21| trên e_real σ = 0,5.
- manuf4: chế tạo (4 hướng + nét tối thiểu) trên ảnh nhánh lưu; manuf50: trên
  ảnh 50² mà FE giải (so mọi nhánh cùng lưới).
- e_shift: trung bình |Δν12| qua 8 phép dịch lệch lưới ở 50² (thước đo độ bền
  độc lập với họ làm mượt của e_real).
- e_ed: trung bình |Δν12| của bản co/giãn (ngưỡng 0,35 / 0,65 trên trường làm
  mượt σ = 0,5, lưới 200²).
File SIMP (benchmark_simp_baseline.py, khóa "full") được nhận tự động: thiết kế
= ảnh 50² tốt nhất của chế độ full, chỉ có loại "guarded".

Chạy:
  python eval_general.py --out scores.json A=path_a.json E1p=path_e.json
Nhánh đầu tiên là tham chiếu cho hiệu ghép cặp + CI bootstrap 95%.
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = "/home/tbm/Documents/Input_SIMP_Analyst"
sys.path.insert(0, f"{ROOT}/pipeline/phase5_cvae")
sys.path.insert(0, f"{ROOT}/outputs/phase5/plan_v3/n1")
from eval_pilot import SHIFTS, realize  # noqa: E402
from manufacturability import check_connectivity  # noqa: E402
from verify_fe import evaluate_density_field, resize_to_fe_grid  # noqa: E402


def realize_eta(img, n, sigma, eta):
    """Như ``realize`` (không dịch) nhưng ngưỡng ``eta`` - bản co/giãn."""
    from scipy import ndimage

    N = img.shape[0]
    phi = ndimage.gaussian_filter(img, sigma * N / 50.0, mode="wrap")
    t = (np.arange(n) + 0.5) / n * N - 0.5
    Y, X = np.meshgrid(t, t, indexing="ij")
    v = ndimage.map_coordinates(phi, [Y, X], order=1, mode="grid-wrap")
    return (v > eta).astype(float)


def score(job):
    """Mọi chỉ số cho 1 ảnh thiết kế. job = (arm, idx, kind, img, target)."""
    arm, idx, kind, img, (t12, t21) = job
    img = (np.asarray(img, float) > 0.5).astype(float)
    d50 = (resize_to_fe_grid(img, 50, 50) > 0.5).astype(float)
    ver = evaluate_density_field(d50)[:2]
    mesh = evaluate_density_field(np.kron(d50, np.ones((4, 4))))[:2]
    r05 = evaluate_density_field(realize(d50, 200, sigma=0.5))[:2]
    r10 = evaluate_density_field(realize(d50, 200, sigma=1.0))[:2]
    shf = [evaluate_density_field(realize(d50, 50, s))[0] for s in SHIFTS]
    ed = [
        evaluate_density_field(realize_eta(d50, 200, 0.5, eta))[0]
        for eta in (0.35, 0.65)
    ]
    c4 = check_connectivity(img, connectivity=4)
    c50 = check_connectivity(d50, connectivity=4)
    return {
        "arm": arm,
        "idx": idx,
        "kind": kind,
        "e_verify": abs(ver[0] - t12),
        "e_mesh": abs(mesh[0] - t12),
        "e_real05": abs(r05[0] - t12),
        "e_real10": abs(r10[0] - t12),
        "pair_real05": abs(r05[0] - t12) + abs(r05[1] - t21),
        "e_shift": float(np.mean(np.abs(np.array(shf) - t12))),
        "e_ed": float(np.mean(np.abs(np.array(ed) - t12))),
        "manuf4": bool(c4["is_connected"] and c4["min_feature_ok"]),
        "manuf50": bool(c50["is_connected"] and c50["min_feature_ok"]),
    }


def jobs_for(arm, path, kinds):
    """Ảnh refined + guarded của 1 nhánh (SIMP: chỉ guarded = ảnh tốt nhất)."""
    out = []
    for i, c in enumerate(json.load(open(path))["per_condition"]):
        t = (c["target_v12"], c["target_v21"])
        if "full" in c:
            out.append((arm, i, "guarded", c["full"]["best"]["image"], t))
            continue
        if "refined" in kinds:
            out.append((arm, i, "refined", c["refined_image"], t))
        g = c["refined_image"] if c["guarded_accept"] else c["baseline_image"]
        out.append((arm, i, "guarded", g, t))
    return out


def boot_ci(diff, n=5000, seed=0):
    """CI95 bootstrap của trung bình hiệu số ghép cặp."""
    rng = np.random.default_rng(seed)
    bs = [rng.choice(diff, len(diff)).mean() for _ in range(n)]
    return np.percentile(bs, [2.5, 97.5])


def main():
    """CLI: chấm, in bảng + hiệu ghép cặp, lưu JSON."""
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="label=path.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--kinds", default="guarded,refined", help="guarded và/hoặc refined"
    )
    ap.add_argument(
        "--cache",
        default=None,
        help="JSON điểm đã chấm (khóa nhãn/idx/loại) - dùng lại, ghi thêm",
    )
    args = ap.parse_args()
    arms = [a.split("=", 1) for a in args.arms]
    kinds = args.kinds.split(",")
    jobs = [j for lab, p in arms for j in jobs_for(lab, p, kinds)]
    cached = {}
    if args.cache and os.path.exists(args.cache):
        for r in json.load(open(args.cache)):
            cached[(r["arm"], r["idx"], r["kind"])] = r
    todo = [j for j in jobs if (j[0], j[1], j[2]) not in cached]
    print(
        f"{len(jobs)} thiết kế ({len(jobs) - len(todo)} có sẵn trong cache), "
        f"{args.workers} tiến trình",
        flush=True,
    )
    res = [
        cached[(j[0], j[1], j[2])]
        for j in jobs
        if (j[0], j[1], j[2]) in cached
    ]
    with Pool(args.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(score, todo, chunksize=2)):
            res.append(r)
            if (k + 1) % 20 == 0:
                print(f"  {k + 1}/{len(todo)}", flush=True)
    json.dump(res, open(args.out, "w"))
    if args.cache:
        for r in res:
            cached[(r["arm"], r["idx"], r["kind"])] = r
        json.dump(list(cached.values()), open(args.cache, "w"))
    keys = [
        "e_verify",
        "e_mesh",
        "e_real05",
        "e_real10",
        "pair_real05",
        "e_shift",
        "e_ed",
    ]
    labels = [lab for lab, _ in arms]
    by = {}
    for r in res:
        by.setdefault((r["arm"], r["kind"]), []).append(r)
    for v in by.values():
        v.sort(key=lambda r: r["idx"])
    print(
        f"{'nhánh/loại':20s}"
        + "".join(f"{k:>12s}" for k in keys)
        + "  manuf4 manuf50"
    )
    for lab in labels:
        for kind in ("refined", "guarded"):
            if (lab, kind) not in by:
                continue
            x = by[(lab, kind)]
            print(
                f"{lab + '/' + kind:20s}"
                + "".join(f"{np.mean([r[k] for r in x]):12.4f}" for k in keys)
                + f"  {np.mean([r['manuf4'] for r in x]):6.2f}"
                + f" {np.mean([r['manuf50'] for r in x]):6.2f}"
            )
    ref = labels[0]
    for lab in labels[1:]:
        for kind in ("guarded", "refined"):
            if (ref, kind) not in by or (lab, kind) not in by:
                continue
            xa, xb = by[(ref, kind)], by[(lab, kind)]
            for k in keys:
                d = np.array([q[k] - p[k] for p, q in zip(xa, xb)])
                lo, hi = boot_ci(d)
                rel = d.mean() / np.mean([p[k] for p in xa])
                print(
                    f"{lab} - {ref} [{kind}] {k}: {d.mean():+.4f} "
                    f"({rel:+.0%}) CI [{lo:+.4f}, {hi:+.4f}]"
                )
            for m in ("manuf4", "manuf50"):
                d = np.array(
                    [float(q[m]) - float(p[m]) for p, q in zip(xa, xb)]
                )
                lo, hi = boot_ci(d)
                print(
                    f"{lab} - {ref} [{kind}] {m}: {d.mean():+.2f} "
                    f"CI [{lo:+.2f}, {hi:+.2f}]"
                )


if __name__ == "__main__":
    main()
