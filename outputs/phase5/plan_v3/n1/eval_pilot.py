"""N1 bước 2 - chấm các nhánh pilot bằng thước đo độc lập (plan.md mục N1).

Với mỗi thiết kế (baseline best-of-30, refined, guarded) của mỗi nhánh:
- e_verify: |Δν12| trên chuỗi verify chuẩn (nearest 64→50, FE 50²).
- e_mesh: cùng hình học 50² nhưng FE trên lưới 200² (kron 4×) = hội tụ lưới.
- e_real: hiện thực hóa làm mượt R(200, 0) - độ bền hình học.
- e_shift: trung bình |Δν12| qua 8 hiện thực hóa lệch lưới R(50, s).
- manuf4: liên thông 4 hướng + nét tối thiểu trên ảnh 64².

Chạy: python eval_pilot.py A_l0 D_k1 N1_k4 C_up2
"""

import json
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = "/home/tbm/Documents/Input_SIMP_Analyst"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{ROOT}/pipeline/phase5_cvae")
from manufacturability import check_connectivity  # noqa: E402
from verify_fe import evaluate_density_field, resize_to_fe_grid  # noqa: E402

SHIFTS = [
    (0, 0),
    (0, 0.5),
    (0.5, 0),
    (0.5, 0.5),
    (0.25, 0.25),
    (0.25, 0.75),
    (0.75, 0.25),
    (0.75, 0.75),
]


def realize(img, n, s=(0.0, 0.0), sigma=0.5):
    """Bản numpy của realization.realize_shifted (ảnh nhị phân, β → ∞)."""
    from scipy import ndimage

    N = img.shape[0]
    phi = ndimage.gaussian_filter(img, sigma * N / 50.0, mode="wrap")
    t = (np.arange(n) + 0.5) / n
    Y, X = np.meshgrid(
        (t + s[0] / 50) * N - 0.5, (t + s[1] / 50) * N - 0.5, indexing="ij"
    )
    v = ndimage.map_coordinates(phi, [Y, X], order=1, mode="grid-wrap")
    return (v > 0.5).astype(float)


def score(job):
    """Mọi chỉ số cho 1 ảnh 64². job = (arm, idx, kind, img, target)."""
    arm, idx, kind, img, (t12, t21) = job
    img = (np.asarray(img, float) > 0.5).astype(float)
    d50 = (resize_to_fe_grid(img, 50, 50) > 0.5).astype(float)
    ver = evaluate_density_field(d50)[:2]
    mesh = evaluate_density_field(np.kron(d50, np.ones((4, 4))))[:2]
    real = evaluate_density_field(realize(d50, 200))[:2]
    shf = [evaluate_density_field(realize(d50, 50, s))[0] for s in SHIFTS]
    c4 = check_connectivity(img, connectivity=4)
    # Chế tạo trên đúng ảnh 50² mà FE giải (N2-E1′ so mọi nhánh cùng lưới).
    c50 = check_connectivity(d50, connectivity=4)
    return {
        "arm": arm,
        "idx": idx,
        "kind": kind,
        "e_verify": abs(ver[0] - t12),
        "e_mesh": abs(mesh[0] - t12),
        "e_real": abs(real[0] - t12),
        "e_shift": float(np.mean(np.abs(np.array(shf) - t12))),
        "pair_mesh": abs(mesh[0] - t12) + abs(mesh[1] - t21),
        "manuf4": bool(c4["is_connected"] and c4["min_feature_ok"]),
        "manuf50": bool(c50["is_connected"] and c50["min_feature_ok"]),
    }


def jobs_for(arm):
    """Ảnh baseline / refined / guarded của 1 nhánh."""
    # Nhánh N1 nằm ở n1/, nhánh N2 (E1, E2, ...) ở n2/ cạnh bên.
    path = f"{HERE}/pilot_B20_{arm}.json"
    if not os.path.exists(path):
        path = f"{HERE}/../n2/pilot_B20_{arm}.json"
    d = json.load(open(path))
    out = []
    for i, c in enumerate(d["per_condition"]):
        t = (c["target_v12"], c["target_v21"])
        out.append((arm, i, "baseline", c["baseline_image"], t))
        out.append((arm, i, "refined", c["refined_image"], t))
        g = c["refined_image"] if c["guarded_accept"] else c["baseline_image"]
        out.append((arm, i, "guarded", g, t))
    return out


def boot_ci(diff, n=5000, seed=0):
    """CI95 bootstrap của trung bình hiệu số ghép cặp."""
    rng = np.random.default_rng(seed)
    bs = [rng.choice(diff, len(diff)).mean() for _ in range(n)]
    return np.percentile(bs, [2.5, 97.5])


if __name__ == "__main__":
    arms = sys.argv[1:]
    jobs = [j for a in arms for j in jobs_for(a)]
    with Pool(8) as pool:
        res = pool.map(score, jobs)
    out_name = "pilot_B20_scores_" + "_".join(arms) + ".json"
    json.dump(res, open(f"{HERE}/{out_name}", "w"))
    keys = ["e_verify", "e_mesh", "e_real", "e_shift", "pair_mesh"]
    get = {(r["arm"], r["kind"]): [] for r in res}
    for r in res:
        get[(r["arm"], r["kind"])].append(r)
    print(
        f"{'arm/kind':18s}" + "".join(f"{k:>11s}" for k in keys) + "   manuf4  manuf50"
    )
    for a in arms:
        for kind in ("baseline", "refined", "guarded"):
            x = sorted(get[(a, kind)], key=lambda r: r["idx"])
            print(
                f"{a + '/' + kind:18s}"
                + "".join(f"{np.mean([r[k] for r in x]):11.4f}" for k in keys)
                + f"   {np.mean([r['manuf4'] for r in x]):.2f}"
                + f"     {np.mean([r['manuf50'] for r in x]):.2f}"
            )
    ref = arms[0]
    for a in arms[1:]:
        for kind in ("refined", "guarded"):
            xa = sorted(get[(ref, kind)], key=lambda r: r["idx"])
            xb = sorted(get[(a, kind)], key=lambda r: r["idx"])
            for k in ("e_verify", "e_mesh", "e_real"):
                diff = np.array([q[k] - p[k] for p, q in zip(xa, xb)])
                lo, hi = boot_ci(diff)
                print(
                    f"{a} - {ref} [{kind}] {k}: {diff.mean():+.4f} "
                    f"CI [{lo:+.4f}, {hi:+.4f}]  (âm = {a} tốt hơn)"
                )
