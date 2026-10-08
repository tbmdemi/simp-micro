"""P1.10 - kiểm tra suy biến của thiết kế cuối cVAE và SIMP (Bài #1).

Thiết kế suy biến = min(E_x, E_y)/E0 < 1e-3 trên lưới verify 50² (không có
đường truyền lực theo ít nhất 1 phương; ν khi đó vô nghĩa). Áp CÙNG tiêu chí
cho cả 2 phương pháp trên IN100/IN100-B/IN100-C (cVAE không fp best-of-30 +
refine guarded vs SIMP `full` best-of-4) và 3 nhóm OOD đổi dấu (P1.6b).
Phát hiện: SIMP trả về ô RỖNG (mọi phần tử E_min → ν = ν0 = 0,3) ở 3/28 mục
tiêu đổi dấu, nên MAE OOD của SIMP bị "đẹp" giả nếu không kiểm suy biến.

Ghi per-set: chỉ số suy biến, MAE ν12 / sai số cặp trên mọi mục tiêu và trên
các mục tiêu mà cả 2 phương pháp đều không suy biến.

Chạy từ gốc repo (CPU, ~1-2 phút, 10 worker):
    python docs/paper1/scripts/p1_10_degenerate_check.py
Output: outputs/phase5/plan_v3/p1_10_degenerate_check.json
"""

import json, os, sys, numpy as np
from multiprocessing import Pool

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
sys.path.insert(0, ".")
sys.path.insert(0, "pipeline/phase5_cvae")
sys.path.insert(0, "docs/paper1/scripts")
from verify_fe import FE_PARAMS, evaluate_density_field, resize_to_fe_grid
from simp.objectives.auxetic import compute_elastic_constants

P = "outputs/phase5/plan_v3"


def ev(img: np.ndarray) -> float:
    """min(E_x, E_y)/E0 của 1 ảnh nhị phân 50²; 0 nếu FE lỗi (coi là suy biến)."""
    try:
        _, _, Q = evaluate_density_field(img, FE_PARAMS)
        e = compute_elastic_constants(Q)
        return min(e["E_x"], e["E_y"]) / FE_PARAMS["E0"]
    except Exception:
        return 0.0


def cv_img(c: dict) -> np.ndarray:
    """Ảnh 50² nhị phân của thiết kế cuối chế độ guarded (ngưỡng 0,5, resize nearest)."""
    k = "refined_image" if c["guarded_accept"] else "baseline_image"
    return resize_to_fe_grid(
        (np.asarray(c[k], np.float32) > 0.5).astype(np.float32), 50, 50
    )


jobs = []
pairs = {
    "IN100": (
        "p1_7_c5_in100_bo30_nofp.json",
        "p1_6a_simp_baseline_in100.json",
    ),
    "IN100-B": ("p1_9_r5_in100b_bo30_nofp.json", "p1_7_simp_in100b_full.json"),
    "IN100-C": ("p1_9_r6_in100c_bo30_nofp.json", "p1_8_simp_in100c_full.json"),
}
for g in ["sym", "v21_01", "v21_03"]:
    pairs["SF_" + g] = (
        f"p1_9_r7_signflip_{g}_nofp.json",
        f"p1_6b_simp_{g}.json",
    )
meta = []
for name, (cf, sf) in pairs.items():
    cv = json.load(open(os.path.join(P, cf)))["per_condition"]
    sp = json.load(open(os.path.join(P, sf)))["per_condition"]
    for i, (c, s) in enumerate(zip(cv, sp)):
        T = np.array([c["target_v12"], c["target_v21"]])
        g = (
            np.array([c["refined_v12"], c["refined_v21"]])
            if c["guarded_accept"]
            else np.array([c["baseline_v12"], c["baseline_v21"]])
        )
        jobs.append(cv_img(c))
        meta.append((name, i, "cvae", T, g, None))
        b = s["full"]["best"]
        jobs.append(np.asarray(b["image"], np.float32))
        meta.append(
            (
                name,
                i,
                "simp",
                T,
                np.array([b["v12"], b["v21"]]),
                s["full"]["per_seed"],
            )
        )
with Pool(10) as p:
    E = p.map(ev, jobs)
out = {}
for (name, i, m, T, v, ps), e in zip(meta, E):
    out.setdefault(name, {}).setdefault(m, []).append(
        {"i": i, "T": T.tolist(), "v": v.tolist(), "Emin": e, "per_seed": ps}
    )
res = {}
for name, d in out.items():
    r = {}
    for m, rows in d.items():
        deg = [x["i"] for x in rows if x["Emin"] < 1e-3]
        T = np.array([x["T"] for x in rows])
        V = np.array([x["v"] for x in rows])
        ok = np.array([x["Emin"] >= 1e-3 for x in rows])
        r[m] = {
            "n": len(rows),
            "degenerate_idx": deg,
            "mae12_all": float(np.abs(V[:, 0] - T[:, 0]).mean()),
            "pair_all": float(np.abs(V - T).sum(1).mean()),
        }
    both = [
        a["Emin"] >= 1e-3 and b["Emin"] >= 1e-3
        for a, b in zip(d["cvae"], d["simp"])
    ]
    for m, rows in d.items():
        T = np.array([x["T"] for x in rows])
        V = np.array([x["v"] for x in rows])
        both_ = np.array(both)
        r[m]["mae12_both_ok"] = float(np.abs(V[both_, 0] - T[both_, 0]).mean())
        r[m]["pair_both_ok"] = float(np.abs(V[both_] - T[both_]).sum(1).mean())
    r["n_both_ok"] = int(sum(both))
    # SIMP với degenerate gán như thất bại: chọn seed không suy biến tốt nhất? (per_seed có volume)
    res[name] = r
    print(
        name,
        json.dumps(
            {
                k: (
                    v
                    if not isinstance(v, dict)
                    else {kk: vv for kk, vv in v.items()}
                )
                for k, v in r.items()
            }
        ),
    )
json.dump(
    {
        "rule": "degenerate if min(E_x,E_y)/E0<1e-3 at 50^2",
        "results": res,
        "simp_per_seed_keys": (
            list(out["SF_sym"]["simp"][0]["per_seed"][0].keys())
            if isinstance(out["SF_sym"]["simp"][0]["per_seed"], list)
            else str(type(out["SF_sym"]["simp"][0]["per_seed"]))
        ),
    },
    open(os.path.join(P, "p1_10_degenerate_check.json"), "w"),
    indent=1,
)
