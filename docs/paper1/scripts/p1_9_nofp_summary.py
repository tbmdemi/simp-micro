"""P1.9 - tổng hợp mọi số của bài trên pipeline KHÔNG force_periodic (C5).

Đọc các lần chạy lại `outputs/phase5/plan_v3/p1_9_*` (và các lần chạy vốn
không fp: single-shot P1.9a, best-of-30 nhận thức nhị phân hóa của ablation
C5), bỏ qua file chưa có. Ghi `p1_9_nofp_summary.json` và in tóm tắt:
  * Bảng 1: single-shot / best-of-30 × không refine / liên tục / nhận thức
    nhị phân hóa / guarded - R²(ν12), MAE ν12, R²(ν21), ΔMAE [CI].
  * Bảng OOD biên độ: sai số tương đối ν12 trung vị theo mục tiêu.
  * Bảng SIMP: cVAE (không fp) vs SIMP `full` trên IN100/IN100-B/IN100-C,
    kèm phân tầng r < 10.
  * OOD đổi dấu cVAE (không fp) vs SIMP.
  * Bảng 2: best-of-30 n=300 (chọn thuần độ chính xác / composite) và
    Spearman composite ↔ Pareto.
Chế tạo được = liên thông + nét tối thiểu (định nghĩa C5).

Chạy từ gốc repo:
    python docs/paper1/scripts/p1_9_nofp_summary.py
"""

import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline", "phase5_cvae"))

from bootstrap_ci import bootstrap_paired_mae_reduction  # noqa: E402
from manufacturability import check_connectivity  # noqa: E402

P = os.path.join(REPO_ROOT, "outputs", "phase5", "plan_v3")


def load(name: str):
    """per_condition (+summary, config) của 1 file, None nếu chưa có."""
    path = os.path.join(P, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def r2(pred, target) -> float:
    """R² = 1 − SS_res/SS_tot."""
    pred, target = np.asarray(pred, float), np.asarray(target, float)
    return float(
        1
        - ((pred - target) ** 2).sum() / ((target - target.mean()) ** 2).sum()
    )


def arrays(pc: list) -> dict:
    """Target, baseline, refined, guarded (n, 2) từ per_condition harness."""
    T = np.array([[c["target_v12"], c["target_v21"]] for c in pc])
    B = np.array([[c["baseline_v12"], c["baseline_v21"]] for c in pc], float)
    R = np.array([[c["refined_v12"], c["refined_v21"]] for c in pc], float)
    if "guarded_accept" in pc[0]:
        acc = np.array([c["guarded_accept"] for c in pc])
    else:
        # JSON cũ (trước khi harness ghi cờ): tính lại đúng quy tắc guarded
        # của harness - nhận refine khi sai số cặp verify nhỏ hơn baseline.
        acc = np.abs(R - T).sum(1) < np.abs(B - T).sum(1)
    G = np.where(acc[:, None], R, B)
    return {"T": T, "base": B, "ref": R, "guard": G, "acc": acc}


def manuf(pc: list, key: str) -> float:
    """Tỉ lệ liên thông + nét tối thiểu trên ảnh `key`_image (guard = theo
    cờ guarded_accept); None nếu file không lưu ảnh."""
    if "baseline_image" not in pc[0]:
        return None
    vals = []
    for c in pc:
        k = key
        if key == "guard":
            k = "refined" if c["guarded_accept"] else "baseline"
        vals.append(
            check_connectivity(np.asarray(c[f"{k}_image"]))["manufacturable"]
        )
    return float(np.mean(vals))


def mode_row(a: dict, mode: str, ref_mode: str = "base") -> dict:
    """R², MAE và ΔMAE ν12 (CI ghép cặp so với baseline cùng lần chạy)."""
    P_, T = a[mode], a["T"]
    row = {
        "r2_v12": r2(P_[:, 0], T[:, 0]),
        "r2_v21": r2(P_[:, 1], T[:, 1]),
        "mae_v12": float(np.abs(P_[:, 0] - T[:, 0]).mean()),
        "pair_mae": float(np.abs(P_ - T).sum(1).mean()),
    }
    if mode != ref_mode:
        ci = bootstrap_paired_mae_reduction(
            np.abs(a[ref_mode][:, 0] - T[:, 0]), np.abs(P_[:, 0] - T[:, 0])
        )
        row["dmae_v12"] = [
            ci["rel_reduction"],
            ci["rel_reduction_ci95_lo"],
            ci["rel_reduction_ci95_hi"],
        ]
    return row


def table1() -> dict:
    """Bảng 1 trên pipeline không fp."""
    out = {}
    runs = {
        "single_cont": "p1_1a_v2_finetuned_in100.json",
        "single_aware": "p1_9_in100_single_nofp_images.json",
        "bo30_cont": "p1_9_r1_in100_bo30_cont_nofp.json",
        "bo30_aware": "p1_7_c5_in100_bo30_nofp.json",
    }
    for name, fname in runs.items():
        d = load(fname)
        if d is None:
            out[name] = "missing"
            continue
        a = arrays(d["per_condition"])
        out[name] = {
            "none": mode_row(a, "base"),
            "refined": mode_row(a, "ref"),
            "guarded": mode_row(a, "guard"),
            "manuf_none": manuf(d["per_condition"], "baseline"),
            "manuf_guarded": manuf(d["per_condition"], "guard"),
        }
    return out


def ood_magnitude() -> dict:
    """Sai số tương đối ν12 trung vị theo mục tiêu (10 lặp/mục tiêu)."""
    out = {}
    for name, fname in (
        ("cont", "p1_9_r3_ood_cont_nofp.json"),
        ("aware", "p1_9_r4_ood_aware_nofp.json"),
    ):
        d = load(fname)
        if d is None:
            out[name] = "missing"
            continue
        a = arrays(d["per_condition"])
        t = a["T"][:, 0]
        res = {}
        for tv in sorted(set(np.round(t, 4)), reverse=True):
            m = np.isclose(t, tv)
            rel = lambda X: float(
                np.median(np.abs(X[m, 0] - tv) / abs(tv))
            )  # noqa: E731
            res[str(tv)] = {
                "none": rel(a["base"]),
                "refined": rel(a["ref"]),
                "guarded": rel(a["guard"]),
            }
        out[name] = res
    return out


def simp_table() -> dict:
    """cVAE không fp (guarded) vs SIMP `full`, kèm phân tầng r < 10."""
    sets = {
        "IN100": (
            "p1_7_c5_in100_bo30_nofp.json",
            "p1_6a_simp_baseline_in100.json",
        ),
        "IN100-B": (
            "p1_9_r5_in100b_bo30_nofp.json",
            "p1_7_simp_in100b_full.json",
        ),
        "IN100-C": (
            "p1_9_r6_in100c_bo30_nofp.json",
            "p1_8_simp_in100c_full.json",
        ),
    }
    out = {}
    for name, (cf, sf) in sets.items():
        cv, sp = load(cf), load(sf)
        if cv is None or sp is None:
            out[name] = "missing"
            continue
        a = arrays(cv["per_condition"])
        T, G = a["T"], a["guard"]
        S = np.array(
            [
                [s["full"]["best"]["v12"], s["full"]["best"]["v21"]]
                for s in sp["per_condition"]
            ]
        )
        assert np.allclose(
            T[:, 0], [s["target_v12"] for s in sp["per_condition"]], atol=1e-6
        )
        r = np.maximum(T[:, 1] / T[:, 0], T[:, 0] / T[:, 1])
        m = r < 10
        eG, eS = np.abs(G - T), np.abs(S - T)
        ci_p = bootstrap_paired_mae_reduction(eS[m].sum(1), eG[m].sum(1))
        ci_v = bootstrap_paired_mae_reduction(eS[m, 0], eG[m, 0])
        out[name] = {
            "cvae": {
                "r2_v12": r2(G[:, 0], T[:, 0]),
                "r2_v21": r2(G[:, 1], T[:, 1]),
                "pair_mae": float(eG.sum(1).mean()),
                "manuf": manuf(cv["per_condition"], "guard"),
            },
            "simp": {
                "r2_v12": r2(S[:, 0], T[:, 0]),
                "r2_v21": r2(S[:, 1], T[:, 1]),
                "pair_mae": float(eS.sum(1).mean()),
            },
            "n_r_lt_10": int(m.sum()),
            "r_lt_10_pair": [
                ci_p["rel_reduction"],
                ci_p["rel_reduction_ci95_lo"],
                ci_p["rel_reduction_ci95_hi"],
            ],
            "r_lt_10_v12": [
                ci_v["rel_reduction"],
                ci_v["rel_reduction_ci95_lo"],
                ci_v["rel_reduction_ci95_hi"],
            ],
            "r_ge_10_pair": [
                [float(x), float(y)]
                for x, y in zip(eG[~m].sum(1), eS[~m].sum(1))
            ],
        }
    return out


def signflip() -> dict:
    """OOD đổi dấu: MAE ν12 và sai số cặp, cVAE không fp (guarded) vs SIMP."""
    out = {}
    for g in ("sym", "v21_01", "v21_03"):
        cv, sp = load(f"p1_9_r7_signflip_{g}_nofp.json"), load(
            f"p1_6b_simp_{g}.json"
        )
        if cv is None or sp is None:
            out[g] = "missing"
            continue
        a = arrays(cv["per_condition"])
        S = np.array(
            [
                [s["full"]["best"]["v12"], s["full"]["best"]["v21"]]
                for s in sp["per_condition"]
            ]
        )
        out[g] = {
            "cvae_mae_v12": float(
                np.abs(a["guard"][:, 0] - a["T"][:, 0]).mean()
            ),
            "cvae_pair": float(np.abs(a["guard"] - a["T"]).sum(1).mean()),
            "simp_mae_v12": float(np.abs(S[:, 0] - a["T"][:, 0]).mean()),
            "simp_pair": float(np.abs(S - a["T"]).sum(1).mean()),
            "cvae_manuf": manuf(cv["per_condition"], "guard"),
        }
    return out


def table2() -> dict:
    """Bảng 2: best-of-30 n=300 và Spearman composite ↔ Pareto (không fp)."""
    out = {}
    for name, fname in (
        ("acc_only", "p1_9_r8_bon300_nofp_acconly.json"),
        ("composite", "p1_9_bon300_nofp.json"),
    ):
        d = load(fname)
        out[name] = (
            "missing"
            if d is None
            else {
                "r2_v12": d["r2_fe_v12_best_of_n"],
                "hit_rate_single": d["hit_rate_single_shot"],
                "mean_frac_manufacturable": d["mean_frac_manufacturable"],
            }
        )
    d = load("p1_9_pareto_nofp.json")
    out["pareto"] = (
        "missing"
        if d is None
        else {
            k: d[k]
            for k in (
                "mean_spearman",
                "median_spearman",
                "min_spearman",
                "max_spearman",
                "frac_positive",
            )
        }
    )
    return out


def main():
    """Tính mọi bảng có đủ dữ liệu, ghi JSON và in."""
    out = {
        "table1": table1(),
        "ood_magnitude": ood_magnitude(),
        "simp": simp_table(),
        "signflip": signflip(),
        "table2": table2(),
    }
    path = os.path.join(P, "p1_9_nofp_summary.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps(out, indent=1, default=lambda x: round(float(x), 4)))
    print("wrote", path)


if __name__ == "__main__":
    main()
