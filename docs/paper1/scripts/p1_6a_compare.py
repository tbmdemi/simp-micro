"""So sánh ghép cặp SIMP từ đầu vs pipeline cVAE trên IN100 (P1.6a)."""

# Chạy từ gốc repo: python docs/paper1/scripts/p1_6a_compare.py
# Ghi outputs/phase5/plan_v3/p1_6a_comparison.json (nguồn số cho bảng SIMP trong bài).
import json
import sys

import numpy as np

sys.path.insert(0, "pipeline/phase5_cvae")
from bootstrap_ci import bootstrap_paired_mae_reduction  # noqa: E402

D = "outputs/phase5/plan_v3/"
simp = json.load(open(D + "p1_6a_simp_baseline_in100.json"))["per_condition"]
ea = json.load(open(D + "p1_1e_a_v2_finetuned_in100.json"))["per_condition"]
eb = json.load(open(D + "p1_1e_b_v2_finetuned_in100_bo30.json"))[
    "per_condition"
]
assert all(
    abs(s["target_v12"] - a["target_v12"]) < 1e-9 for s, a in zip(simp, ea)
)
T = np.array([[c["target_v12"], c["target_v21"]] for c in ea])


def r2(p, t):
    return 1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()


def arr(rows):
    return np.array(
        [[np.nan if v is None else v for v in r] for r in rows], float
    )


modes = {
    "cVAE single": (
        arr([(c["baseline_v12"], c["baseline_v21"]) for c in ea]),
        1,
    ),
    "cVAE single+refine": (
        arr([(c["refined_v12"], c["refined_v21"]) for c in ea]),
        78,
    ),
    "cVAE bo30": (
        arr([(c["baseline_v12"], c["baseline_v21"]) for c in eb]),
        30,
    ),
    "cVAE bo30+refine guarded": (
        arr(
            [
                (
                    (c["refined_v12"], c["refined_v21"])
                    if c["guarded_accept"]
                    else (c["baseline_v12"], c["baseline_v21"])
                )
                for c in eb
            ]
        ),
        107,
    ),
}
for bud in ("matched", "full"):
    best = [c[bud]["best"] for c in simp]
    modes[f"SIMP {bud} best-of-4"] = (
        arr([(b["v12"], b["v21"]) for b in best]),
        np.mean([c[bud]["n_fe_total"] for c in simp]),
    )
    hg = [
        next(r for r in c[bud]["per_seed"] if r["seed"] == "hourglass")
        for c in simp
    ]
    modes[f"SIMP {bud} hourglass only"] = (
        arr([(r["v12"], r["v21"]) for r in hg]),
        np.mean([r["n_fe"] for r in hg]),
    )

print(
    f"{'mode':28s} {'R2v12':>7s} {'MAEv12':>7s} {'R2v21':>7s} {'MAEv21':>7s} {'FE':>6s} {'fail':>4s}"
)
for k, (P, fe) in modes.items():
    ok = ~np.isnan(P).any(1)
    print(
        f"{k:28s} {r2(P[ok,0],T[ok,0]):7.3f} {np.abs(P[ok,0]-T[ok,0]).mean():7.4f} "
        f"{r2(P[ok,1],T[ok,1]):7.3f} {np.abs(P[ok,1]-T[ok,1]).mean():7.4f} {fe:6.0f} {(~ok).sum():4d}"
    )

for bud in ("matched", "full"):
    best = [c[bud]["best"] for c in simp]
    man = np.mean([b["manufacturable"] for b in best])
    allman = np.mean(
        [r["manufacturable"] for c in simp for r in c[bud]["per_seed"]]
    )
    t = np.mean([c[bud]["elapsed_total_s"] for c in simp])
    seeds = {}
    for b in best:
        seeds[b["seed"]] = seeds.get(b["seed"], 0) + 1
    print(
        f"SIMP {bud}: manuf(best)={man:.2f} manuf(all runs)={allman:.2f} "
        f"CPU s/target={t:.0f} winning seeds={seeds}"
    )
    gap = [
        abs(r["raw_v12"] - T[i, 0]) + abs(r["raw_v21"] - T[i, 1])
        for i, c in enumerate(simp)
        for r in c[bud]["per_seed"]
    ]
    ver = [
        (
            abs(r["v12"] - T[i, 0]) + abs(r["v21"] - T[i, 1])
            if r["v12"] is not None
            else np.nan
        )
        for i, c in enumerate(simp)
        for r in c[bud]["per_seed"]
    ]
    print(
        f"   median grey err={np.median(gap):.2e}  median verified err={np.nanmedian(ver):.2e}"
    )

ref = modes["cVAE bo30+refine guarded"][0]
for k in (
    "SIMP matched best-of-4",
    "SIMP full best-of-4",
    "SIMP matched hourglass only",
):
    P = modes[k][0]
    ok = ~np.isnan(P).any(1)
    e_s = np.abs(P[ok, 0] - T[ok, 0])
    e_c = np.abs(ref[ok, 0] - T[ok, 0])
    ci = bootstrap_paired_mae_reduction(e_s, e_c)
    print(
        f"cVAE guarded vs {k}: ",
        {
            kk: (round(v, 3) if isinstance(v, float) else v)
            for kk, v in ci.items()
        },
    )

print("\n--- joint error |dv12|+|dv21| ---")
for k in modes:
    P = modes[k][0]
    print(
        f"{k:28s} pairMAE={np.abs(P-T).sum(1).mean():.4f} median={np.median(np.abs(P-T).sum(1)):.4f}"
    )
for k in (
    "SIMP matched best-of-4",
    "SIMP full best-of-4",
    "SIMP matched hourglass only",
):
    P = modes[k][0]
    ci = bootstrap_paired_mae_reduction(
        np.abs(P - T).sum(1), np.abs(ref - T).sum(1)
    )
    print(
        f"joint: cVAE guarded vs {k}: rel_red={ci['rel_reduction']:.3f} [{ci['rel_reduction_ci95_lo']:.3f}, {ci['rel_reduction_ci95_hi']:.3f}]"
    )

summary = {}
for k, (P, fe) in modes.items():
    e = np.abs(P - T)
    summary[k] = {
        "r2_v12": float(r2(P[:, 0], T[:, 0])),
        "r2_v21": float(r2(P[:, 1], T[:, 1])),
        "mae_v12": float(e[:, 0].mean()),
        "mae_v21": float(e[:, 1].mean()),
        "pair_mae": float(e.sum(1).mean()),
        "pair_median": float(np.median(e.sum(1))),
        "fe_calls_per_target": float(fe),
    }
for k in (
    "SIMP matched best-of-4",
    "SIMP full best-of-4",
    "SIMP matched hourglass only",
):
    P = modes[k][0]
    summary[k]["vs_cvae_guarded_v12"] = bootstrap_paired_mae_reduction(
        np.abs(P - T)[:, 0], np.abs(ref - T)[:, 0]
    )
    summary[k]["vs_cvae_guarded_pair"] = bootstrap_paired_mae_reduction(
        np.abs(P - T).sum(1), np.abs(ref - T).sum(1)
    )
for bud in ("matched", "full"):
    best = [c[bud]["best"] for c in simp]
    summary[f"SIMP {bud} best-of-4"]["manufacturable_best"] = float(
        np.mean([b["manufacturable"] for b in best])
    )
    summary[f"SIMP {bud} best-of-4"]["cpu_s_per_target"] = float(
        np.mean([c[bud]["elapsed_total_s"] for c in simp])
    )
json.dump(
    summary, open(D + "p1_6a_comparison.json", "w"), indent=1, default=float
)
print("wrote", D + "p1_6a_comparison.json")
