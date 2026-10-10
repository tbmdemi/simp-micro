"""Kiểm tính vòng lặp của N1: chấm lại bằng hiện thực hóa có σ khác σ tối ưu
(0,5) và lưới 200² - nếu N1 chỉ thắng ở đúng σ=0,5 thì là overfit thước đo."""
import sys
from multiprocessing import Pool
import numpy as np
from eval_pilot import realize, jobs_for
sys.path.insert(0, "/home/tbm/Documents/Input_SIMP_Analyst/pipeline/phase5_cvae")
from verify_fe import evaluate_density_field, resize_to_fe_grid

SIGMAS = [0.25, 0.75, 1.0]

def score(job):
    arm, idx, kind, img, (t12, t21) = job
    d50 = (resize_to_fe_grid((np.asarray(img, float) > 0.5).astype(float), 50, 50) > 0.5).astype(float)
    return {"arm": arm, "idx": idx, "kind": kind,
            **{f"s{s}": abs(evaluate_density_field(realize(d50, 200, sigma=s))[0] - t12) for s in SIGMAS}}

if __name__ == "__main__":
    arms = sys.argv[1:]
    jobs = [j for a in arms for j in jobs_for(a) if j[2] in ("refined", "guarded")]
    with Pool(8) as p:
        res = p.map(score, jobs)
    rng = np.random.default_rng(0)
    for kind in ("refined", "guarded"):
      print(f"--- {kind}")
      base = sorted([r for r in res if r["arm"] == arms[0] and r["kind"] == kind], key=lambda r: r["idx"])
      for a in arms:
        x = sorted([r for r in res if r["arm"] == a and r["kind"] == kind], key=lambda r: r["idx"])
        line = f"{a:12s}"
        for s in SIGMAS:
            d = np.array([q[f"s{s}"] - p[f"s{s}"] for p, q in zip(base, x)])
            bs = [rng.choice(d, len(d)).mean() for _ in range(3000)]
            line += f" | σ={s}: {np.mean([r[f's{s}'] for r in x]):.4f} (Δ {d.mean():+.4f} [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}])"
        print(line)
