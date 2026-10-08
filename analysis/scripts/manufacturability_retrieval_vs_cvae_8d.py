"""
Manufacturability/connectivity: retrieval vs cVAE o 8D
========================================================
Bo sung cho Nhom 4.2 (docs/archive/PROJECT_PLAN.md): curse_of_dimensionality_
comparison.py da do accuracy (R2) o 8D va thay retrieval (0.998) gan ngang
cVAE da tune dung (0.997) - khong con khoang cach accuracy de lam luan diem
trung tam. Cau hoi moi: retrieval co "hy sinh" chat luong cau truc (lien
thong, kich thuoc net in, doi xung) de dat duoc accuracy do khong, trong khi
cVAE (co force_periodic() + composite scoring uu tien manuf/aesthetic) giu
duoc ca hai?

LUU Y VE GIA THUYET: retrieval KHONG sinh cau truc moi - no tra ve NGUYEN
XI 1 anh THAT da co san trong train.npz (da duoc SIMP toi uu that). Khoang
cach z-score xa KHONG lam "sup do" cau truc do - no chi khien mau duoc chon
khop kem hon ve dieu kien, khong anh huong gi den chinh cau truc. Theo
LIMITATIONS.md, ty le manufacturable tu nhien cua toan dataset chi ~25-35%
(khong loc) - nhieu kha nang retrieval se cho ty le THAP nhung ON DINH DEU
(khong "sup do" o 8D), khac voi gia thuyet "cang xa z-score cang te".
Script nay DO THAT, khong gia dinh truoc ket qua.

Thiet ke: CUNG 24 target (seed=123) va CUNG checkpoint 8D
(cvae_a4_full8d_finetuned_v2.pt, da tune dung bang --select-by fe_r2) da
dung o curse_of_dimensionality_comparison.py - tai dung ket qua cVAE da co
san trong outputs/phase5/reports/curse_of_dimensionality_comparison.json
(khong chay lai model, tiet kiem thoi gian). Phan MOI: retrieval - lay anh
THAT cua nearest-neighbor trong khong gian 8D z-score, ap dung DUNG pipeline
hau xu ly nhu cVAE (force_periodic() truoc khi cham manufacturability/
aesthetic, dung ham co san trong manufacturability.py/aesthetics.py,
KHONG viet lai) de so sanh cong bang.

Cach chay (can conda env 'simp'):
    /home/tbm/miniconda3/envs/simp/bin/python3 \\
        analysis/scripts/manufacturability_retrieval_vs_cvae_8d.py

Output: outputs/phase5/reports/manufacturability_retrieval_vs_cvae_8d.json
"""

import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline", "phase5_cvae"))
sys.path.insert(0, os.path.dirname(__file__))

from manufacturability import check_manufacturability, force_periodic  # noqa: E402
from aesthetics import aesthetic_score as compute_aesthetic_score  # noqa: E402
from curse_of_dimensionality_comparison import _select_targets  # noqa: E402

PHASE3_A4_DIR = os.path.join(REPO_ROOT, "outputs", "phase3_a4")
PHASE5_DIR = os.path.join(REPO_ROOT, "outputs", "phase5")
CVAE_RESULT_PATH = os.path.join(
    PHASE5_DIR, "reports", "curse_of_dimensionality_comparison.json"
)


def retrieval_manufacturability_8d(targets: list, n_conditions: int, seed: int):
    """Voi moi target, tim nearest-neighbor THAT trong khong gian 8D
    z-score (v12,v21,volfrac,void_size_frac,nu0) - dung CUNG cong thuc
    chuan hoa da dung trong curse_of_dimensionality_comparison.py de nhat
    quan - roi cham manufacturability/aesthetic tren CHINH anh that do
    (KHONG sinh anh moi), ap dung force_periodic() truoc de cong bang voi
    pipeline cVAE."""
    train = np.load(os.path.join(PHASE3_A4_DIR, "train.npz"), allow_pickle=True)
    param_names = list(train["param_names"])
    void_idx = param_names.index("void_size_frac")
    images = train["images"]
    train_arrays = {
        "v12": train["v12"].astype(np.float64),
        "v21": train["v21"].astype(np.float64),
        "nu0": train["nu"].astype(np.float64),
        "volfrac": train["volfrac_achieved"].astype(np.float64),
        "void_size_frac": train["params"][:, void_idx].astype(np.float64),
    }
    feature_names = ["v12", "v21", "volfrac", "void_size_frac", "nu0"]
    means = {k: float(train_arrays[k].mean()) for k in feature_names}
    stds = {k: float(train_arrays[k].std()) for k in feature_names}
    train_z = np.stack(
        [(train_arrays[k] - means[k]) / stds[k] for k in feature_names], axis=1
    )

    per_condition = []
    for t in targets:
        cond_z = np.array([(t[k] - means[k]) / stds[k] for k in feature_names])
        d2 = ((train_z - cond_z) ** 2).sum(axis=1)
        j = int(np.argmin(d2))

        img = force_periodic(images[j].astype(np.float32))
        img_bin = (img > 0.5).astype(np.float32)
        manuf = check_manufacturability(img_bin)
        aesthetic = compute_aesthetic_score(img, img_bin)
        manuf_score = (
            manuf.get("is_connected", manuf["passes_all"])
            + manuf.get("min_feature_ok", manuf["passes_all"])
            + manuf.get("periodic_ok", manuf["passes_all"])
        ) / 3.0

        per_condition.append({
            "target_v12": t["v12"],
            "nearest_train_idx": j,
            "condition_dist_z": float(np.sqrt(d2[j])),
            "passes_all": bool(manuf["passes_all"]),
            "manuf_score": float(manuf_score),
            "aesthetic_score": float(aesthetic),
        })
    return per_condition


def main():
    with open(CVAE_RESULT_PATH) as f:
        cvae_result = json.load(f)
    n_conditions = cvae_result["n_conditions"]
    seed = cvae_result["seed"]
    targets = cvae_result["targets"]
    cvae_pc = cvae_result["space_8d"]["cvae_per_condition"]

    retrieval_pc = retrieval_manufacturability_8d(targets, n_conditions, seed)

    # frac_manufacturable cua cVAE la ty le TRONG POOL N mau/condition (co
    # san trong best_of_n output) - dung truc tiep, khong suy dien lai.
    cvae_frac_manufacturable_pool = np.mean(
        [c["frac_manufacturable"] for c in cvae_pc]
    )
    cvae_mean_manuf_score = np.mean([c["manuf_score"] for c in cvae_pc])
    cvae_mean_aesthetic = np.mean([c["aesthetic_score"] for c in cvae_pc])
    cvae_mean_composite = np.mean([c["composite_score"] for c in cvae_pc])

    retr_frac_passes_all = np.mean([c["passes_all"] for c in retrieval_pc])
    retr_mean_manuf_score = np.mean([c["manuf_score"] for c in retrieval_pc])
    retr_mean_aesthetic = np.mean([c["aesthetic_score"] for c in retrieval_pc])

    out = {
        "n_conditions": n_conditions,
        "seed": seed,
        "cvae_ckpt": cvae_result["space_8d"]["ckpt"],
        "cvae_per_condition": cvae_pc,
        "retrieval_per_condition": retrieval_pc,
        "summary": {
            "cvae_mean_frac_manufacturable_in_pool": float(
                cvae_frac_manufacturable_pool
            ),
            "cvae_mean_manuf_score": float(cvae_mean_manuf_score),
            "cvae_mean_aesthetic_score": float(cvae_mean_aesthetic),
            "cvae_mean_composite_score": float(cvae_mean_composite),
            "retrieval_frac_passes_all_strict": float(retr_frac_passes_all),
            "retrieval_mean_manuf_score": float(retr_mean_manuf_score),
            "retrieval_mean_aesthetic_score": float(retr_mean_aesthetic),
        },
    }

    out_path = os.path.join(
        PHASE5_DIR, "reports", "manufacturability_retrieval_vs_cvae_8d.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"N targets = {n_conditions}\n")
    print(f"{'Phuong an':<28}{'manuf_score':>13}{'aesthetic':>12}")
    print(
        f"{'Retrieval (anh that)':<28}{retr_mean_manuf_score:>13.3f}"
        f"{retr_mean_aesthetic:>12.3f}"
    )
    print(
        f"{'cVAE (8D, tuned)':<28}{cvae_mean_manuf_score:>13.3f}"
        f"{cvae_mean_aesthetic:>12.3f}"
    )
    print(f"\nRetrieval frac passes_all (nhi phan, nghiem ngat): {retr_frac_passes_all:.3f}")
    print(f"cVAE mean frac_manufacturable trong pool N mau/condition: {cvae_frac_manufacturable_pool:.3f}")
    print(f"\nDa luu: {out_path}")


if __name__ == "__main__":
    main()
