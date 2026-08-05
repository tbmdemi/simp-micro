"""
Backfill f1=E11/E0, f2=E22/E0 (Pha B roadmap gốc, xem docs/PIPELINE.md muc
5) tren toan bo outputs/phase3/manifest.csv (10.270 mau "sach" da dung de
build outputs/phase3/{train,val,test}.npz hien tai, xem
pipeline/phase3_dataset/README.md).

Q = compute_homogenized_tensor(...) da duoc run_simp() tinh o MOI lan chay
FE nhung CHUA TUNG luu xuong dia (chi v12/v21/obj_value duoc giu, xem
pipeline/phase2_multi_batch/runner.py::evaluate_single()). Backfill = chay
LAI 1 FE-solve/anh CUOI da luu (verify_fe.py::evaluate_density_field(),
KHONG viet lai logic FE), dung dung `penal` THAT cua tung mau (co san trong
manifest.csv) thay vi gia tri xap xi 3.0 mac dinh cua verify_fe.py (von
danh cho anh cVAE sinh moi khong co penal goc).

Q_ij da o don vi tuyet doi (KE duoc build voi E0 nhung cong thuc
k_e = E_penal/E0 * KE huy E0 di, xem simp/homogenization/compute.py docstring)
=> f1 = Q[0,0]/E0, f2 = Q[1,1]/E0 (E0 = FE_PARAMS['E0'] = 199.0, hang so vat
lieu co dinh, KHONG doi theo mau).

Cach chay:
    python3 analysis/scripts/backfill_f1_f2.py --workers 8

Output: outputs/phase3/manifest_f1f2.csv (image_path,f1,f2,error)
"""
import os
import sys
import argparse
import multiprocessing as mp

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline", "phase5_cvae"))

PHASE3_DIR = os.path.join(REPO_ROOT, "outputs", "phase3")
MANIFEST_PATH = os.path.join(PHASE3_DIR, "manifest.csv")
OUT_PATH = os.path.join(PHASE3_DIR, "manifest_f1f2.csv")


def _resolve(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)


def _solve_one(args) -> dict:
    image_path, penal = args
    from verify_fe import evaluate_density_field, resize_to_fe_grid, FE_PARAMS
    fe_params = dict(FE_PARAMS)
    fe_params["penal"] = float(penal)
    try:
        im = Image.open(_resolve(image_path)).convert("L")
        img = np.asarray(im, dtype=np.float32) / 255.0
        img_fe = resize_to_fe_grid(img, fe_params["nely"], fe_params["nelx"])
        v12_check, v21_check, Q = evaluate_density_field(img_fe, fe_params)
        f1 = float(Q[0, 0]) / fe_params["E0"]
        f2 = float(Q[1, 1]) / fe_params["E0"]
        return {"image_path": image_path, "f1": f1, "f2": f2,
                "v12_recheck": v12_check, "v21_recheck": v21_check, "error": ""}
    except Exception as e:
        return {"image_path": image_path, "f1": float("nan"), "f2": float("nan"),
                "v12_recheck": float("nan"), "v21_recheck": float("nan"), "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--limit", type=int, default=None, help="Chi chay N mau dau (debug/timing).")
    args = parser.parse_args()

    manifest = pd.read_csv(MANIFEST_PATH)
    if args.limit:
        manifest = manifest.iloc[:args.limit]
    tasks = list(zip(manifest["image_path"], manifest["penal"]))
    print(f"Backfill f1/f2 tren {len(tasks)} mau, {args.workers} worker...")

    results = []
    with mp.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap(_solve_one, tasks, chunksize=16)):
            results.append(r)
            if (i + 1) % 1000 == 0:
                print(f"  ... {i + 1}/{len(tasks)}")

    out = pd.DataFrame(results)
    n_err = (out["error"] != "").sum()
    if n_err:
        print(f"CANH BAO: {n_err} mau loi FE (xem cot 'error'), f1/f2=NaN.")

    # Sanity check: v12/v21 tinh lai tu backfill (dung penal that) phai khop
    # gan dung voi v12/v21 goc trong manifest.csv (cung 1 anh, cung penal,
    # chi khac lan chay resize/PIL - sai so nho la binh thuong).
    merged = manifest.merge(out, on="image_path", how="left")
    ok = merged["error"].fillna("") == ""
    dv12 = (merged.loc[ok, "v12"] - merged.loc[ok, "v12_recheck"]).abs()
    print(f"Sanity check |v12_goc - v12_recheck|: mean={dv12.mean():.6f}, max={dv12.max():.6f}")

    out.to_csv(OUT_PATH, index=False)
    print(f"Da luu: {OUT_PATH} ({len(out)} dong)")


if __name__ == "__main__":
    main()
