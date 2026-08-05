"""
Backfill f1=E11/E0, f2=E22/E0 (Pha B roadmap goc) TRUC TIEP tren anh da luu
trong outputs/phase3/{train,val,test}.npz - KHONG join qua manifest.csv.

Ly do doi huong (xem EXPERIMENT_LOG.md 2026-08-05): manifest.csv (10.269
dong) KHONG khop so luong voi dataset_64.npz/train+val+test.npz thuc te
(13.624 mau truoc augment, suy tu 57216/6 train + 2044 val + 2044 test) -
join theo image_path se rui ro sai lech ma khong phat hien duoc. Giai phap
an toan: chay lai FE THANG tren chinh anh (64x64) da luu trong npz, dung
`penal` THAT cua tung mau (params[:,1], param_names=[volfrac,penal,rmin,
move,void_size_frac]) - khong can join/khop thu tu voi bat ky file nao
khac. Toi 6x rieng tren train (vi augment x6 tao ban sao xoay/lat cua cung
1 mau goc) nhung van du re (~0,1s/solve) va AN TOAN hon nhieu so voi suy
luan hoan doi f1<->f2 tay theo logic augment_symmetry.py.

Sanity check tich hop: so v12/v21 tinh lai voi v12/v21 da luu san trong npz
(cung anh, cung penal) - phai khop gan dung, xac nhan khong co landmine
join/thu tu nao khac dang an giau.

Cach chay:
    python3 analysis/scripts/backfill_f1_f2_npz.py --split train --workers 12
    python3 analysis/scripts/backfill_f1_f2_npz.py --split val --workers 12
    python3 analysis/scripts/backfill_f1_f2_npz.py --split test --workers 12

Output: outputs/phase3/{split}_f1f2.npz (chi f1,f2 + index, KHONG copy lai
toan bo anh - ghep voi {split}.npz goc bang chinh thu tu hang, xem
load_with_f1f2() de dung an toan).
"""
import os
import sys
import argparse
import multiprocessing as mp

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline", "phase5_cvae"))

PHASE3_DIR = os.path.join(REPO_ROOT, "outputs", "phase3")

_IMAGES = None
_PENAL = None


def _init_worker(images, penal):
    global _IMAGES, _PENAL
    _IMAGES = images
    _PENAL = penal


def _solve_one(i: int) -> tuple:
    from verify_fe import evaluate_density_field, resize_to_fe_grid, FE_PARAMS
    fe_params = dict(FE_PARAMS)
    fe_params["penal"] = float(_PENAL[i])
    try:
        img_fe = resize_to_fe_grid(_IMAGES[i], fe_params["nely"], fe_params["nelx"])
        v12_re, v21_re, Q = evaluate_density_field(img_fe, fe_params)
        f1 = float(Q[0, 0]) / fe_params["E0"]
        f2 = float(Q[1, 1]) / fe_params["E0"]
        return (i, f1, f2, v12_re, v21_re, "")
    except Exception as e:
        return (i, float("nan"), float("nan"), float("nan"), float("nan"), str(e))


def run(split: str, workers: int, limit: int = None):
    path = os.path.join(PHASE3_DIR, f"{split}.npz")
    data = np.load(path, allow_pickle=True)
    images = data["images"]
    v12_stored = data["v12"]
    v21_stored = data["v21"]
    penal = data["params"][:, 1]
    n = len(images) if limit is None else min(limit, len(images))
    print(f"[{split}] n={n}, {workers} worker...")

    f1 = np.full(n, np.nan, dtype=np.float64)
    f2 = np.full(n, np.nan, dtype=np.float64)
    v12_re_arr = np.full(n, np.nan, dtype=np.float64)
    v21_re_arr = np.full(n, np.nan, dtype=np.float64)
    errors = 0

    with mp.Pool(workers, initializer=_init_worker, initargs=(images, penal)) as pool:
        for count, (i, f1v, f2v, v12v, v21v, err) in enumerate(
                pool.imap_unordered(_solve_one, range(n), chunksize=32)):
            f1[i], f2[i], v12_re_arr[i], v21_re_arr[i] = f1v, f2v, v12v, v21v
            if err:
                errors += 1
            if (count + 1) % 5000 == 0:
                print(f"  [{split}] ... {count + 1}/{n}")

    ok = ~np.isnan(v12_re_arr[:n])
    dv12 = np.abs(v12_stored[:n][ok] - v12_re_arr[:n][ok])
    print(f"[{split}] loi FE: {errors}/{n}. Sanity |v12_goc-v12_recheck|: "
          f"mean={dv12.mean():.6f}, max={dv12.max():.6f}")

    out_path = os.path.join(PHASE3_DIR, f"{split}_f1f2.npz")
    np.savez_compressed(out_path, f1=f1, f2=f2, v12_recheck=v12_re_arr, v21_recheck=v21_re_arr)
    print(f"[{split}] Da luu: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(args.split, args.workers, args.limit)


if __name__ == "__main__":
    main()
