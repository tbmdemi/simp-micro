"""
Ghep f1=E11/E0, f2=E22/E0 (da backfill boi backfill_f1_f2_npz.py, cung thu
tu hang - KHONG can join key gi, xem docstring backfill_f1_f2_npz.py) vao
ban sao MOI cua outputs/phase3/{split}.npz - KHONG ghi de file production
dang dung.

Cach chay:
    python3 analysis/scripts/merge_f1f2_into_npz.py
Output: outputs/phase3/{train,val,test}_ext.npz (giu nguyen moi key cu +
them f1, f2, v12_recheck, v21_recheck, dv12_backfill_qc).
"""
import os
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHASE3_DIR = os.path.join(REPO_ROOT, "outputs", "phase3")


def merge(split: str):
    base = np.load(os.path.join(PHASE3_DIR, f"{split}.npz"), allow_pickle=True)
    f1f2 = np.load(os.path.join(PHASE3_DIR, f"{split}_f1f2.npz"), allow_pickle=True)
    assert len(base["images"]) == len(f1f2["f1"]), \
        f"{split}: so mau khong khop ({len(base['images'])} vs {len(f1f2['f1'])})"

    out = {k: base[k] for k in base.files}
    out["f1"] = f1f2["f1"].astype(np.float32)
    out["f2"] = f1f2["f2"].astype(np.float32)
    dv12_qc = np.abs(base["v12"].astype(np.float64) - f1f2["v12_recheck"])
    out["dv12_backfill_qc"] = dv12_qc.astype(np.float32)

    out_path = os.path.join(PHASE3_DIR, f"{split}_ext.npz")
    np.savez_compressed(out_path, **out)
    print(f"[{split}] Da luu {out_path} ({len(out['images'])} mau), "
          f"dv12_backfill_qc: mean={dv12_qc.mean():.4f}, "
          f"n(qc>0.1)={int((dv12_qc>0.1).sum())} ({100*(dv12_qc>0.1).mean():.2f}%)")


if __name__ == "__main__":
    for split in ("train", "val", "test"):
        merge(split)
