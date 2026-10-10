"""N2-C6: hậu xử lý xóa đảo vật liệu rời trên thiết kế F (thăm dò, không tiêu chí).

Đảo rắn không nối với khung chính (theo liên thông 4 hướng TUẦN HOÀN - ô được
lát) không chịu lực, nên xóa đi gần như không đổi ν nhưng có thể làm thiết kế
qua kiểm tra chế tạo. Script đo, trên thiết kế guarded của F (IN100, IN100-C):
số pixel bị xóa, |Δν12| verify 50², sai số biên thực tế σ = 0,5 trước/sau, và
tỉ lệ chế tạo trước/sau.

Chạy: python island_cleanup.py
"""

import json
import sys
from multiprocessing import Pool

import numpy as np
from scipy import ndimage

ROOT = "/home/tbm/Documents/Input_SIMP_Analyst"
sys.path.insert(0, f"{ROOT}/pipeline/phase5_cvae")
sys.path.insert(0, f"{ROOT}/outputs/phase5/plan_v3/n1")
from eval_pilot import realize  # noqa: E402
from manufacturability import check_connectivity  # noqa: E402
from verify_fe import evaluate_density_field  # noqa: E402


def remove_islands(img: np.ndarray) -> np.ndarray:
    """Giữ thành phần rắn chính (liên thông 4 hướng, tuần hoàn), xóa phần còn lại.

    Args:
        img: ảnh nhị phân (n, n), 1 = rắn.

    Returns:
        Ảnh nhị phân cùng shape chỉ còn khung chính.
    """
    n = img.shape[0]
    tiled = np.tile(img, (3, 3))
    lab, k = ndimage.label(tiled, structure=ndimage.generate_binary_structure(2, 1))
    if k == 0:
        return img.copy()
    centre = lab[n : 2 * n, n : 2 * n]
    ids, counts = np.unique(centre[centre > 0], return_counts=True)
    main = ids[np.argmax(counts)]
    return (centre == main).astype(img.dtype)


def evaluate(job):
    """Chỉ số trước/sau xóa đảo cho 1 thiết kế."""
    tag, i, img, (t12, t21) = job
    img = (np.asarray(img, float) > 0.5).astype(float)
    clean = remove_islands(img)
    out = {"set": tag, "idx": i, "removed_px": int(img.sum() - clean.sum())}
    for key, d in (("before", img), ("after", clean)):
        v = evaluate_density_field(d)[:2]
        r = evaluate_density_field(realize(d, 200, sigma=0.5))[:2]
        c = check_connectivity(d, connectivity=4)
        out[f"v12_{key}"] = v[0]
        out[f"e_verify_{key}"] = abs(v[0] - t12)
        out[f"e_real05_{key}"] = abs(r[0] - t12)
        out[f"manuf_{key}"] = bool(c["is_connected"] and c["min_feature_ok"])
    return out


if __name__ == "__main__":
    jobs = []
    for tag, p in (("IN100", "e3f_in100_filter.json"), ("IN100-C", "e3f_in100c_filter.json")):
        for i, c in enumerate(json.load(open(p))["per_condition"]):
            img = c["refined_image"] if c["guarded_accept"] else c["baseline_image"]
            jobs.append((tag, i, img, (c["target_v12"], c["target_v21"])))
    with Pool(8) as pool:
        res = pool.map(evaluate, jobs, chunksize=2)
    json.dump(res, open("island_cleanup.json", "w"))
    for tag in ("IN100", "IN100-C"):
        x = [r for r in res if r["set"] == tag]
        touched = [r for r in x if r["removed_px"] > 0]
        dv = [abs(r["v12_after"] - r["v12_before"]) for r in touched]
        print(f"{tag}: {len(touched)}/{len(x)} thiết kế có đảo; px xóa trung vị "
              f"{np.median([r['removed_px'] for r in touched]) if touched else 0:.0f}; "
              f"|Δν12| verify max {max(dv) if dv else 0:.5f}")
        for k in ("e_verify", "e_real05"):
            print(f"  MAE {k}: trước {np.mean([r[k + '_before'] for r in x]):.4f} "
                  f"sau {np.mean([r[k + '_after'] for r in x]):.4f}")
        print(f"  chế tạo: trước {np.mean([r['manuf_before'] for r in x]):.2f} "
              f"sau {np.mean([r['manuf_after'] for r in x]):.2f}")
