"""
Audit "dao động nhãn" (label oscillation) - yêu cầu advisor 2026-07-24: user
gửi 3 ảnh ví dụ (mờ/xám, không rõ cấu trúc) và hỏi "còn hình nào thế này
không, loại vì không hội tụ". Điều tra cho thấy các ảnh xấu đó KHÔNG phải
lỗi hình ảnh đơn thuần mà là triệu chứng của 1 vấn đề sâu hơn: vòng lặp OC
(optimality criteria) không hội tụ mà RƠI VÀO LIMIT-CYCLE hoặc còn TRÔI
(drift) ở max_iter - xem ví dụ thật
outputs/multi_batch/batch_3/circle_half_quarter/sample_0401/iteration_data.csv
(v12 nhảy qua lại 0.186264 <-> 0.135430 MỖI iteration suốt 15+ vòng cuối,
và quan trọng: v12 ở đây LÀ SỐ DƯƠNG - tức mẫu này thậm chí không auxetic,
nhãn "auxetic" gán sai).

Vì vậy dùng đúng lịch sử tối ưu (`iteration_data.csv`, có sẵn cho MỌI mẫu)
thay vì đoán qua thống kê pixel (đã thử: ambiguous-pixel-fraction, gray-
plateau-histogram, local-gradient - đều có false positive/negative rõ ràng
khi so bằng mắt, xem draft trong phiên làm việc) - đáng tin cậy hơn nhiều
vì đo trực tiếp trên chính đại lượng dùng làm target ML (v12), không phải
suy diễn gián tiếp qua ảnh render.

Metric: `osc_score = (max-min)/(|mean|+1e-3)` của Poisson_v12 trên
min(20, n_iters) vòng lặp CUỐI. Hiệu chỉnh trên dữ liệu thật (xem
EXPERIMENT_LOG.md): mẫu tốt (đã xác nhận bằng mắt qua ảnh + lịch sử mượt)
osc_score ~0.001-0.03; 2 mẫu xấu xác nhận bằng mắt (401, 210, dao động biên
độ ~40%) osc_score ~0.31-0.32; 2 mẫu biên (circle/25 - drift chưa ổn định,
cross_rectangular/238 - dao động chu kỳ 2 biên độ nhỏ hơn) đều osc~0.19-0.20
và xác nhận thật sự CHƯA ổn định khi xem tail. Ngưỡng chọn: **0.15**
(~9,1% dataset trên mẫu thử 3.000, xem log phiên làm việc) - bảo thủ vừa
phải, có margin rõ với mẫu tốt (p75=0,028) và dưới hẳn 2 mẫu xấu xác nhận
(0,31-0,32).

Áp dụng cho CẢ 2: Phase 1 LHS screening (outputs/pipeline/phase1, 562 mẫu,
không nằm trong dataset train cVAE) VÀ Phase 2 DOE (outputs/phase3/manifest.csv,
7.920 mẫu, dataset train THẬT).

Cách chạy:
    python3 analysis/scripts/audit_label_stability.py

Output:
    outputs/phase3/manifest_quality.csv        thêm cột osc_score, osc_unstable
    outputs/phase3/label_stability_flagged.csv  danh sách mẫu bị flag (Phase 2/3)
    outputs/pipeline/phase1/label_stability_report.json  báo cáo Phase 1
    outputs/pipeline/phase1/label_stability_flagged.csv   danh sách mẫu bị flag (Phase 1)
"""
import os
import sys
import glob
import json

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHASE3_DIR = os.path.join(REPO_ROOT, "outputs", "phase3")
PHASE1_DIR = os.path.join(REPO_ROOT, "outputs", "pipeline", "phase1")

OSC_THRESHOLD = 0.15
TAIL_K = 20


def osc_score(sample_dir: str, k: int = TAIL_K) -> float:
    csv_path = os.path.join(sample_dir, "iteration_data.csv")
    if not os.path.exists(csv_path):
        return np.nan
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return np.nan
    if len(df) < 4 or "Poisson_v12" not in df.columns:
        return np.nan
    tail = df["Poisson_v12"].tail(min(k, len(df)))
    rng = tail.max() - tail.min()
    denom = abs(tail.mean()) + 1e-3
    return float(rng / denom)


def audit_phase23():
    print("=== Phase 2/3 (dataset train cVAE, outputs/phase3/manifest_quality.csv) ===")
    manifest_path = os.path.join(PHASE3_DIR, "manifest_quality.csv")
    if not os.path.exists(manifest_path):
        manifest_path = os.path.join(PHASE3_DIR, "manifest.csv")
        print(f"CẢNH BÁO: chưa có manifest_quality.csv, dùng {manifest_path} "
              f"(thiếu cột is_connected/degenerate - chạy "
              f"audit_convergence_and_geometry.py trước để có đủ).")
    manifest = pd.read_csv(manifest_path)

    scores = []
    for i, row in enumerate(manifest.itertuples()):
        sample_dir = os.path.dirname(row.image_path)
        scores.append(osc_score(sample_dir))
        if (i + 1) % 2000 == 0:
            print(f"  ... {i + 1}/{len(manifest)}")
    manifest["osc_score"] = scores
    manifest["osc_unstable"] = manifest["osc_score"] > OSC_THRESHOLD

    n_unstable = int(manifest["osc_unstable"].sum())
    print(f"\nDao động nhãn (osc_score > {OSC_THRESHOLD}): {n_unstable} "
          f"({n_unstable/len(manifest)*100:.2f}%)")
    print("\nTheo seed:")
    by_seed = manifest.groupby("seed")["osc_unstable"].mean().sort_values(ascending=False) * 100
    for seed, pct in by_seed.items():
        print(f"  {seed:22s} {pct:5.1f}%")

    has_geom_cols = "is_connected" in manifest.columns and "degenerate" in manifest.columns
    if has_geom_cols:
        exclude_mask = (
            (manifest["is_connected"] == False)  # noqa: E712
            | (manifest["degenerate"] == True)   # noqa: E712
            | (manifest["osc_unstable"] == True)  # noqa: E712
        )
    else:
        exclude_mask = manifest["osc_unstable"] == True  # noqa: E712
    n_exclude = int(exclude_mask.sum())
    print(f"\nTổng số mẫu bị loại (rời rạc HOẶC suy biến HOẶC dao động nhãn): "
          f"{n_exclude} ({n_exclude/len(manifest)*100:.2f}%)")

    manifest.to_csv(manifest_path, index=False)
    print(f"Đã cập nhật: {manifest_path}")

    flagged = manifest[exclude_mask]
    out_flagged = os.path.join(PHASE3_DIR, "label_stability_flagged.csv")
    flagged.to_csv(out_flagged, index=False)
    print(f"Đã lưu {len(flagged)} mẫu bị flag: {out_flagged}")

    return manifest, exclude_mask


def audit_phase1():
    print("\n=== Phase 1 LHS screening (outputs/pipeline/phase1, KHÔNG nằm trong dataset train cVAE) ===")
    dirs = sorted(glob.glob(os.path.join(PHASE1_DIR, "*", "sample_*")))
    if not dirs:
        print("Không tìm thấy mẫu Phase 1 - bỏ qua.")
        return None, None
    print(f"Tổng số mẫu Phase 1: {len(dirs)}")

    rows = []
    for i, d in enumerate(dirs):
        seed = os.path.basename(os.path.dirname(d))
        sample_id = os.path.basename(d)
        score = osc_score(d)
        rows.append({"seed": seed, "sample_id": sample_id, "sample_dir": d, "osc_score": score})
        if (i + 1) % 200 == 0:
            print(f"  ... {i + 1}/{len(dirs)}")
    df = pd.DataFrame(rows)
    df["osc_unstable"] = df["osc_score"] > OSC_THRESHOLD

    n_unstable = int(df["osc_unstable"].sum())
    n_valid = df["osc_score"].notna().sum()
    print(f"\nDao động nhãn (osc_score > {OSC_THRESHOLD}): {n_unstable}/{n_valid} "
          f"({n_unstable/n_valid*100:.2f}%)")
    print("\nTheo seed:")
    by_seed = df.groupby("seed")["osc_unstable"].mean().sort_values(ascending=False) * 100
    for seed, pct in by_seed.items():
        print(f"  {seed:22s} {pct:5.1f}%")

    out_json = os.path.join(PHASE1_DIR, "label_stability_report.json")
    with open(out_json, "w") as f:
        json.dump({
            "n_total": len(dirs), "n_unstable": n_unstable, "threshold": OSC_THRESHOLD,
            "per_seed_unstable_pct": by_seed.to_dict(),
        }, f, indent=2, ensure_ascii=False)
    print(f"\nĐã lưu: {out_json}")

    flagged = df[df["osc_unstable"] == True]  # noqa: E712
    out_csv = os.path.join(PHASE1_DIR, "label_stability_flagged.csv")
    flagged.to_csv(out_csv, index=False)
    print(f"Đã lưu {len(flagged)} mẫu bị flag: {out_csv}")

    return df, flagged


if __name__ == "__main__":
    audit_phase23()
    audit_phase1()
