#!/usr/bin/env bash
# =============================================================================
# run_gamma_sweep.sh
# Sweep gamma (property-consistency loss weight) trong Phase 5 cVAE training.
# Chạy tuần tự gamma = 10, 30, 50 với epochs=50, tự động backup checkpoint
# + report của mỗi run trước khi run tiếp theo ghi đè (train.py/evaluate.py
# dùng path cố định outputs/phase5/cvae_best.pt, .../evaluation_report.json).
#
# Cách dùng:
#   chmod +x run_gamma_sweep.sh
#   ./run_gamma_sweep.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

GAMMAS=(20 50 100 150)
EPOCHS=50
RESULTS_DIR="outputs/phase5/gamma_sweep_results"
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "  Gamma sweep: ${GAMMAS[*]}  |  epochs=${EPOCHS}"
echo "  Device: $(python3 -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
echo "============================================================"

# --- Kiểm tra tiền điều kiện ---
if [ ! -f "outputs/phase3/train.npz" ] || [ ! -f "outputs/phase3/val.npz" ] || [ ! -f "outputs/phase3/test.npz" ]; then
    echo "LỖI: thiếu outputs/phase3/{train,val,test}.npz — không thể train." >&2
    exit 1
fi
if [ ! -f "outputs/phase4/surrogate_for_phase5.pt" ]; then
    echo "LỖI: thiếu outputs/phase4/surrogate_for_phase5.pt" >&2
    echo "     Chạy trước: python3 pipeline/phase4_surrogate/export_for_phase5.py" >&2
    exit 1
fi

START_TIME=$(date +%s)

for GAMMA in "${GAMMAS[@]}"; do
    echo ""
    echo "------------------------------------------------------------"
    echo "  [gamma=${GAMMA}] Bắt đầu train (epochs=${EPOCHS})"
    echo "------------------------------------------------------------"
    RUN_START=$(date +%s)

    python3 pipeline/phase5_cvae/train.py --gamma "${GAMMA}" --epochs "${EPOCHS}"

    echo "  [gamma=${GAMMA}] Train xong, đang evaluate..."
    python3 pipeline/phase5_cvae/evaluate.py

    RUN_END=$(date +%s)
    echo "  [gamma=${GAMMA}] Thời gian chạy: $(( (RUN_END - RUN_START) / 60 )) phút"

    # --- Backup output trước khi run tiếp theo ghi đè ---
    cp outputs/phase5/cvae_best.pt        "${RESULTS_DIR}/cvae_best_gamma${GAMMA}.pt"
    cp outputs/phase5/train_history.json  "${RESULTS_DIR}/train_history_gamma${GAMMA}.json"
    cp outputs/phase5/evaluation_report.json "${RESULTS_DIR}/eval_gamma${GAMMA}.json"
    # Diagnostics images (diversity_*.png, interpolation_*.png) cũng bị ghi đè
    # mỗi lần evaluate.py chạy — PHẢI backup riêng theo gamma, nếu không sẽ mất.
    mkdir -p "${RESULTS_DIR}/diagnostics_gamma${GAMMA}"
    cp outputs/phase5/diagnostics/*.png "${RESULTS_DIR}/diagnostics_gamma${GAMMA}/" 2>/dev/null || \
        echo "  [gamma=${GAMMA}] CẢNH BÁO: không tìm thấy ảnh diagnostics để backup"
    echo "  [gamma=${GAMMA}] Đã lưu vào ${RESULTS_DIR}/"
done

END_TIME=$(date +%s)
echo ""
echo "============================================================"
echo "  HOÀN TẤT. Tổng thời gian: $(( (END_TIME - START_TIME) / 60 )) phút"
echo "  Kết quả nằm trong: ${RESULTS_DIR}/"
echo "    - eval_gamma{20,50,100,150}.json"
echo "    - train_history_gamma{20,50,100,150}.json"
echo "    - cvae_best_gamma{20,50,100,150}.pt"
echo "    - diagnostics_gamma{20,50,100,150}/ (ảnh diversity + interpolation)"
echo "============================================================"

# --- Tổng hợp nhanh R²(v12) từ 3 file eval để xem trend ngay ---
python3 - << 'PYEOF'
import json, glob, os

results_dir = "outputs/phase5/gamma_sweep_results"
print("\n=== TÓM TẮT R²(v12) theo gamma ===")
for gamma in [1, 5, 20, 30, 50, 80, 100]:
    # gamma 1,5,20 co san tu truoc (outputs/phase5/), gamma 10,30,50 vua chay xong
    candidates = [
        os.path.join(results_dir, f"eval_gamma{gamma}.json"),
        os.path.join("outputs", "phase5", f"eval_gamma{gamma}.json"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        continue
    with open(path) as f:
        data = json.load(f)
    # Cau truc report co the khac nhau tuy version evaluate.py - thu vai key pho bien
    r2 = data.get("r2_v12") or data.get("property_accuracy", {}).get("r2_v12")
    print(f"  gamma={gamma:>3}  R²(v12) = {r2}")
print("\n(Nếu giá trị hiển thị None, mở trực tiếp các file JSON trong")
print(" outputs/phase5/gamma_sweep_results/ để xem cấu trúc report đầy đủ.)")
PYEOF