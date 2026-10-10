# Pipeline chi tiết: Screening → Multi-Batch DOE → Dataset → Surrogate → cVAE

> Tài liệu này mô tả đầy đủ từng bước của pipeline 8-phase, tách ra từ `README.md` để giữ README ngắn gọn. Xem [README.md](../README.md#trạng-thái-dự-án) cho bảng trạng thái tổng quan, và [LIMITATIONS.md](LIMITATIONS.md) cho phạm vi claim khoa học + giới hạn đã biết.

## 1. LHS Screening (Phase 1)

Quét không gian tham số (`volfrac`, `penal`, `rmin`, `move`, `void_size_frac`, `rotation_deg`) bằng Latin Hypercube Sampling.

**2026-08-18 - thêm `nu` (ν0, hệ số Poisson vật liệu nền) vào `PARAM_SPACE`** (`pipeline/params.py`, dải `(0.2, 0.4)` - Giai đoạn A, xem mục 4-5 bên dưới): trước đây vật liệu nền cố định (`nu=0.3` mọi mẫu), giờ là 1 trục thiết kế bổ sung được quét cùng các tham số DOE khác. Dải hẹp có chủ đích - `nu=0.5` làm `(1-nu²)=0` trong ma trận độ cứng vật liệu (`Material._compute_element_stiffness`), `nu` gần `-1` cũng phân kỳ - không mở sát biên vật lý `(-1, 0.5)` khi chưa xác nhận hội tụ FE ổn định (đã kiểm chứng bằng pilot 150 mẫu, xem [EXPERIMENT_LOG.md](../EXPERIMENT_LOG.md) mục 2026-08-18).

```bash
python -m pipeline.phase1_screening.screening_parallel --objective auxetic --seed hexagonal
python -m pipeline.phase1_screening.screening_parallel --all   # quét toàn bộ, tất cả seed
python -m pipeline.phase1_screening.analyst                    # tổng hợp kết quả -> _all_correlations.json, _all_summaries_parallel.json
```

Phân tích độ nhạy (tương quan Spearman) xác định **`volfrac` là tham số chi phối** (r ≈ 0,87–0,96); `move`, `rmin`, `void_size_frac` không có ý nghĩa thống kê. (Lịch sử debug lần chạy đầu - 0 mẫu auxetic - xem [EXPERIMENT_LOG.md](../EXPERIMENT_LOG.md).)

## 2. Multi-Batch Adaptive DOE (Phase 2) - ✅ hoàn thành

Các lô tuần tự, mỗi lô được định hướng bởi phân tích độ phủ (KDE + phát hiện vùng thưa) trên kết quả tích lũy. `adaptive.py` quyết định **tinh chỉnh** (thu hẹp khoảng tham số + nhắm vào vùng thưa), **mở rộng** (thêm seed/mục tiêu), hoặc **dừng**.

```bash
python -m pipeline.phase2_multi_batch.main --phase1-summary outputs/pipeline/phase1
```

**Kết quả (8 lô, 7.920 mẫu, tỷ lệ hội tụ FE 100%):**

| Lô | Chiến lược | Số mẫu | % Auxetic | ν₁₂ tốt nhất |
|-------|----------|-----------|-----------|----------|
| 1 | Sobol (khám phá) | 1.320 | 74,9% | −0,612 |
| 2 | Sobol (khám phá) | 600 | 79,7% | −0,519 |
| 3 | Sobol (khám phá) | 720 | 71,8% | −0,565 |
| 4 | Optimized LHS (tinh chỉnh) | 1.056 | 83,6% | −0,605 |
| 5 | Optimized LHS (tinh chỉnh) | 1.067 | 85,7% | −0,752 |
| 6 | Optimized LHS (tinh chỉnh) | 1.045 | 85,4% | −0,649 |
| 7 | Optimized LHS (tinh chỉnh) | 1.056 | 85,3% | −0,621 |
| 8 | Optimized LHS (tinh chỉnh) | 1.056 | 87,8% | **−0,807** |

Khoảng `volfrac` hội tụ từ `[0,45, 0,70]` xuống `[0,50, 0,58]` ở lô 8; pipeline tự dừng sau 2 lô liên tiếp không cải thiện mục tiêu (độ thưa ổn định ~18,5%).

**2026-07-24 - 2 cải tiến tích hợp** (chi tiết đầy đủ + bảng số liệu: [EXPERIMENT_LOG.md](../EXPERIMENT_LOG.md)):
- Phân tích ngược 7.920 mẫu cho thấy manufacturability hầu như KHÔNG tương quan với tham số DOE liên tục (|Spearman r|<0,12) - biến chi phối là **SEED** (7,9% ở `hexagonal` tới 62,8% ở `reentrant_bowtie`). Thêm đo manufacturability tại thời điểm sinh (miễn phí) + `compute_seed_sample_allocation()` phân bổ mẫu lệch theo seed thay vì chia đều. Validated bằng 99 mẫu thật (lô 9): joint rate (auxetic∧manufacturable) 17,6%→**25,3%**.
- Lỗi hoán vị `dQ` (xem [Giới hạn #10](LIMITATIONS.md#giới-hạn-đã-biết--known-limitations)) khiến 3 seed bất đối xứng hội tụ dưới tiềm năng thật (2/3 từng sai cả dấu) → rebuild đầy đủ 2.160 mẫu (lô 11). Auxetic rate toàn dataset: 82,1%→**91,9%**.

## 3. Dataset Build (Phase 3) - ✅ hoàn thành

```bash
python3 pipeline/phase3_dataset/scan_dataset.py       # -> outputs/phase3/manifest.csv
python3 pipeline/phase3_dataset/build_npz.py --resolution 64   # -> dataset_64.npz
python3 pipeline/phase3_dataset/finalize_dataset.py --resolution 64  # -> train/val/test.npz
```

- Ảnh PNG mật độ (từ lưới `xPhys` 50×50) resize về 64×64 bằng box-filter downsampling; 33/7.920 mẫu (0,4%) bị loại (topology suy biến, `volfrac_achieved` ngoài `[0,05, 0,95]`).
- **Chia train/val/test 70/15/15, phân tầng theo seed**; **tăng cường đối xứng** (chỉ train): xoay 90°/270° hoán đổi `ν₁₂↔ν₂₁`, xoay 180°/lật giữ nguyên. Train: 5.520 → 33.120 mẫu (×6) *(số liệu lịch sử - xem cập nhật ngay dưới)*.
- Target xuất ra: `v12`, `v21`, `volfrac_achieved`. `f1, f2` (roadmap gốc) - xem cập nhật 2026-08-05 ngay dưới.

**2026-08-18 - dataset Giai đoạn A (ν0 vật liệu nền biến thiên):** `analysis/scripts/generate_production_batch.py --n-raw 3000 --vary-nu` sinh 3.000 mẫu quét đầy đủ `nu∈(0.2,0.4)` (không thu hẹp theo seed - pilot xác nhận không có "ranh giới hội tụ sắc" như 1 comment cũ tưởng, xem [EXPERIMENT_LOG.md](../EXPERIMENT_LOG.md) mục 2026-08-18), 2.635 mẫu sạch (87,8%). `analysis/scripts/assemble_phase3_a4.py` gộp với pool `dataset_64.npz` cũ (13.624 mẫu, `nu=0,3` fallback) → `outputs/phase3_a4/` = 16.259 mẫu sạch, train 68.286 (sau augment ×6) / val 2.439 / test 2.439 - **chưa promote thành `outputs/phase3/` production**, dùng `--data-dir outputs/phase3_a4` để trỏ tới. `pipeline/phase3_dataset/build_npz.py` thêm field `nu` (fallback `0.3` cho mẫu cũ không có).

**2026-08-05 - Backfill f1=E₁₁/E₀, f2=E₂₂/E₀ (Pha B), Phase 4 surrogate 5-chiều:** `analysis/scripts/backfill_f1_f2_npz.py` chạy FE trực tiếp trên ảnh đã lưu trong `{train,val,test}.npz` (không join qua manifest.csv - lý do và caveat nhiễu resize xem [EXPERIMENT_LOG.md](../EXPERIMENT_LOG.md) mục 2026-08-05). Surrogate mở rộng (`train.py --include-f1f2`) đạt R²(test): f1=0,933, f2=0,961 - cùng bậc v12/v21. Checkpoint: `outputs/phase4/surrogate_f1f2.pt`. **Chưa làm:** nối f1/f2 làm condition cho cVAE Phase 5.

**2026-09-30 - Tính chất suy từ Q cho dataset A4:** `analysis/scripts/backfill_elastic_props_npz.py` FE lại toàn bộ `outputs/phase3_a4/{train,val,test}.npz` (penal và ν0 thật từng mẫu, cùng phương pháp resize 64→50 như backfill f1/f2) và ghi `{split}_props.npz`: Q thô (n,3,3) cùng E_x, E_y, G_xy, B_eff, M_x = Q11/E0 (= f1), M_y = Q22/E0 (= f2), tốc độ sóng quasi-static c_qL/c_qT theo x, y, `rel_density` và v12/v21 tính lại (sanity). Mọi mô-đun đã chia E0. Ghép với npz gốc theo chỉ số hàng. Kết quả phân tích: [`notebooks/03_cheap_physical_properties.ipynb`](../notebooks/03_cheap_physical_properties.ipynb) và [EXPERIMENT_LOG.md](../EXPERIMENT_LOG.md) mục 2026-09-30. Tóm tắt: E/B/M dự đoán được 92–97% từ 5 điều kiện hiện có; **G_xy** (R² 0,84) mang nhiều thông tin mới nhất.

**2026-07-24 - dọn dữ liệu tận gốc:** manifest hiện tại đã lọc bỏ 2.662/7.920 mẫu (33,6%, nhãn dao động/rời rạc/ngoài khoảng vật lý - xem [Giới hạn #13](LIMITATIONS.md#giới-hạn-đã-biết--known-limitations)). Pipeline cho ra `train.npz`=**22.080** mẫu, `val.npz`/`test.npz`=**789** mẫu mỗi tập (khác số liệu lịch sử 33.120/33.246 ở trên). `cvae_gamma20.pt` cũ vẫn train trên bản CŨ và được giữ nguyên làm baseline lịch sử; checkpoint mới train trên bản sạch - xem `surrogate_clean.pt`/`cvae_clean_v2.pt` ở mục 4-5 ngay dưới.

**2026-07-25 - Rebuild v2 (mở rộng quy mô):** sinh thêm raw sample bằng `analysis/scripts/generate_production_batch.py` (song song, độc lập với `pipeline/phase2_multi_batch/`), gộp với pool sạch cũ qua `analysis/scripts/assemble_phase3_v2.py`. Quá trình phát hiện + sửa 2 bug trong chính script sinh dữ liệu mới (range tham số quá rộng cho seed nhạy cảm; `beta=0,8` thay vì mặc định production `1,0`) - chi tiết đầy đủ (mục "2026-07-25"): [EXPERIMENT_LOG.md](../EXPERIMENT_LOG.md). **Kết quả: train=57.216 (đạt mục tiêu 40-50k), val=2.044, test=2.044.** Đã thay thế `outputs/phase3/{train,val,test,dataset_64}.npz` (bộ cũ backup nguyên vẹn tại `outputs/phase3_backup/`).

## 4. CNN Surrogate Model (Phase 4) - ✅ hoàn thành

```bash
python3 pipeline/phase4_surrogate/train.py
python3 pipeline/phase4_surrogate/evaluate.py
python3 pipeline/phase4_surrogate/export_for_phase5.py
```

Kiến trúc (baseline "Phương án A"): 4× khối `Conv(3x3) + BatchNorm + ReLU + MaxPool` → global average pool → nối với one-hot của seed → 2 lớp FC → 3 đầu ra (ν₁₂, ν₂₁, volfrac_achieved). Huấn luyện trên `outputs/phase3/train.npz`, xác thực trên `val.npz`.

**Hiệu năng trên test set** (`outputs/phase4/evaluation_report.json`, `test.npz` giữ riêng - **1.184 mẫu**, không rò rỉ dữ liệu):

| Target | R² | 95% CI (bootstrap, n=1.184) | MAE |
|---|---|---|---|
| ν₁₂ | 0,910 | [0,896, 0,923] | 0,037 |
| ν₂₁ | 0,911 | [0,892, 0,926] | 0,036 |
| volfrac_achieved | 0,982 | [0,979, 0,984] | 0,007 |

CI tính bằng `pipeline/phase4_surrogate/bootstrap_ci.py` (percentile bootstrap trên 1.184 mẫu test - không cần train lại). Khác hẳn Phase 5 (CI rất rộng do chỉ 19-24 điều kiện, xem [LIMITATIONS.md](LIMITATIONS.md)), CI ở đây **hẹp** vì cỡ mẫu lớn - con số R²=0,91 đáng tin cậy, không phải một điểm ước lượng may rủi.

MAE theo seed dao động 0,021–0,048, không seed nào kém nghiêm trọng. Nếu R² < 0,90 trên bất kỳ target nào, thử mở rộng `channels` trong `SurrogateCNN` trước khi đổi kiến trúc.

> **Bảng trên đo trên `surrogate_best.pt` (data cũ, lẫn 33,6% nhãn lỗi).** Sau khi dọn dữ liệu (mục 3), `surrogate_clean.pt` đo trên cùng 1 test set sạch (789 mẫu): **v12 R²=0,922** [CI 0,907,0,935], **v21 R²=0,889** [CI 0,860,0,913] - so với `surrogate_best.pt` đo trên CHÍNH test set sạch này chỉ 0,484/0,384. Bằng chứng thực nghiệm rằng 33,6% mẫu nhãn lỗi là nguyên nhân chính khiến R² cũ thấp, không phải giới hạn kiến trúc (chi tiết + ablation xác nhận: [EXPERIMENT_LOG.md](../EXPERIMENT_LOG.md)). `surrogate_best.pt`/`surrogate_for_phase5.pt` **không bị ghi đè** - dùng `--surrogate-path outputs/phase4/surrogate_clean.pt` cho công việc mới.

> **2026-07-25 - retrain trên dataset v2** (57.216 mẫu train): `surrogate_v2.pt` đo trên test set v2 (2.044 mẫu) - **v12 R²=0,974**, **v21 R²=0,964**, **volfrac R²=0,983** - cải thiện so với `surrogate_clean.pt` (0,922/0,889), chủ yếu nhờ quy mô dữ liệu lớn hơn. Export cho Phase 5 tại `outputs/phase4/surrogate_for_phase5_v2.pt`. Không ghi đè `surrogate_best.pt`/`surrogate_for_phase5.pt` (quy ước cũ) - dùng `--ckpt`/`--surrogate-path` trỏ tới bản `_v2` cho công việc mới.

> **2026-08-18 - `--include-nu0` (Giai đoạn A, A4):** `SurrogateCNN(include_nu0=True)` nhận thêm ν0 (nối vào `fc_in` cùng seed one-hot sau global-average-pool), `--data-dir outputs/phase3_a4` để train trên dataset có field `nu`. So 2 model CÙNG dataset A4 (68.286 mẫu) để tách bạch hiệu ứng ν0 khỏi hiệu ứng cỡ dữ liệu: `surrogate_a4_control.pt` (không ν0, v12 R²=0,9816/v21 R²=0,9734) so với `surrogate_a4_nu0.pt` (có ν0, v12 R²=0,9816/v21 R²=0,9731) - **khác biệt không đáng kể** (trong nhiễu train-to-train), vì mẫu ν0 biến thiên thật chỉ chiếm ~16% dataset (2.635/16.259). Sàn cứng CLAUDE.md vẫn đạt (cả 2 vượt rõ baseline `surrogate_v2.pt` 0,974/0,964, chủ yếu nhờ dataset lớn hơn) - không phải đánh đổi accuracy/performance, mục tiêu A4 là hạ tầng (surrogate sẵn sàng nhận ν0) chứ chưa phải đo lợi ích ở tầng này. Khuyến nghị `surrogate_a4_nu0.pt` cho công việc nối tiếp Phase 5 (mục 5.2). Xem [EXPERIMENT_LOG.md](../EXPERIMENT_LOG.md) mục 2026-08-18 cho số liệu đầy đủ - lợi ích thật của ν0 chỉ lộ rõ ở tầng cVAE (mục 5.2), không phải ở surrogate.

## 5. Conditional VAE (Phase 5) - ✅✅ đã sửa tận gốc bằng differentiable-physics (2026-07-24)

> **Checkpoint khuyến nghị hiện tại: `outputs/phase5/cvae_v2_finetuned.pt`** (train đúng recipe 2-stage trên dataset v2, xem mục "retrain trên dataset v2" bên dưới) - dùng để nhất quán với dataset production hiện tại (57.216 mẫu). `cvae_realphysics.pt` (fine-tune trên dataset v1) vẫn được giữ nguyên làm tham chiếu lịch sử/không bị ghi đè, hiệu năng ngang ngửa (xem bảng so sánh bên dưới) nên vẫn dùng được nếu cần tái lập kết quả cũ. Không dùng `cvae_gamma20.pt` (baseline lịch sử tiền differentiable-physics) hay `cvae_clean_weighted.pt` (tốt nhất TRƯỚC differentiable-physics) cho công việc mới.

```bash
# Fine-tune từ checkpoint có sẵn với differentiable-physics (khuyến nghị - cách đã tạo ra cvae_realphysics.pt):
python3 pipeline/phase5_cvae/train.py --gamma 20.0 --epochs 35 --kl-warmup 1 \
  --surrogate-path outputs/phase4/surrogate_clean.pt --resume-from outputs/phase5/cvae_clean_weighted.pt \
  --output-name cvae_realphysics.pt --select-by fe_r2 --fe-eval-every 2 \
  --lambda-real-physics 20.0 --real-physics-subsample 8 --real-physics-every 2 --real-physics-workers 8

# Train từ đầu, KHÔNG differentiable-physics (baseline lịch sử, phần dưới đây mô tả câu chuyện đã sửa):
python3 pipeline/phase5_cvae/train.py --gamma 20.0 --epochs 50
python3 pipeline/phase5_cvae/evaluate.py
python3 pipeline/phase5_cvae/sample.py --ckpt outputs/phase5/cvae_v2_finetuned.pt   # single-shot giờ đáng tin ở checkpoint này
```

`sample.py` sinh 1 ứng viên/lần gọi. Với các checkpoint CŨ (không differentiable-physics), không lọc FE nên chỉ để xem qua - in cảnh báo mỗi lần chạy, dùng `best_of_n_eval.py` mới đáng tin. Với `cvae_v2_finetuned.pt`/`cvae_realphysics.pt`, single-shot đã đủ tin cậy (≥98% hit rate) nhưng `best_of_n_eval.py` vẫn là cách đo lường chính thức/nghiêm ngặt nhất.

Huấn luyện trên `train.npz`, dùng surrogate Phase-4 **đóng băng** để tính loss property-consistency (`recon + beta·kl + gamma·PROP_LOSS_SCALE·prop_loss`).

**Quy trình đo lường chính thức (`best_of_n_eval.py`):** sinh N ứng viên/điều kiện, để **FE thực** (không phải surrogate) chọn người thắng cuộc - công thức lấy cảm hứng từ pipeline Deep-DRAM đã công bố (Pahlavani et al. 2024, xem [README.md § Tài liệu Tham khảo](../README.md#tài-liệu-tham-khảo)).

```bash
python3 pipeline/phase5_cvae/best_of_n_eval.py --n-samples 30                        # oracle (FE trên toàn bộ N)
python3 pipeline/phase5_cvae/best_of_n_eval.py --n-samples 30 --k-fe-verify 10        # thực dụng (lọc sơ bộ bằng surrogate)
python3 pipeline/phase5_cvae/best_of_n_eval.py --n-samples 1500 --k-fe-verify 8 --require-manufacturable   # bắt buộc khả năng chế tạo
```

**Tóm tắt hành trình phát hiện + sửa** (mọi bảng số liệu/CI đầy đủ: [EXPERIMENT_LOG.md](../EXPERIMENT_LOG.md)):

1. R² đo qua surrogate đóng băng không đáng tin (**surrogate exploitation** - decoder học đánh lừa CNN thay vì sinh hình học đúng): FE thực cho R² âm sâu ở mọi mức gamma đã thử. Hai biện pháp khắc phục lúc huấn luyện (self-play, ensemble surrogate) đều **không** khắc phục được vấn đề single-shot.
2. `force_periodic()` (ép cứng periodicity bằng 1 phép gán pixel, không cần train) tích hợp làm mặc định BẬT trong `best_of_n_eval.py`/`sample.py` (cờ `--no-force-periodic` để tắt) - nâng manufacturability >10× gần như miễn phí.
3. Dọn dữ liệu (loại 33,6% mẫu nhãn dao động, [Giới hạn #13](LIMITATIONS.md#giới-hạn-đã-biết--known-limitations)) + weighted-sampling theo phổ auxetic + chọn checkpoint bằng `--select-by fe_r2 --fe-eval-every N` (**không dùng `val_loss` mặc định** - bị KL-warmup đánh lừa, xác nhận độc lập 2 lần) → `cvae_clean_weighted.pt`, checkpoint tốt nhất trước differentiable-physics (oracle R²=0,83, K=10 R²=0,65 ở n=24).
4. Đánh giá lại ở quy mô lớn (n=789, TOÀN BỘ test set): oracle mode giữ được cải thiện (khiêm tốn hơn n=24: R²=0,44 chứ không phải 0,83). Chế độ "thực dụng" K=10 **sụp đổ về CI âm/cắt-0 cho mọi checkpoint TRƯỚC differentiable-physics** - khuyến nghị trước đây ("K=10 giữ gần hết R² gain") đã SAI và bị gỡ bỏ.
5. **Differentiable-physics** (`real_physics_prior_loss()` - FE-solve thật + gradient giải tích ngay trong training loop, không qua surrogate) sửa **tận gốc**: `cvae_realphysics.pt` đạt R²(FE, n=789)=**0,999** (oracle)/**0,992** (K=10), hit rate single-shot **98,7%**. `train.py` hỗ trợ `--lambda-real-physics/--real-physics-subsample/--real-physics-every/--real-physics-workers` để giữ chi phí FE-solve hợp lý (~50-100ms/mẫu tuần tự, song song qua `multiprocessing.Pool`).

**Khả năng chế tạo:** best-of-N mặc định tối ưu độ chính xác Poisson; thêm `--require-manufacturable` + N lớn khi cần đảm bảo khả thi chế tạo (đổi lấy R² thấp hơn ở checkpoint chưa dùng differentiable-physics - không cần đánh đổi này ở checkpoint hiện tại, `frac_manufacturable` đã cao sẵn ~0,25-0,35). Biện pháp huấn luyện lại từng thử (regularize posterior/prior-samples) kém hiệu quả hơn `force_periodic()` - chi tiết: [EXPERIMENT_LOG.md](../EXPERIMENT_LOG.md).

**2026-07-25 - retrain trên dataset v2:** train from-scratch với real-physics loss ngay từ epoch 1 **thất bại** (R²(FE) âm sâu) - xác nhận quy trình 2-stage (base rồi fine-tune, xem mục trên) là **bắt buộc**, không thể gộp 1 bước. `cvae_v2_finetuned.pt` (đúng recipe 2-stage) so với `cvae_realphysics.pt`, đo trên CÙNG test set v2 (n=300):

| Checkpoint | R²(FE) oracle | hit rate single-shot | frac manufacturable |
|---|---|---|---|
| `cvae_v2_finetuned.pt` (train trên dataset v2) | 0,9953 | **0,997** | 0,247 |
| `cvae_realphysics.pt` (train trên dataset cũ, giữ nguyên) | **0,9984** | 0,983 | **0,280** |

Hai checkpoint ngang ngửa nhau (cả hai đạt hit-rate best-of-N=1,0); `cvae_v2_finetuned.pt` nhỉnh hơn ở hit-rate single-shot (số quan trọng nhất cho dùng thực tế không lọc), `cvae_realphysics.pt` nhỉnh hơn nhẹ ở R²/manufacturability. **Khuyến nghị: dùng `cvae_v2_finetuned.pt` cho nhất quán với dataset production hiện tại**; `cvae_realphysics.pt` vẫn được giữ nguyên làm tham chiếu lịch sử, không bị ghi đè.

**2026-08-23 - KAN regression head + WIRE decoder (Task 1/2, nhánh `substrate-material`):**

Kiến trúc `pipeline/phase5_cvae/model.py` mở rộng:
- **KAN-hóa bộ hồi quy (Task 1):** `Encoder.fc_mu`/`fc_logvar` + `Decoder.fc` đều là `EfficientKANLinear` (base_weight + spline_weight + grid B-spline bậc 3). `resize_condition_dim_weights()` đã xử lý đúng cấu trúc KAN khi `--resume-from` condition_dim khác.
- **WIRE decoder (Task 2):** flag `decoder_type="conv"|"wire"` (mặc định `conv`, tương thích ngược). `wire` dùng `WireContinuousDecoder` - kích hoạt Gabor Wavelet phức, sinh mật độ ρ ∈ [0,1] tại tọa độ liên tục `(x,y) ∈ [-1,1]²`, resolution-agnostic qua `generate(resolution=...)` (128²/256²/512²); API đầu ra `(B,1,H,W)` giữ nguyên cho downstream (`sample.py`/`best_of_n_eval.py`/`verify_fe.py`).
- CLI mới: `train.py --decoder-type --wire-hidden-dim --wire-omega0 --wire-s0`; `sample.py --resolution`.

Kết quả train KAN (đo bằng `evaluate.property_accuracy`, surrogate v2 hiện tại): KAN base (chọn val_loss) R²=0,19; fine-tune real-physics 2 vòng → `cvae_kan_realphysics_v2.pt` R²=0,64 (v12 0,82, v21 0,45), R²(FE thật) tốt nhất trong train = 0,8889 - vượt baseline Linear đo lại (0,29/0,35) gần 2×. **Mục tiêu (DoN v2, Task 1): R² tổng≥0,60 VÀ R²(FE)≥0,85 - ĐÃ ĐẠT cả 2.** **Lưu ý:** report tháng 7 (R²=0,85) không tái lập với code hiện tại (cùng checkpoint đo lại = 0,35) - chi tiết [EXPERIMENT_LOG.md](../EXPERIMENT_LOG.md) mục 2026-08-23. WIRE đã có checkpoint `cvae_wire_v2.pt`, nhưng benchmark best-of-N mới đạt 16,67% và R²(FE)=-10,29, chưa đạt KPI. **Mục tiêu (DoN Task 2): hit-rate/`passes_all` single-shot ≥75% - đo được 4,17%, còn cách rất xa.**

**Exp 3 — đối chứng MLP/KAN (2026-09-11):** với cùng test loader, surrogate
v2 và seed prior cố định, MLP + Physics đạt `R²(v12)=0,432`,
`R²(v21)=0,087`, `R²(volfrac)=-2,483`, `R²(FE,v12)=-9,834` (n=24).
Checkpoint KAN v2 đạt tương ứng `0,836`, `0,476`, `-0,221`, `0,581`.
MLP đã chạy đủ 150 epoch; checkpoint tốt nhất theo FE-R² là epoch 20
(`-6,2465`). Vì KAN v2 chưa retrain lại sau khi thêm decoder symmetry và
volfrac loss, kết quả này là ablation định hướng, chưa phải so sánh hoàn toàn
đồng nhất về thời điểm mã nguồn.

**2026-09-23 - WIRE (Task 2): root-cause tìm ra, thử sửa, vẫn THẤT BẠI - cập nhật con số `16,67%`/`-10,29` ở trên (đó vẫn là kết quả TỐT NHẤT từng đo cho WIRE, mọi lần thử sau đều tệ hơn).** Root-cause thật: `cvae_wire_v2.pt` train với `--lambda-real-physics 0.0` (chưa từng dùng real-physics loss) + `WireContinuousDecoder` thiếu `enforce_symmetry` mà `Decoder` (conv) có sẵn - đã sửa cả hai. Retrain 2 round (~75 epoch hiệu dụng, `--wire-hidden-dim 128 --lambda-real-physics 1.0→2.0`): R²(FE) đo trên 8-condition validation dùng lúc train cải thiện đều (tới -10,10, tốt nhất từng đo), nhưng benchmark CHÍNH THỨC 24-condition (`best_of_n_eval.py`) cho `cvae_wire_realphysics_v3_round2.pt` **tệ hơn baseline gốc trên mọi trục**: hit-rate single-shot/best-of-N = 0%/0% (gốc 4,17%/16,67%), R²(FE)=-13,87 (gốc -10,29), frac_manufacturable=2,78% (gốc 4,17%). Bằng chứng cụ thể của overfitting lên validation subset nhỏ dùng để chọn checkpoint. **Task 2 (WIRE) coi như đóng ở trạng thái thất bại** - chi tiết đầy đủ [EXPERIMENT_LOG.md](../EXPERIMENT_LOG.md) mục 2026-09-23, [task_progress.md](archive/task_progress.md) Task 2.

**2026-09-25 - plan v3 (`docs/plan.md`): refinement nhận thức nhị phân hóa, ablation KAN, đóng WIRE:**

- **Bug loader (đã sửa):** `adversarial_dataset.load_cvae()` bỏ qua `use_kan`/`enforce_symmetry` →
  mọi đánh giá qua loader này từ 2026-09-11 trên checkpoint cũ bị sai (KAN cũ bị ép đối xứng,
  Linear crash). Nay mọi loader dùng chung `model.cvae_kwargs_from_checkpoint()`. Số benchmark
  refinement 2026-09-23 ở trên (R² −16 → −7,6) là sai; đúng là 0,058 → 0,92. `LIMITATIONS.md` mục 28.
- **Refinement nhận thức nhị phân hóa:** tối ưu z trên ảnh liên tục khai thác vật liệu xám (loss
  liên tục ~1e-10 nhưng sai số sau nhị phân hóa lớn). Objective mới
  (`tandem_lbfgs(projection_betas=, periodic=)`, `pipeline/phase5_cvae/heaviside.py`):
  force_periodic → Heaviside (β tăng dần) → resize nearest khớp PIL tuyệt đối. Trên
  `cvae_v2_finetuned.pt`, `IN100`: single-shot R²(v12) 0,874→**0,998** (MAE −91,8%), sau best-of-30
  0,986→0,997 (MAE −62,6%). Ngoài dải train (v12 < −1,95) vẫn bão hòa - không claim ngoại suy.
  Target đối xứng ν12=ν21≤−1 bất khả thi vật lý (cần ν12·ν21<1) - OOD phải dị hướng.
- **Ablation KAN vs Linear** (cùng recipe 2 giai đoạn production, `--disable-symmetry`, 2 seed):
  KAN không thắng; ở best-of-30 Linear tốt hơn có ý nghĩa trên cả 2 seed. Production giữ Linear.
- **WIRE đóng:** lần thử cuối (Heaviside trong loss train `--rp-projection-beta-max 64
  --rp-periodic`) đạt hit-rate single-shot 12,5%, R²(FE) −7,19 - không đạt tiêu chí dừng đặt trước.

**2026-10-07 - P1.7-P1.9 (`docs/plan.md`): lai, xác nhận phân tầng, bỏ `force_periodic`:**
- **P1.7** lai cVAE → SIMP (`benchmark_hybrid.py`) không đạt 2/4 tiêu chí đăng ký trước (sai số cặp
  CI dưới −96% do 1 ô đứt; chế tạo 0,33 < 0,70) → bài viết theo Khung A.
- **P1.8** (đăng ký trước, IN100-C seed 789): với mục tiêu dị hướng r < 10, sai số cặp cVAE thấp hơn
  SIMP hội tụ 27% [8; 42] (đạt), ν₁₂ +19% [−10; 40] (không đạt). Mọi ô cVAE đứt là r ≥ 10.
- **P1.9 / C5:** bỏ `force_periodic`; chạy lại mọi số → `p1_9_nofp_summary.json`. Không fp: best-of-30
  + refine R²(ν₁₂) 0,998, chế tạo 0,32, OOD −2,0 trung vị 1,2% (2 cụm), so SIMP r<10 ν₁₂ +37-39% ở cả
  3 tập (phân tích độ nhạy, P1.8 giữ kết quả có fp). R² 0,995 (n=300) là chọn thuần độ chính xác;
  composite cho 0,983 (có fp) / 0,988 (không fp).

**2026-10-06 - P1.6 (`docs/plan.md`): baseline SIMP + kiểm tra bổ sung trước khi nộp Bài #1:**

- **SIMP chạy từ đầu** (`benchmark_simp_baseline.py`, inverse homogenization MMA + Heaviside,
  4 seed) [**sửa 2026-10-07:** không tái lập có ý nghĩa ở IN100-B/IN100-C - claim cuối là "ngang SIMP", xem mục 2026-10-07 phía trên]: trên IN100 cVAE + refine guarded (107 FE) tốt hơn SIMP hội tụ (897 FE) về ν₁₂ 34%
  [13; 50]; cặp (ν₁₂, ν₂₁) hòa (1 mục tiêu cVAE sinh ô đứt; bỏ đi thì cVAE +29% [9; 46]); SIMP
  thắng chế tạo được (77-84% vs ~25%) và OOD đổi dấu. SIMP β=64 cũng có khoảng lệch xám↔nhị phân.
- **IN100-B** (seed 456): refine tái lập (−87,8% single-shot, −65,8% best-of-30).
- **Lưới**: ν 50² vs 200² lệch trung vị 0,015; thiết kế cuối trên 200²: cVAE R²(ν₁₂) 0,992, SIMP 0,985.
- **Retrieval verified**: 0,94-0,98 (không phải 1,000); nhãn dataset lệch ν verify 0,019-0,055;
  `force_periodic` có thể sai khái niệm (LIMITATIONS #34, chờ quyết định).
- Refine giảm chế tạo được 30% → 24% (**đính chính 2026-10-07:** không vững - 5 lần chạy −6 đến +9 điểm %, gộp 61 thêm / 44 mất, p=0,12). Chi phí dataset ≈2,7-3,0M FE-solve.
- **Lưu ý huấn luyện:** chỉ 2,5% ảnh train đối xứng qua đường chéo - decoder `enforce_symmetry=True`
  (mặc định mới của `CVAE`) buộc v12=v21, nên train mới dùng `--disable-symmetry`.

**2026-08-24 - các module roadmap bổ sung:**

- `pipeline/mno/` cung cấp MNO dự đoán 18 trường displacement, dataset adapter
  NPZ và train/evaluate CLI. `mamba_ssm` là backend tùy chọn; fallback GRU hai
  chiều giữ pipeline chạy được khi CUDA kernel chưa cài.
- `pipeline/kinn/` cung cấp KINN dùng `EfficientKANLinear`, deep-energy loss
  differentiable bằng PyTorch autograd và adapter JAX-CG/SciPy. Hook
  `losses.kinn_prior_loss()` đã nối vào namespace Phase 5; chưa phải solver
  hyperelastic production.
- `pipeline/ickans/` cung cấp mô hình năng lượng lồi cho composite với signed
  features, trọng số hiệu dụng dương và quadratic curvature floor; evaluate CLI
  báo R² cùng eigenvalue nhỏ nhất của tangent Hessian.
- Kết quả compute: MNO smoke đúng shape `(1,18,16,16)` và khoảng 0,0004 s/sample
  trên GPU fallback; KINN gradient hữu hạn; ICKAN eigenvalue nhỏ nhất 0,00102,
  nhưng R² smoke=-0,069. Các KPI MNO/KINN/ICKAN production chưa đạt vì thiếu
  dataset displacement FE 18 kênh và môi trường chưa có `mamba_ssm`/`jax-amg`.

Các lệnh roadmap xem trong `CLI_GUIDE.md`; luôn chạy sau `conda activate simp`.

### 5.1. Tham số input tùy chọn volfrac và void size frac, chấm điểm toàn diện

> Tính năng này đã merge vào `main` (PR #13, nhánh phát triển cũ `feature/optional-multi-condition`). Mục này mô tả hành vi hiện có, có thể bật qua cờ `--extended-condition` khi cần - **không** phải hành vi mặc định của các lệnh ở mục 5 phía trên (checkpoint mặc định vẫn `condition_dim=2`, chỉ `v12/v21`).

Roadmap yêu cầu: (a) cho phép chỉ định thêm tham số ngoài `v12, v21` nhưng **KHÔNG bắt buộc** (người dùng chỉ cần v12/v21 vẫn dùng được bình thường); (b) chấm điểm lời giải toàn diện thay vì chỉ theo độ chính xác Poisson - độ chính xác/ổn định ưu tiên **cao**, khả năng chế tạo ưu tiên **trung bình**, thẩm mỹ ưu tiên **thấp**.

- **`volfrac`/`void_size_frac` làm condition optional**: 2 tham số DOE này vốn đã có sẵn miễn phí trong `outputs/phase3/*.npz` (`params`/`param_names`, không cần backfill dữ liệu) nhưng trước đây không được đưa vào condition vector của cVAE. `--extended-condition` (train.py) mở `condition_dim` từ 2 lên 6: `[v12, v21, volfrac, volfrac_mask, void_size_frac, void_size_frac_mask]`. "Optional" triển khai bằng **presence-mask + condition-dropout kiểu classifier-free-guidance** (`train.py::apply_condition_dropout`, `--optional-dropout-p`, mặc định 0,5) - không chỉ set sentinel=0, vì model cần phân biệt "không chỉ định" với "giá trị thật bằng 0". `volfrac` có thêm loss riêng rẻ (`losses.volfrac_consistency_loss` - suy trực tiếp từ `recon.mean()`, không cần surrogate/FE); `void_size_frac` chưa có loss riêng (chỉ học ngầm qua reconstruction, giống cách v12/v21 hoạt động trước khi có property-consistency loss).

```bash
# Train với condition mở rộng (fine-tune từ checkpoint hiện có):
python3 pipeline/phase5_cvae/train.py --extended-condition --lambda-volfrac 1.0 \
  --resume-from outputs/phase5/cvae_v2_finetuned.pt --output-name cvae_extended.pt

# Suy diễn - volfrac/void_size_frac optional, bỏ trống = không ràng buộc:
python3 pipeline/phase5_cvae/sample.py --ckpt outputs/phase5/cvae_extended.pt --v12 -0.5 --v21 -0.5
python3 pipeline/phase5_cvae/sample.py --ckpt outputs/phase5/cvae_extended.pt --v12 -0.5 --v21 -0.5 --volfrac 0.35
```

> Lưu ý fine-tune từ checkpoint `condition_dim=2` (như `cvae_v2_finetuned.pt`/`cvae_realphysics.pt`) sang `--extended-condition` (`condition_dim=6`): `train.py::resize_condition_dim_weights()` tự động mở rộng 3 layer FC phụ thuộc `condition_dim` (giữ nguyên cột base + v12/v21, random-init cột mới) thay vì crash khi `load_state_dict`. Xem [EXPERIMENT_LOG.md](../EXPERIMENT_LOG.md) mục 2026-08-03 cho bối cảnh (bug này từng khiến đúng lệnh khuyến nghị crash, đã sửa + có test regression).

- **Chấm điểm tổng hợp trong `best_of_n_eval.py`**: thay `argmin(|Δv12|)` thuần túy bằng `composite_score = w_accuracy·accuracy_score + w_manuf·manuf_score + w_aesthetic·aesthetic_score` (mặc định `0,6/0,3/0,1`, đúng thứ tự ưu tiên cao/trung bình/thấp; `--w-accuracy 1 --w-manuf 0 --w-aesthetic 0` để tái hiện hành vi gốc). `accuracy_score` chuẩn hóa min-max `|Δv12|` trong chính pool ứng viên; `manuf_score` là điểm graded (trung bình 3 cờ con `is_connected/min_feature_ok/periodic_ok`, không chỉ nhị phân `passes_all`); `aesthetic_score` (module `aesthetics.py`) = đối xứng (flip ngang/dọc) + độ trơn viền (tỉ lệ chu vi/diện tích), cố tình giữ rẻ/không cần model riêng vì là trục ưu tiên thấp nhất. `--require-manufacturable` (hard filter cũ) vẫn hoạt động song song, áp trước khi chấm composite.

```bash
python3 pipeline/phase5_cvae/best_of_n_eval.py --cvae-ckpt outputs/phase5/cvae_extended.pt \
  --v12 -0.5 --v21 -0.5 --volfrac 0.35 --n-samples 30 \
  --w-accuracy 0.6 --w-manuf 0.3 --w-aesthetic 0.1
```

- **[XONG 2026-08-05, chỉ ở mức Phase 4]** `f1=E₁₁/E₀, f2=E₂₂/E₀` (Pha B) đã backfill + surrogate 5-chiều đạt R²≈0,93-0,96 - xem mục 3 ở trên. **Chưa làm:** nối f1/f2 làm condition cho cVAE Phase 5 (mới chỉ Pha A - volfrac/void_size_frac - có ở Phase 5).

### 5.2. Vật liệu nền tùy chọn (ν0) - Giai đoạn A (2026-08-18 → 2026-08-19)

> Đã merge vào nhánh `substrate-material` (chưa merge `main` tại thời điểm viết mục này). `--include-nu0` **độc lập** với `--extended-condition` (mục 5.1) - có thể bật riêng hoặc cùng lúc, `condition_dim` composable ∈ `{2,4,6,8}`.

Roadmap Giai đoạn A (`docs/archive/PROJECT_PLAN.md` Nhóm 1) mở rộng bài toán từ "1 vật liệu nền cố định (`nu=0,3`)" sang "vật liệu nền là 1 trục thiết kế" - cho phép cVAE sinh hình học tối ưu cho **đúng** vật liệu nền mục tiêu thay vì luôn giả định thép/nhựa mặc định.

- **`CVAEDataset(include_nu0=...)`** thêm 2 cột `[nu0, nu0_mask]` vào cuối condition vector (cùng cơ chế presence-mask + condition-dropout kiểu classifier-free-guidance đã dùng cho volfrac/void_size_frac ở mục 5.1) - đòi hỏi dataset có field `nu` (`--data-dir outputs/phase3_a4`, `outputs/phase3/` mặc định KHÔNG có field này vì sinh trước Giai đoạn A).
- **Điều kiện bắt buộc trước đó (`real_physics.py`, A5):** differentiable-physics fine-tune (mục 5, bước "Fine-tune... differentiable-physics") trước đây chỉ nhận `nu`/`E0` là scalar dùng chung cho cả batch. A5 tách cache mesh topology (`(nelx,nely)`, phần đắt) khỏi việc dựng `Material(E0,Emin,nu)` (rẻ, ~97µs/lần, ~0,1-0,2% chi phí FE-solve) - cho phép mỗi mẫu trong batch có `nu`/`E0` riêng mà không mất tác dụng tăng tốc cache. `losses.py::real_physics_loss` nhận `nu0_col` để trích đúng ν0 per-sample từ condition (mask=1) thay vì `fe_params['nu']=0,3` cố định cho mọi mẫu.

```bash
# Train với ν0 optional (fine-tune từ checkpoint surrogate A4):
python3 pipeline/phase5_cvae/train.py --include-nu0 --data-dir outputs/phase3_a4 \
  --surrogate-path outputs/phase4/surrogate_a4_nu0.pt --resume-from outputs/phase5/cvae_a4_nu0_base.pt \
  --output-name cvae_a4_nu0_finetuned.pt --lambda-real-physics 20.0

# Suy diễn - ν0 optional, bỏ trống = không ràng buộc vật liệu nền:
python3 pipeline/phase5_cvae/sample.py --ckpt outputs/phase5/cvae_a4_nu0_finetuned.pt --v12 -0.5 --v21 -0.5 --nu0 0.35
python3 pipeline/phase5_cvae/best_of_n_eval.py --cvae-ckpt outputs/phase5/cvae_a4_nu0_finetuned.pt \
  --v12 -0.5 --v21 -0.5 --nu0 0.35 --data-dir outputs/phase3_a4 --n-samples 30
```

**2026-08-19 - A7, đo lợi ích thật (2-stage: base rồi fine-tune real-physics trên `outputs/phase3_a4/`, cùng quy trình bắt buộc từ 2026-07-25):**

| Checkpoint | R²(FE, n=300) | hit rate single-shot | frac manufacturable |
|---|---|---|---|
| `cvae_a4_control_finetuned.pt` (không ν0, `condition_dim=2`) | 0,9776 | 0,9967 | 0,312 |
| `cvae_a4_nu0_finetuned.pt` (có ν0, `condition_dim=4`) | **0,9843** | 0,9967 | **0,365** |

Khác A4 (surrogate Phase 4, mục 4) - nơi thêm ν0 KHÔNG cho lợi ích R² đo được - ở tầng cVAE lợi ích **nhất quán trên cả 2 trục**: chính xác Poisson (ΔR²=+0,0067) và khả năng chế tạo (Δfrac_manufacturable=+0,053), nhờ differentiable real-physics loss dùng đúng ν0 per-sample khi tính gradient fine-tune (property-consistency loss của surrogate không có cùng độ chính xác vật lý). **Giai đoạn A coi là hoàn thành đầy đủ về khoa học** từ mốc này - hạ tầng thông suốt Phase 1→5 + lợi ích đo được, không chỉ hạ tầng nằm im. Checkpoint kết quả: `outputs/phase5/cvae_a4_nu0_finetuned.pt` (**chưa promote** thành checkpoint production mặc định thay `cvae_v2_finetuned.pt` - cần quyết định riêng có "chốt" dataset A4 làm production hay không).

Lần chạy A7 cũng phát hiện + sửa 3 bug hạ tầng chặn cứng (cùng root cause: A4-A6 mới nối tới 1-2 call site đã có unit test, chưa từng chạy end-to-end với checkpoint `include_nu0=True` thật) - nghiêm trọng nhất: `best_of_n_eval.py` verify FE dưới `FE_PARAMS['nu']=0,3` cố định cho MỌI condition kể cả khi target ν0 thật khác 0,3, làm sai chính con số dùng để kết luận thí nghiệm. Chi tiết đầy đủ 3 bug: [EXPERIMENT_LOG.md](../EXPERIMENT_LOG.md) mục 2026-08-19.
