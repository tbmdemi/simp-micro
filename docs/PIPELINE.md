# Pipeline chi tiết: Screening → Multi-Batch DOE → Dataset → Surrogate → cVAE

> Tài liệu này mô tả đầy đủ từng bước của pipeline 8-phase, tách ra từ `README.md` để giữ README ngắn gọn. Xem [README.md](../README.md#trạng-thái-dự-án) cho bảng trạng thái tổng quan, và [LIMITATIONS.md](LIMITATIONS.md) cho phạm vi claim khoa học + giới hạn đã biết.

## 1. LHS Screening (Phase 1)

Quét không gian tham số (`volfrac`, `penal`, `rmin`, `move`, `void_size_frac`, `rotation_deg`) bằng Latin Hypercube Sampling.

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
- Target xuất ra: `v12`, `v21`, `volfrac_achieved`. `f1, f2` (roadmap gốc) chưa khả dụng - cần mở rộng `compute_homogenized_tensor()` để xuất độ cứng chuẩn hóa trước.

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

- **Chưa làm** (xem [EXPERIMENT_LOG.md](../EXPERIMENT_LOG.md)): `f1=E₁₁/E₀, f2=E₂₂/E₀` (Pha B, mục 5 [Giới hạn Đã biết](LIMITATIONS.md#giới-hạn-đã-biết--known-limitations)) suy được từ tensor `Q` mà `run_simp()` đã trả về nhưng chưa từng lưu xuống đĩa cho dataset hiện có - cần backfill 1 FE-solve/mẫu (rẻ, không phải chạy lại DOE) + train lại Phase 4 surrogate (hiện chỉ xuất 3 chiều `[v12,v21,volfrac]`) + kiểm tra multicollinearity (f1/f2 cùng suy từ `Q` với v12/v21) - để lại như việc kế tiếp.
