# KẾ HOẠCH DỰ ÁN AUXFORGE (v3 — 2026-09-25)

> **Đây là nguồn kế hoạch + trạng thái DUY NHẤT.** `task_progress.md` chỉ còn là log chi tiết
> từng lần chạy (lịch sử); `PROJECT_PLAN.md` là roadmap cũ (ν0/retrieval, 2026-08) đã được thay
> thế. Mọi thay đổi trạng thái cập nhật trực tiếp vào bảng ở mục 4 của file này.
> Bản v2 (23-24/08, KAN/WIRE/Mamba) xem trong git history của file này.

---

## 1. Vì sao có v3 — thay đổi so với v2

### 1.1 Phát hiện: luận điểm "KAN vượt Linear ~2×" là artifact của thước đo

> **Đính chính 2026-09-25 (cùng ngày viết v3):** bản đầu của mục này nói `cvae_v2_finetuned.pt` "chưa
> từng fine-tune real-physics" - **SAI**. Checkpoint này đã fine-tune real-physics 2 giai đoạn trên
> dataset v2 (`PIPELINE.md` mục 2026-07-25; `fe_r2` lúc train 0,905; R²(FE) oracle best-of-30,
> n=300 = 0,995). Baseline v2 dùng là đúng checkpoint; cái sai là **thước đo**.

v2 so KAN với Linear bằng `property_accuracy()` - sinh 1 mẫu/condition rồi chấm bằng **surrogate**
(chính `evaluate.py` cảnh báo con số này "KHÔNG đáng tin"). Trên thước đo đó, checkpoint Linear
production chỉ đạt 0,27/0,46 dù đo bằng FE thật (best-of-30) đạt 0,995. Đo lại 2026-09-25, cùng
test set, cùng surrogate v2, seed 123:

| Checkpoint (đều đã fine-tune real-physics) | R² v12 (surrogate) | R² v21 (surrogate) | `fe_r2` tốt nhất lúc train |
|---|---:|---:|---:|
| `cvae_v2_finetuned` (Linear, dataset v2, **production**) | 0,267 | 0,460 | 0,905 |
| `cvae_realphysics` (Linear, dataset v1) | 0,793 | **0,739** | **0,921** |
| `cvae_kan_realphysics_v2` (KAN, dataset v2) | **0,836** | 0,476 | 0,889 |

→ Thước đo surrogate xếp hạng ngược với FE thật (v2_finetuned đứng chót ở surrogate nhưng FE oracle
0,995), nên không dùng được để kết luận KAN hơn Linear. Trên thước đo FE lúc train, KAN (0,889) chưa
vượt 2 checkpoint Linear. Thêm 2 confound chưa kiểm soát: `gamma` khác nhau (20 vs 1) và code khác
nhau. Exp 3 (MLP, R²(FE)=−9,8) dùng recipe 1-giai-đoạn `lambda_real_physics=20` (đã biết thất bại,
`PIPELINE.md` 2026-07-25) VÀ train với `enforce_symmetry=True` mặc định mới (xem 1.4) → không phải
bằng chứng cho KAN.

### 1.2 Trạng thái thật các task v2

| Task v2 | Kết quả | Quyết định v3 |
|---|---|---|
| 1 KAN head | Đạt DoN v2 nhưng DoN đo bằng metric surrogate (`evaluate.py` tự cảnh báo "KHÔNG đáng tin"), xếp hạng ngược FE thật | Làm lại thành **ablation có kiểm soát** (P1.2) |
| 2 WIRE decoder | Thất bại 3 checkpoint, tệ dần (hit-rate 4,2% → 0%, R²(FE) −10,3 → −13,9) | **1 lần thử cuối** có tiêu chí dừng (P1.3), không đạt → đóng, ghi vào Limitations |
| 3 ConvKAN surrogate | R² thấp hơn Linear (0,964/0,957/0,916 vs 0,974/0,964/0,983); tham số **tăng** 65% | **Tùy chọn** (P1.4). Lý do tồn tại ("gradient mượt cho L-BFGS") đã mất vì Task 4 giờ lấy gradient từ FE thật |
| 4 Physics-guided refinement | MAE(FE) giảm ~40% trên 24 condition — kết quả dương duy nhất, đo bằng FE độc lập | **Trọng tâm Bài báo #1** (P1.1) |
| Phần II (MNO/KINN/ICKAN) | Mới có khung code, chưa đo được KPI nào, thiếu điều kiện tiên quyết | **Đóng băng** tới khi Bài #1 có bản thảo (mục 5) |

### 1.3 Định vị lại Bài báo #1

- **Cũ:** "Inverse design of continuous auxetic metamaterials via WIRE INR and KAN" — 2/4 trụ cột
  thất bại, trụ cột thứ 3 (KAN) chưa có bằng chứng bằng thước đo FE thật.
- **Mới (tên làm việc):** *"Physics-guided latent refinement for inverse design of auxetic
  metamaterials with a conditional VAE and differentiable homogenization"* — trọng tâm là cVAE +
  tối ưu latent bằng gradient FE khả vi, kiểm chứng FE độc lập; KAN là 1 ablation (báo cáo
  trung thực kể cả khi hòa); WIRE là kết quả âm tính trong Limitations.

### 1.4 Bug loader phát hiện khi làm P1.0 (đã sửa, `LIMITATIONS.md` mục 28)

`adversarial_dataset.load_cvae()` (dùng bởi `best_of_n_eval`, `self_play`, benchmark refinement...)
bỏ qua `use_kan`/`enforce_symmetry` của checkpoint từ 2026-09-11 → checkpoint KAN cũ bị ép đối xứng
lúc đánh giá, checkpoint Linear crash. Benchmark refinement 2026-09-23 (KAN v2, `IN24`) đo lại đúng:
R²(v12) baseline **0,058** (không phải −16,00), refined **0,92** (không phải −7,60), MAE giảm 68%
[CI95 57-76%]. Mọi số đo qua `load_cvae` trên checkpoint cũ trong khoảng 2026-09-11 → 2026-09-25
cần đo lại trước khi trích dẫn. Ngoài ra: chỉ 2,5% ảnh train đối xứng qua đường chéo, nên decoder
`enforce_symmetry=True` (mặc định train mới) không biểu diễn được dữ liệu dị hướng - P1.2 phải chạy
`--disable-symmetry`.

---

## 2. Nguyên tắc đo lường chung (áp cho mọi task)

1. **Baseline chuẩn:** `cvae_v2_finetuned.pt` (Linear + real-physics, cùng dataset v2 với mọi
   checkpoint mới). `cvae_realphysics.pt` (dataset v1) là baseline phụ.
2. **Metric chính = FE thật**, không phải `property_accuracy()` (qua surrogate). Surrogate R² chỉ
   được ghi làm số phụ.
3. **Tập condition cố định** (seed 123) cho mọi so sánh: `IN24` (24 condition hiện dùng ở
   `best_of_n_eval.py`, giữ để so với số cũ) + `IN100` (100 condition test set, để có CI đủ hẹp).
4. **Bootstrap CI 95%** (`bootstrap_ci.py`) cho mọi hiệu số được dùng làm kết luận. "Thắng" chỉ
   khi CI của hiệu số không chứa 0.
5. **Tiêu chí dừng ghi TRƯỚC khi chạy** (đã ghi trong bảng mục 3) — không hạ ngưỡng sau khi có số
   (lỗi v2 đã mắc: DoN Task 1 hạ từ 0,990 xuống 0,60 sau khi có kết quả).
6. **Chọn checkpoint theo `fe_r2`**, không theo `val_loss` (`LIMITATIONS.md` mục 12).

---

## 3. Bài báo #1 — các task

### P1.0 Harness benchmark chung (điều kiện cần cho P1.1-P1.3)

Mở rộng `pipeline/phase5_cvae/benchmark_physics_guided_refinement.py`: thêm `--targets`
(danh sách ν* tùy ý cho OOD), `--n-samples` (kết hợp best-of-N), xuất bootstrap CI; tái dùng
`best_of_n_eval.py`/`bootstrap_ci.py`, không viết lại logic FE. Kèm test trong `tests/`.

- **DoN:** chạy lại được đúng số 2026-09-23 (MAE 0,587 → 0,355 trên `IN24`, KAN v2) — xác nhận
  harness mới không đổi hành vi cũ.
- **ETA:** ~0,5 ngày (code + test), GPU ~30 phút.
- **✅ XONG 2026-09-25:** với loader cũ, baseline tái lập bit-for-bit (MAE 0,5868); refined 0,384
  (cũ 0,355 - L-BFGS trên GPU không tất định, 2 lần chạy liên tiếp lệch R² ~0,006). Thêm
  `--targets/--target-v21/--n-samples/--force-periodic/--n-boot`, CI paired
  (`bootstrap_ci.bootstrap_paired_mae_reduction`), 11 test mới. Phát hiện + sửa bug loader (mục 1.4)
  - mọi số từ đây dùng loader đã sửa.

### P1.1 Physics-guided latent refinement (TRỌNG TÂM)

| Bước | Nội dung | Tiêu chí (đặt trước) | ETA |
|---|---|---|---|
| a | ✅ **ĐẠT 2026-09-25** (xem bảng dưới). Refinement trên **baseline production** `cvae_v2_finetuned.pt` + `cvae_realphysics.pt` (không chỉ KAN), `IN100`, 1 mẫu/condition | Đạt nếu MAE(FE) giảm ≥ 20% với CI không chứa 0 | GPU ~1,5-2 giờ |
| b | Refinement **kết hợp** best-of-N (N=30 + `force_periodic`, đúng pipeline production) | Đạt nếu hit-rate và R²(FE) không giảm so với best-of-N không refine; báo cả số FE calls để so chi phí | GPU ~2-3 giờ |
| c | ✅ **XONG 2026-09-25 (giải tích, không cần chạy SIMP):** vật liệu trực hướng 2D cần ν12·ν21 < 1 → target đối xứng ν12=ν21=ν*≤−1 **không tồn tại**. Dữ liệu khớp: mẫu cực trị v12=−1,95 đi với v21=−0,04; max(v12·v21) trong train = 0,43 | Target OOD phải **dị hướng** | – |
| d | Quét OOD dị hướng v12=ν* ∈ {−1,0; −1,5; −1,75; −2,0; −2,25; −2,5}, v21=−0,05 (`--targets ... --target-v21 -0.05`), mỗi target 10 lần (z khác nhau), best-of-30 + `force_periodic`; báo cáo **đường cong sai số tương đối v12 theo khoảng cách ra ngoài dải train** | Đạt nếu sai số tương đối v12 ≤ 8% tại ν*=−2,0 (sát biên train). ν*≤−2,25 là **thăm dò**: báo số thật, không đặt ngưỡng (memory 2026-08-04: chưa model nào ngoại suy được biên độ) | GPU ~1-2 giờ |

**Kết quả bước a (2026-09-25, `IN100`, 30 bước L-BFGS, 1 mẫu/condition, loader đã sửa;
`outputs/phase5/plan_v3/p1_1a_*_in100.json`):**

| Checkpoint | R²(v12) trước → sau | MAE(v12) trước → sau | Giảm MAE(v12) [CI95] | Tiêu chí ≥20% |
|---|---|---|---|---|
| `cvae_v2_finetuned` (production) | 0,874 → **0,976** | 0,052 → 0,022 | **57,7%** [49,4; 64,6] | ✅ |
| `cvae_realphysics` | 0,893 → 0,931 | 0,043 → 0,035 | 19,4% [0,2; 35,0] | ⚠️ sát ngưỡng (v12+v21: 23,1% [6,4; 36,8]) |
| `cvae_kan_realphysics_v2` | 0,675 → 0,942 | 0,077 → 0,033 | 57,1% [49,5; 64,6] | ✅ |

Chi phí: ~50 FE-solve/condition cho refine (so với 1 cho baseline). Ghi chú: R²(v12) single-shot
bằng FE thật của `cvae_v2_finetuned` là 0,874, trong khi `property_accuracy()` (surrogate) báo 0,27
- xác nhận thêm thước đo surrogate không dùng được (mục 1.1).

**Kết quả bước b, d (2026-09-25, `cvae_v2_finetuned`, best-of-30 + `force_periodic`) - KHÔNG ĐẠT:**

- **b** (`IN100`): refine SAU best-of-30 làm TỆ hơn - MAE(v12) 0,0152 → 0,0246 (−62,5% [CI95
  −98,3; −33,9]), R²(v12) 0,986 → 0,968. Best-of-30 đơn thuần đã rất mạnh (R² 0,986, sai số tương
  đối v12 5,2%).
- **d** (quét OOD dị hướng, 10 lặp/target): best-of-30 KHÔNG refine đạt sai số tương đối v12 trung
  vị 0,5% / 0,9% / 4,1% tại −1,0 / −1,5 / −1,75, nhưng **bão hòa ở ~−1,95** (= min tập train):
  11% / 16% / 23% tại −2,0 / −2,25 / −2,5. Refine làm tệ hơn ngoài dải (36-44%). DoN ≤8% tại −2,0:
  không đạt (tốt nhất 11%, không refine).
- **Nguyên nhân (đã xác nhận):** refine tối ưu trên ảnh mật độ **liên tục** (resize bilinear), còn
  verify làm **force_periodic → nhị phân hóa → resize nearest**. Loss liên tục cuối ~1e-10 (khớp
  target gần tuyệt đối, kể cả ν*=−2,5) nhưng MSE sau nhị phân hóa 1e-3-1e-2 → refine **khai thác
  vật liệu xám** (cùng họ với surrogate exploitation). Ở single-shot (bước a) lợi vẫn lớn hơn hại;
  khi baseline đã tốt (b) hoặc ngoài dải (d) thì hại trội.

| e | **Refine nhận thức nhị phân hóa** (thêm 2026-09-25 sau b/d): trong objective áp `force_periodic` (torch) → Heaviside projection khả vi (η=0,5, β tăng dần {1,4,16,64}, L-BFGS khởi động lại mỗi mức β) → resize `nearest-exact` khớp verify; + chế độ **guarded** (chỉ nhận refine khi FE verify tốt hơn baseline). Chạy lại a, b, d | **Đặt trước khi chạy:** (1) a: giảm MAE(v12) ≥ 20%, CI không chứa 0 (như cũ); (2) b, bản KHÔNG guarded: MAE không tăng có ý nghĩa (cận trên CI của mae_diff ≥ 0); bản guarded: báo mức giảm + CI; (3) d: sai số tương đối v12 trung vị ≤ 8% tại −2,0. Không đạt (2)-(3) → refinement chỉ được claim cho chế độ single-shot | ~0,5 ngày |

**Kết quả bước e (2026-09-25, `cvae_v2_finetuned`, 32 bước, β {1,4,16,64};
`outputs/phase5/plan_v3/p1_1e_*.json`):**

| Bước | Trước (ảnh liên tục) | **Sau e (nhận thức nhị phân hóa)** | Guarded | Tiêu chí đặt trước |
|---|---|---|---|---|
| a (single-shot, `IN100`) | MAE −57,7% | **MAE −91,8% [89,0; 93,9]**, R²(v12) 0,874 → **0,998**, sai số tương đối v12 14,4% → 2,6% | như bản thường (nhận 100%) | ✅ (1) |
| b (sau best-of-30, `IN100`) | MAE +62% (tệ hơn) | **MAE −62,6% [50,7; 72,1]**, R²(v12) 0,986 → 0,997 | −69,1% [61,6; 75,4], nhận 92% | ✅ (2) |
| d (OOD, sai số tương đối v12 trung vị) | −2,0: 36% | −1,75: 0,3% · **−2,0: 12,5%** · −2,25: 22% · −2,5: 32% | **−2,0: 4,9%** · −2,25: 14,8% · −2,5: 21,9% | (3): ❌ bản thường, ✅ guarded |

- **Kết luận:** refine nhận thức nhị phân hóa là **đóng góp chính của Bài #1**: trong dải train
  đưa R²(FE) lên ~0,998 ở cả single-shot lẫn sau best-of-30. Ngoài dải (OOD): bản thường vẫn
  bão hòa; chế độ guarded (luôn FE-verify rồi chọn, không tốn thêm FE) đạt ≤8% tại −2,0 nhưng
  vẫn bão hòa từ −2,25 - **không claim ngoại suy biên độ** (khớp memory 2026-08-04). n=10/target
  ở d nên số OOD chỉ là định hướng.
- **Chi phí:** ~77 FE-solve/condition cho refine (baseline best-of-30: 30).

- **Tổng ETA P1.1:** ~1,5-2 ngày (phần lớn là chờ GPU; bước c chạy CPU song song với a).
- **Nếu bước a không đạt:** refinement chỉ giúp model yếu (KAN), không giúp production → luận điểm
  chính của bài phải xem lại; dừng lại để quyết định trước khi làm P1.2.

### P1.2 Ablation KAN vs Linear (có kiểm soát)

Thiết kế 2×2 trên **code hiện tại**, cùng recipe 2 giai đoạn (base → fine-tune real-physics), cùng
`gamma`, cùng seed, chỉ khác head (`--use-mlp-head` bật/tắt):

| | Không real-physics (base) | Có real-physics (fine-tune) |
|---|---|---|
| Linear | train mới | train mới (không dùng lại `cvae_v2_finetuned.pt` vì code cũ, `gamma`=20) |
| KAN | train mới | train mới |

- Cả 4 run dùng `--disable-symmetry` (mục 1.4: dữ liệu 97,5% không đối xứng qua đường chéo).
- **Recipe (chốt 2026-09-25 trước khi chạy):** dùng đúng recipe 2 giai đoạn đã tạo ra checkpoint
  production (`PIPELINE.md`), KHÔNG dùng recipe KAN v2 - lý do: đây là recipe cho baseline mạnh nhất
  đã kiểm chứng (R²(FE) oracle 0,995), so sánh công bằng hơn khi cả 2 head cùng dùng recipe tốt nhất
  đã biết. Chung: `--disable-symmetry --gamma 20 --surrogate-path
  outputs/phase4/surrogate_for_phase5_v2.pt --n-fe-eval-conditions 24` (chọn checkpoint theo
  `fe_r2` trên 24 condition **validation**, tăng từ 8 - bài học WIRE overfit tập chọn nhỏ).
  Base: `--epochs 50 --fe-eval-every 5 --seed 123`. Fine-tune: `--resume-from <base> --epochs 35
  --kl-warmup 1 --lambda-real-physics 20 --real-physics-subsample 8 --real-physics-every 2
  --real-physics-workers 6 --fe-eval-every 2 --seed {123,7}`. Linear = `--use-mlp-head`.
  Checkpoint: `outputs/phase5/p12_{linear,kan}_{base,rp_s123,rp_s7}.pt`.
- **Đánh giá:** harness P1.0 trên `IN100`, 2 chế độ: single-shot và best-of-30 + `force_periodic`,
  cả 2 kèm refine P1.1e. So sánh head bằng CI paired theo condition (sai số |Δv12| của Linear vs
  KAN, cùng condition).
- Đo thử 1 epoch base: Linear ~35 s, KAN ~55 s → ETA thực tế thấp hơn ước tính ban đầu (mục 4).
- Ô "có real-physics" chạy **2 seed** (123, 7) — đây là so sánh sẽ vào bài báo nên cần kiểm tra độ
  ổn định. Ô "base" chạy 1 seed.
- Đo: best-of-N FE trên `IN100` + CI; kèm param count, thời gian/epoch.
- **Tiêu chí (đặt trước):** KAN "thắng" chỉ khi Δ R²(FE) > 0, CI không chứa 0, trên cả 2 seed.
  Nếu hòa (CI chứa 0): production giữ **Linear**.
  *(Đánh đổi 60/40: khi độ chính xác ngang nhau, KAN tốn ~3,2× tham số head → ưu tiên hiệu năng.)*
- **ETA:** GPU ~12-16 giờ (4 base/fine-tune + 1 seed phụ, ~2-3 giờ/run) + eval ~3 giờ →
  **~3-4 ngày lịch**, chạy nền qua đêm.

**Kết quả P1.2 (2026-09-25, `IN100`, refine P1.1e; `outputs/phase5/plan_v3/p12_eval_*.json`):**

| Chế độ | Head | R²(v12) s123 | R²(v12) s7 | KAN vs Linear, giảm MAE(v12) [CI95] s123 / s7 |
|---|---|---|---|---|
| Single-shot | Linear / KAN | 0,852 / 0,904 | 0,815 / 0,911 | +13,3% [−0,2; 24,4] / **+18,2% [3,8; 30,4]** |
| Single-shot + refine | Linear / KAN | 0,994 / 0,997 | 0,998 / 0,999 | +33,2% [−15,1; 60,5] / **+33,9% [11,7; 52,3]** |
| Best-of-30 + fp | Linear / KAN | 0,977 / 0,974 | 0,982 / 0,980 | **−25,2% [−56,2; −1,0]** / **−19,9% [−42,1; −1,7]** |
| Best-of-30 + fp + refine | Linear / KAN | 0,998 / 0,997 | 0,998 / 0,999 | +11,3% [−38,7; 43,6] / **+44,1% [24,9; 59,7]** |

- **Theo tiêu chí đặt trước (thắng khi CI > 0 trên CẢ 2 seed): KAN KHÔNG thắng ở chế độ nào.**
  Ở chế độ production (best-of-30 không refine), **Linear thắng có ý nghĩa trên cả 2 seed**. Có
  refine: cả 2 head đạt R² ~0,997-0,999, khác biệt tuyệt đối không đáng kể.
- KAN có xu hướng tốt hơn ở single-shot (có ý nghĩa ở 1/2 seed) - báo cáo như xu hướng, không claim.
- **Quyết định: production giữ Linear.** *(Đánh đổi 60/40: độ chính xác production ngang/Linear
  nhỉnh hơn; KAN tốn 3,2× tham số (4,79M vs 1,49M) và ~1,6× thời gian/epoch → chọn Linear.)*
- `fe_r2` validation lúc train: Linear 0,929/0,920, KAN 0,932/0,938. Checkpoint:
  `outputs/phase5/p12_{linear,kan}_rp_s{123,7}.pt` (không promote - best-of-30 thô của Linear mới
  0,977-0,982 thấp hơn `cvae_v2_finetuned` 0,986, sàn cứng `CLAUDE.md`).

### P1.3 WIRE — lần thử cuối rồi đóng

Chạy đúng 1 lần phương án chưa thử duy nhất (Heaviside projection khả vi + real-physics, recipe 2
giai đoạn giống KAN) — chi tiết kỹ thuật ở Phụ lục A, phần B.

- **Tiêu chí dừng (đặt trước):** tiếp tục đầu tư WIRE CHỈ KHI single-shot hit-rate ≥ 30% (ngang
  `frac_manufacturable` 0,30-0,35 của conv production) VÀ R²(FE) > 0 trên `IN24`. Không đạt →
  đóng Task WIRE, viết 1 đoạn Limitations (INR/Gabor không hợp với trường mật độ nhị phân hóa ở
  64²) — không retrain thêm.
- Mục tiêu 75% của v2 bị bỏ: không có căn cứ, cao hơn cả baseline conv production.
- **ETA:** code Heaviside ~2,5 giờ + GPU ~3-3,5 giờ + eval ~45 phút → **~1 ngày**. Làm **sau**
  P1.1/P1.2 (không chặn bài báo).

**Recipe P1.3 (chốt 2026-09-25 trước khi chạy):** Heaviside đưa vào **real-physics loss lúc train**
(`--rp-projection-beta-max 64 --rp-periodic`, β tăng hình học 1→64 qua các epoch, dùng chung
`heaviside.project_for_fe` với refine P1.1e) thay vì sửa `CVAE.forward` - gọn hơn và đúng cơ chế
đã chứng minh ở P1.1e. Khác Phụ lục B: `--disable-symmetry` (mục 1.4). Chung: `--decoder-type wire
--wire-hidden-dim 128 --lr 1e-4 --gamma 20 --surrogate-path outputs/phase4/surrogate_for_phase5_v2.pt
--n-fe-eval-conditions 24`. Base: `--epochs 25 --fe-eval-every 5` → `p13_wire_base.pt`. Fine-tune:
`--resume-from <base> --epochs 25 --kl-warmup 1 --lambda-real-physics 20 --real-physics-subsample 8
--real-physics-every 2 --real-physics-workers 8 --fe-eval-every 2` + projection → `p13_wire_rp.pt`.
Đánh giá: `best_of_n_eval.py --n-conditions 24 --n-samples 30` (hit-rate single-shot + R²(FE)) +
harness P1.0 `IN24`. Tiêu chí dừng giữ nguyên (hit-rate single-shot ≥ 30% VÀ R²(FE) > 0).

**Kết quả P1.3 (2026-09-25) - KHÔNG ĐẠT → ĐÓNG hướng WIRE** (`outputs/phase5/plan_v3/p13_wire_*.json`,
checkpoint `outputs/phase5/p13_wire_rp.pt`, `fe_r2` validation tốt nhất −5,64):

| Chỉ số (`IN24`) | P1.3 | Tiêu chí | WIRE trước (v2 / v3 round2) | Conv production |
|---|---|---|---|---|
| hit-rate single-shot | 12,5% | ≥ 30% ❌ | 4,2% / 0% | ~99,7% |
| R²(FE, best-of-30) | −7,19 | > 0 ❌ | −10,3 / −13,9 | ~0,98-0,99 |
| R²(v12) sau refine P1.1e | −1,20 (MAE −65%) | – | – | 0,998 |
| frac manufacturable | 4,6% | – | 4,2% | ~25-35% |

Tốt nhất trong mọi lần thử WIRE (Heaviside + tắt đối xứng + 2 giai đoạn đều có ích) nhưng còn rất
xa tiêu chí dừng đặt trước → không retrain thêm; ghi vào Limitations của Bài #1 như kết quả âm tính
(INR Gabor không cạnh tranh được decoder conv trên trường mật độ nhị phân 64²).

### P1.4 ConvKAN surrogate — tùy chọn

Chỉ chạy nếu GPU rảnh: 1 run `grid_size=8` (Phụ lục A, phần A). Chỉ promote khi ≥ Linear trên cả
3 đầu ra. Bỏ KPI "giảm ≥ 80% tham số" của v2 (sai về thiết kế: KAN có nhiều tham số hơn Linear).
- **ETA:** ~1,5-2 giờ code + GPU ~1 giờ. Không nằm trên đường găng.

### P1.5 Viết lại bản thảo Bài báo #1

`docs/paper1_{methods,results,discussion}_draft.md` hiện viết trên khung so sánh sai (baseline
0,35, tiêu đề WIRE+KAN) → viết lại theo mục 1.3 bằng số của P1.1-P1.3. Tái dùng được: phần
Methods về FE/real-physics, phát hiện "val_loss ≠ property fidelity", phát hiện baseline 0,85
không tái lập (giữ, là bài học phương pháp có giá trị).
- **ETA:** ~4-5 ngày sau khi có đủ số.
- **✅ BẢN THẢO ĐẦU XONG 2026-09-30:** `docs/paper1/` (LaTeX EN + VI, PDF, 41 tài liệu tham
  khảo đã đối chiếu DOI, 7 hình/6 bảng, mọi số truy được về JSON - xem `docs/paper1/README.md`).
  Tên bài (tác giả chốt): *"Deep Generative Inverse Design of Auxetic Metamaterials: Navigating
  Multi-Objective Demands and Out-of-Distribution Generalization"* - Trịnh Bình Minh. Từ khóa:
  Auxetic Metamaterials; Deep Generative Models; Differentiable Physics; Inverse Design; Topology
  Optimization; Homogenization. Khung bài: (1) độ chính xác kiểm chứng FE (P1.1e), (2) đa mục tiêu
  (composite score vs Pareto, Nhóm 3.2), (3) OOD tách **dấu** (Nhóm 4.1, cVAE thắng retrieval) và
  **biên độ** (P1.1d, bão hòa ở biên train), (4) ablation KAN/WIRE (P1.2, P1.3).
- **Phát hiện khi viết (2026-09-30):** cả 6 mục tiêu nhóm `extreme_auxetic` của thí nghiệm OOD
  2026-08-04 vi phạm ν12·ν21 < 1 (vd (−2,1; −2,1)) → không tồn tại về vật lý, thất bại của nhóm đó
  không nói gì về ngoại suy. Bài chỉ dùng nhóm đổi dấu (hợp lệ, 6/6 đúng dấu). Đã đính chính
  `LIMITATIONS.md` mục 19 + phần Phạm vi Claim + thêm mục 29 (refine) và 30 (KAN/WIRE) ngày 2026-09-30.

### P1.6 Đánh giá khả năng đăng bài + việc cần làm trước khi nộp (2026-09-30)

**Đánh giá nếu nộp ngay:** tạp chí top (npj Comput. Mater., Nat. Commun.) - thấp; Q1 chuyên ngành
phù hợp - trung bình, nhiều khả năng *major revision*. (Đánh giá định tính, không có số liệu tỉ lệ
chấp nhận.)

- **Điểm mạnh:** đo lường chặt (FE thật, bootstrap CI ghép cặp, tiêu chí đặt trước); cơ chế rõ có
  bằng chứng trực tiếp (khoảng lệch objective-kiểm chứng 6 bậc); báo trung thực kết quả âm; phát
  hiện benchmark "cực trị" dùng mục tiêu không tồn tại.
- **Điểm yếu reviewer chắc chắn hỏi:**
  - 🔴 Thiếu baseline **SIMP chạy từ đầu** cho cùng mục tiêu ("sao không tối ưu tô-pô trực tiếp?").
  - 🔴 Tính mới vừa phải: Heaviside projection + tối ưu latent đều đã có; đóng góp là kết hợp đúng.
  - 🟠 OOD mỏng: đổi dấu chỉ 6 mục tiêu, checkpoint cũ (dataset v1); biên độ là kết quả âm → tên bài
    có nguy cơ bị xem là claim quá.
  - 🟠 Chỉ 2D, đàn hồi tuyến tính, lưới 50×50, không thực nghiệm.
  - 🟡 Composite score trượt ngưỡng 0,7 đặt trước (TB 0,693); aesthetic là heuristic; chưa đo refine
    có làm giảm khả năng chế tạo không.

**Nơi nộp (mức phù hợp):** workshop NeurIPS AI4Mat / ICLR ML4Materials (cao, non-archival, nộp thử
trước) · *Structural and Multidisciplinary Optimization* (khá - hợp góc TO + projection) ·
*Materials & Design* / *Computational Materials Science* (khá; M&D mạnh hơn nhiều nếu có thực
nghiệm) · *CMAME* (trung bình, cần baseline SIMP + chiều sâu toán) · *Composite Structures* (thấp
với bản hiện tại - hợp chuyên môn GS Đức nhưng thiếu góc composite) · npj/Nat. Commun. (thấp).

**Khuyến nghị:** làm P1.6a-c (~2 ngày) → nộp SMO hoặc Materials & Design + song song bản rút gọn
cho workshop AI4Mat. Bàn với GS Nguyễn Đình Đức về đồng tác giả và tạp chí đích trước khi nộp.

| # | Việc | Lý do | ETA |
|---|---|---|---|
| P1.6a | **Baseline SIMP từ đầu** ~20 mục tiêu IN100: so độ chính xác, số lần giải FE, thời gian | Quan trọng nhất - chứng minh lợi ích chi phí | ~1 ngày |
| P1.6b | Chạy lại OOD đổi dấu với `cvae_v2_finetuned.pt`, 20-30 mục tiêu, có CI | Củng cố trục OOD của tên bài | ~0,5 ngày (GPU 1-2 giờ) |
| P1.6c | Đo manufacturability sau refine + hình ô cơ sở trước/sau refine, lưới lát 3×3 | Lỗ hổng dễ bị hỏi, rẻ để lấp | ~0,5 ngày |
| P1.6d | Trao đổi GS Đức: đồng tác giả, tạp chí đích, góc composite/FGM | Tăng khả năng qua vòng biên tập | – |
| P1.6e | Công bố mã + dữ liệu (GitHub + DOI Zenodo) | Nhiều tạp chí yêu cầu | ~0,5 ngày |
| P1.6f | (Dài hạn) In 3D vài thiết kế + đo ν thật, hoặc kiểm chứng FE thương mại/biến dạng lớn | Nâng hạng bài với tạp chí vật liệu | tuần |

**Kết quả P1.6a (2026-10-06, IN100, ghép cặp theo condition;
`pipeline/phase5_cvae/benchmark_simp_baseline.py`, so sánh bằng
`docs/paper1/scripts/p1_6a_compare.py` → `outputs/phase5/plan_v3/p1_6a_comparison.json`):**
SIMP inverse homogenization ½‖ν−ν*‖², MMA, ràng buộc thể tích ≤ 0,55 + Q11,Q22 ≥ δ, Heaviside
continuation β 1→64, verify nhị phân hóa + FE độc lập giống cVAE; đa khởi tạo 4 seed (hourglass,
hexagonal, reentrant_bowtie, circle); 2 ngân sách: `matched` ≤105 FE/seed, `full` ≤301 FE/seed.

| Phương pháp | FE/target | R²(ν12) | MAE ν12 | R²(ν21) | MAE ν21 | MAE cặp | Chế tạo được |
|---|---:|---:|---:|---:|---:|---:|---:|
| cVAE best-of-30 + refine guarded | 107 | **0,998** | **0,0047** | 0,838 | 0,0108 | 0,0155 | ~0,25 (chưa đo sau refine, P1.6c) |
| SIMP 1 seed (hourglass), `matched` | 106 | 0,970 | 0,0159 | 0,984 | 0,0130 | 0,0289 | – |
| SIMP best-of-4 seed, `matched` | 344 | 0,980 | 0,0093 | 0,993 | 0,0070 | 0,0163 | 0,84 |
| SIMP best-of-4 seed, `full` | 897 | 0,997 | 0,0071 | **0,997** | **0,0057** | **0,0128** | 0,77 |

- **ν12:** cVAE giảm MAE 34% [CI95 13; 50] so với SIMP `full` (ít hơn 8,4× FE), 71% [61; 77] so
  với SIMP cùng ngân sách 1 seed.
- **Cặp (ν12, ν21):** cVAE hòa SIMP đa khởi tạo (vs `matched`: +4,6% [−37; 44]; vs `full`: −21%
  [−143; 42], CI chứa 0), thắng SIMP cùng ngân sách 1 seed 46% [10; 72]. Trung vị sai số cặp cVAE
  0,0066 < SIMP 0,0102 nhưng đuôi nặng hơn - **ν21 là điểm yếu của cVAE** (R² 0,838 vs 0,997).
- **Chế tạo được:** SIMP 77-84% vs cVAE ~25% → SIMP thắng xa; ảnh hưởng trực tiếp trục đa mục
  tiêu của bài.
- **SIMP cũng có khoảng lệch xám ↔ nhị phân:** trung vị sai số trên x̂ (β=64) 2e-7 nhưng sau
  ngưỡng 0,5 là 3,6e-2 (mỗi seed); seed hexagonal thường sụp. Củng cố luận điểm chính của bài
  (vấn đề chung, không riêng cVAE) - chọn theo verify (best-of-seeds) mới cứu được.
- Thời gian CPU SIMP (đo khi máy chạy song song job khác, cận trên): 54 s/target (`matched`
  best-of-4), 138 s (`full`); 1 FE-solve 50×50 + sensitivity = 0,086 s.
- **Đính chính 2026-10-06 (phát hiện khi vẽ `fig_stiffness`):** "hòa" ở cặp (ν12,ν21) và R²(ν21) 0,838 của cVAE do **1 condition duy nhất** (#46, target (−0,036; −1,25) - ν21 thấp nhất IN100, đòi E_x/E_y ≈ 0,03): cVAE sinh ô **đứt** (1 cột rỗng, E_x/E0 ≈ 2e-11, ν12 = 0 suy biến; manufacturability đã bắt). Bỏ #46: cVAE R²(ν21) 0,997 (SIMP 0,996), sai số cặp **−29% [9; 46] so với SIMP `full`** (−33% [13; 47] vs `matched`). Báo cáo đề xuất: đủ 100 + coi E < 1e-3·E0 là thất bại (cVAE 1/100, SIMP 0/100) + phân tích độ nhạy 99 - **không** loại #46 khỏi kết quả chính.
- **Hệ quả cho bài:** không claim "cVAE chính xác hơn SIMP" chung chung; claim được: (i) ν12 chính
  xác hơn với ít FE hơn 3-8×; (ii) cùng ngân sách FE thắng SIMP 1 khởi tạo. Lập luận khấu hao
  (amortization) yếu: chi phí sinh dataset ~10^4 lần chạy SIMP chỉ hòa vốn sau rất nhiều target -
  cần đo chi phí sinh dataset thật trước khi viết con số.

### P1.7 Khung claim + thí nghiệm lai cVAE → SIMP (tiêu chí ghi 2026-10-06, TRƯỚC khi viết code/chạy)

**Khung claim (đề xuất, mentor đồng ý chạy thử B):**
- **Khung A - xương sống (đã đủ bằng chứng):** "tối ưu đúng thứ được kiểm chứng" - tối ưu dựa trên
  gradient (refine latent của mô hình sinh LẪN SIMP) khai thác khoảng lệch objective↔verify; đưa
  chính các toán tử verify vào objective đóng khoảng lệch. Bằng chứng: P1.1e, IN100-B, P1.6a (SIMP
  β=64 lệch 2e-7 → 3,6e-2). Tạp chí: SMO.
- **Khung B - nâng hạng nếu đạt:** cVAE + refine làm điểm khởi đầu cho SIMP ("lai") đạt độ chính
  xác + khả năng chế tạo của SIMP hội tụ với ít FE hơn nhiều. Lý do kỳ vọng: 2 phương pháp bù đúng
  điểm yếu của nhau (P1.6a: cVAE mạnh ν12/chi phí, SIMP mạnh chế tạo/OOD/#46).

**Thiết kế thí nghiệm lai (1 cấu hình chính, không tinh chỉnh sau khi xem kết quả):**
- Điểm khởi tạo SIMP = thiết kế cuối cVAE best-of-30 + refine guarded (ảnh nhị phân trong
  `p1_6c_in100_bo30_images.json`, qua đúng đường verify force_periodic → ngưỡng → resize 50²),
  clip [X_MIN, 1], co về volfrac_max = 0,55 nếu vượt (cùng quy tắc `initial_design`).
- SIMP = `benchmark_simp_baseline.py` (objective, ràng buộc, verify giữ nguyên), nhưng β bắt đầu
  sắc để giữ tô-pô khởi tạo: **β ∈ {8, 16, 32, 64}, tối đa 15 eval/mức β (≤ 61 FE gồm verify)**.
  Tổng chi phí/mục tiêu = 107 (cVAE) + ≤ 61 = **≤ 168 FE**.
- Chọn kết quả cuối: **guarded** - giữ thiết kế lai chỉ khi sai số cặp verify nhỏ hơn thiết kế
  cVAE khởi tạo (không tốn thêm FE, giống guarded của refine).
- So sánh ghép cặp với SIMP best-of-4 `full` (897 FE) và cVAE guarded (107 FE), cùng 100 mục tiêu
  IN100, **giữ cả #46**.

**Tiêu chí chính (IN100, đủ 100 mục tiêu) - Khung B ĐẠT chỉ khi đạt CẢ 4:**
1. **Sai số cặp |Δν12|+|Δν21| không kém SIMP `full`:** cận dưới CI95 (bootstrap ghép cặp,
   10 000 lần) của mức giảm tương đối so với SIMP `full` ≥ **−10%** (biên không-kém-hơn 10%).
2. **Tỉ lệ chế tạo được** (`check_manufacturability(...)["passes_all"]` trên ảnh nhị phân cuối)
   ≥ **0,70**.
3. **Chi phí:** FE trung bình/mục tiêu ≤ **299** (≤ 1/3 của 897).
4. **Không suy biến:** tỉ lệ thiết kế có E_x hoặc E_y < 1e-3·E0 (lưới 50²) ≤ **1%**.

**Xác nhận bắt buộc nếu đạt:** chạy lại y hệt trên IN100-B (seed 456, cần chạy trước cVAE
best-of-30 + refine `--save-images` cho IN100-B); claim Khung B chỉ khi đạt cả 4 tiêu chí trên
CẢ IN100 và IN100-B.

**Thăm dò (không đặt ngưỡng, báo số thật):** 28 mục tiêu OOD đổi dấu (P1.6b) - MAE ν12, sai số
cặp, chế tạo được so với SIMP `full`; ν trên lưới 200² của thiết kế lai.

**Quy tắc quyết định:**
- Đạt cả 4 trên IN100 và IN100-B → Khung B thành kết quả chính, Khung A thành cơ chế giải thích.
- Không đạt → viết bài theo Khung A; thí nghiệm lai báo cáo trung thực như kết quả âm/một phần
  (nêu tiêu chí nào trượt). Không đổi β, ngân sách hay ngưỡng sau khi có số.

**ETA:** code + test ~2 giờ; chạy IN100 ~15 phút (CPU, 10 worker); nếu đạt: IN100-B ~1,5 giờ
(cVAE best-of-30 + refine trên GPU, rồi lai). Tổng ~0,5 ngày.

**Kết quả P1.7 (2026-10-07, IN100, `pipeline/phase5_cvae/benchmark_hybrid.py` →
`outputs/phase5/plan_v3/p1_7_hybrid_in100.json`; cấu hình đúng như ghi trước, không chỉnh sau):
Khung B KHÔNG ĐẠT (trượt 2/4) → bài viết theo Khung A; không chạy xác nhận IN100-B.**

| Tiêu chí | Kết quả | Ngưỡng | |
|---|---|---|---|
| 1. Sai số cặp vs SIMP `full` | 0,0119 vs 0,0128 (+7,3%, CI95 [−96; +62]) | cận dưới ≥ −10% | ❌ |
| 2. Chế tạo được | 0,33 | ≥ 0,70 | ❌ |
| 3. FE/target | 166 | ≤ 299 | ✅ |
| 4. Suy biến (E < 1e-3·E0) | 1% (#46) | ≤ 1% | ✅ |

Chẩn đoán (chỉ để báo cáo, KHÔNG đổi kết luận): (i) tiêu chí 1 trượt hoàn toàn do #46 - ô cVAE đứt
(E_x≈0) nên SIMP lai không phục hồi, guarded giữ thiết kế cVAE; bỏ #46 lai thắng SIMP `full` 51%
[30; 65], trung vị sai số cặp 0,0033 vs 0,0102. (ii) Tiêu chí 2 trượt thật: chế tạo 0,24 (cVAE) →
0,39 (lai thô) → 0,33 (guarded) - β khởi đầu sắc giữ luôn các nét không chế tạo được của cVAE.
Báo cáo trong bài như kết quả một phần: lai cải thiện độ chính xác cặp với 166 FE (5,4× ít hơn
SIMP `full`) nhưng không thừa hưởng khả năng chế tạo của SIMP.

**Tái lập P1.6a trên IN100-B (2026-10-07; cVAE `p1_7_in100b_bo30_images.json`, SIMP best-of-4
`full` `p1_7_simp_in100b_full.json`, 889 FE/target):**
- cVAE guarded R²(ν12) 0,994, MAE ν12 0,0061 vs SIMP 0,996 / 0,0073 → **ν12: +16% [−21; 43],
  KHÔNG có ý nghĩa** trên đủ 100 (IN100: 34% [13; 50]). Cặp: cVAE 0,0233 vs SIMP 0,0139 (−67%
  [−224; 41], hòa). ν21 cVAE yếu hơn nữa (R² 0,758 vs 0,997).
- Nguyên nhân: **2/100 thiết kế cVAE đứt** (#1, #57, E_x ≈ 2e-11·E0) - cùng kiểu lỗi #46 ở
  IN100. Bỏ 2 ô này: ν12 +31% [6; 50] (tái lập IN100), cặp +28,5% [−2; 51]; trung vị sai số cặp
  0,0055 vs 0,0096.
- → **[ĐÃ THAY bởi P1.8, 2026-10-07: không claim ν12 hơn SIMP]** ~~Claim an toàn cho bài:~~ "trên các thiết kế không suy biến, ν12 cVAE chính xác hơn SIMP hội
  tụ ~30% với ít FE hơn ~8×; cVAE sinh ô đứt ở 1-2% mục tiêu, khi đó SIMP thắng." KHÔNG claim
  thắng trên đủ 100 mục tiêu.
- **Ứng viên sửa (CHƯA claim được - quyết định tác giả):** loại ứng viên có E_x hoặc E_y < 1e-3·E0
  ngay lúc chọn best-of-30 (Q đã có từ lần verify, không tốn FE). Là thay đổi phương pháp sau khi
  xem số → nếu dùng phải ghi tiêu chí trước và đánh giá trên tập mới (vd seed 789).
- Chế tạo được, cùng thước đo liên thông + nét tối thiểu (bỏ kiểm tra cạnh, xem C5): SIMP `full`
  0,77 (IN100) / 0,74 (IN100-B) vs cVAE 0,24 (fp) / 0,32 (không fp).

**Chẩn đoán hậu kiểm (2026-10-07, KHÔNG phải claim):** mọi ca cVAE thất bại (#46 IN100; #1, #57,
#70 IN100-B) là mục tiêu dị hướng cực đoan, tỉ số r = max(ν21/ν12, ν12/ν21) ≥ 10 (≈ E_y/E_x với
vật liệu trực hướng); trong 30 ứng viên best-of-30 của #46/#1/#57 **cả 30 đều đứt** → lọc ô suy
biến (C6) không cứu được, đây là giới hạn phủ dữ liệu (train: 2,8% mẫu có r>10, 1,2% có r>20).
Trên r < 10 (99/97 mục tiêu), cVAE vs SIMP `full`: ν12 +42% [23; 56] / +38% [18; 53], cặp +27%
[2; 46] / +40% [22; 54] (IN100 / IN100-B).

**P1.8 - Xác nhận phân tầng dị hướng trên tập mới IN100-C (tiêu chí ghi 2026-10-07 ~13:00,
TRƯỚC khi chạy):** seed 789, 100 condition test.npz (cùng cách chọn), cVAE best-of-30 + refine
guarded (y hệt P1.6c, có fp - giữ nguyên pipeline đã báo cáo), SIMP best-of-4 `full` (y hệt P1.6a).
Phân tầng theo r của **mục tiêu** (biết trước khi giải, không dùng kết quả): r < 10 vs r ≥ 10.
- **Tiêu chí chính (đạt khi CẢ 2):** trên r < 10, mức giảm tương đối sai số cặp cVAE vs SIMP
  `full` có cận dưới CI95 > 0 VÀ mức giảm MAE ν12 có cận dưới CI95 > 0 (bootstrap ghép cặp 10 000).
- **Báo thật, không ngưỡng:** số mục tiêu r ≥ 10 và sai số từng mục tiêu; số ô suy biến.
- Đạt → claim được "cVAE + refine chính xác hơn SIMP hội tụ với ~8× ít FE hơn trên mục tiêu dị
  hướng vừa phải (r < 10, ~97% phân phối test), thất bại ở dị hướng cực đoan do thiếu dữ liệu".
  Không đạt → chỉ báo như chẩn đoán hậu kiểm.

**Kết quả P1.8 (2026-10-07 13:17; `docs/paper1/scripts/p1_8_anisotropy_strata.py` →
`p1_8_anisotropy_strata.json`; cVAE `p1_8_in100c_bo30_images.json`, SIMP
`p1_8_simp_in100c_full.json` 893 FE/target): KHÔNG ĐẠT (1/2).**

| Tầng r < 10 (n=99) | cVAE vs SIMP `full` | |
|---|---|---|
| Sai số cặp | **+26,8% [7,9; 42,4]** | ✅ |
| MAE ν12 | +18,7% [−9,6; 39,7] | ❌ CI chứa 0 |

- Toàn bộ IN100-C: cVAE R²(ν12) 0,998, R²(ν21) 0,998, MAE ν12 0,0052 vs SIMP 0,998 / 0,996 /
  0,0064; mục tiêu r ≥ 10 duy nhất cVAE làm tốt (sai số cặp 0,020 vs 0,058); không có ô suy biến.
- Chênh lệch ν12 giữa 2 phương pháp (~0,001) nhỏ hơn ~10× sai số rời rạc hóa lưới 50² (0,015,
  P1.6x) → so "ν12 chính xác hơn SIMP" không có ý nghĩa vật lý dù có ý nghĩa thống kê ở IN100.
- **Claim được phép (sau P1.6a + IN100-B + P1.8):** cVAE best-of-30 + refine **đạt độ chính xác
  ngang SIMP hội tụ đa khởi tạo** (MAE ν12 ~0,005, R² ~0,998 so với mô hình FE 50²) **với ~8× ít
  FE hơn**; trên mục tiêu dị hướng vừa phải (r < 10) sai số cặp thấp hơn SIMP ~27% (xác nhận
  trên tập ghi trước IN100-C, nhất quán với IN100/IN100-B); cVAE có thể sinh ô đứt ở mục tiêu
  dị hướng cực đoan (3/4 mục tiêu r ≥ 10 qua 3 tập). **KHÔNG claim** ν12 chính xác hơn SIMP.

**C5 - ablation `force_periodic` (2026-10-07, IN100, cùng z, chỉ khác cờ;
`docs/paper1/scripts/c5_force_periodic.py` → `p1_7_c5_comparison.json`):**

| | R²(ν12) guarded | MAE cặp guarded | Liên thông + nét tối thiểu | Kiểm tra cạnh khớp | passes_all |
|---|---:|---:|---:|---:|---:|
| có fp (hiện tại) | 0,998 | 0,0148 | 0,24 | 1,00 | 0,24 |
| không fp | 0,998 | 0,0139 | **0,32** | 0,30 | 0,09 |

- Độ chính xác **không đổi** (cặp, không fp vs fp: +6% [−11; 34], CI chứa 0).
- Kiểm tra "cạnh đối diện khớp" chỉ đạt 100% vì fp ép nó; bỏ fp còn 30%. Với lưới FE dựa trên
  phần tử + PBC, mọi ảnh pixel đều lát được bằng tịnh tiến → kiểm tra này không đo khả năng lát
  thật; fp lại làm giảm liên thông + nét tối thiểu (0,32 → 0,24).
- **Đề xuất cho C5 (quyết định tác giả):** bỏ fp và bỏ kiểm tra cạnh khỏi định nghĩa chế tạo được
  trong bài; báo chế tạo được = liên thông + nét tối thiểu. Hệ quả: các số 24,7% / 30% → 24% của
  bài phải đo lại trên pipeline không fp (refine P1.1e chạy lại không fp ~45 phút GPU nếu cần số
  single-shot). Nếu giữ fp: phải sửa câu "opposite edges must match so that the cell tiles" và nói
  rõ fp chỉ là quy ước biểu diễn.

---

## 4. Bảng trạng thái + ETA tổng (cập nhật trực tiếp tại đây)

| # | Task | Phụ thuộc | Trạng thái | ETA |
|---|---|---|---|---|
| P1.0 | Harness benchmark | – | ✅ XONG 2026-09-25 (+ sửa bug loader, mục 1.4; 660/660 test) | – |
| P1.1 | Physics-guided refinement | P1.0 | ✅ XONG 2026-09-25: e sửa được b; d chỉ đạt ở guarded (xem mục P1.1) | – |
| P1.2 | Ablation KAN vs Linear 2×2 | P1.0 | ✅ XONG 2026-09-25: KAN không thắng, giữ Linear | – |
| P1.3 | WIRE lần cuối | P1.0 | ❌ ĐÓNG 2026-09-25: không đạt tiêu chí dừng (12,5% / R² −7,19) | – |
| P1.4 | ConvKAN (tùy chọn) | – | ⬜ Tùy chọn | ~0,5 ngày |
| P1.5 | Bản thảo Bài #1 | P1.1-P1.3 | ✅ Bản đầu XONG 2026-09-30 (`docs/paper1/`, EN + VI) | – |
| P1.6a | Baseline SIMP từ đầu | P1.5 | ✅ XONG 2026-10-06 (IN100 đủ 100 target, xem mục P1.6a bên dưới): ν12 cVAE thắng, cặp (ν12,ν21) hòa SIMP đa khởi tạo, chế tạo SIMP thắng xa. **Đính chính: "hòa" do 1 condition #46 (ô đứt); bỏ #46 cVAE thắng cặp 29% [9; 46]** | – |
| P1.6b | OOD đổi dấu chạy lại (checkpoint production, 20-30 mục tiêu) | P1.5 | ✅ XONG 2026-10-06: 28 mục tiêu ν>0 (`p1_6b_signflip_*.json`, SIMP `p1_6b_simp_*.json`). cVAE+refine guarded MAE(ν12) 0,038/0,104/0,089 (đối xứng / ν21=0,1 / ν21=0,3; refine giảm MAE 59-76%), nhưng **SIMP từ đầu thắng rõ** (0,005-0,040, 300-830 FE) → lợi thế OOD chỉ claim được so với retrieval, không so với SIMP | – |
| P1.6c | Manufacturability sau refine + hình trước/sau | P1.5 | ✅ XONG 2026-10-06: chế tạo được 30% → 24% sau refine (**đính chính 2026-10-07: không vững - 5 lần chạy −6 đến +9 điểm %, gộp 61 thêm / 44 mất, p=0,12; refine không làm giảm chế tạo có hệ thống**) (`p1_6c_in100_bo30_images.json`, tái lập P1.1e-b R² 0,9984). 4 hình mới trong `make_figures.py` (EN+VI, đã kiểm bằng mắt): `fig_refine_examples` (z tái tạo khớp JSON tới 1e-6, `refine_examples.py`), `fig_designs` (cVAE vs SIMP), `fig_tiling` (3×3), `fig_stiffness`, `fig_deformation` (kéo đơn trục, `deformation_examples.py`: ν12 tính lại khớp JSON chính xác), `fig_ood` thêm panel mật độ dữ liệu train | – |
| P1.6x | Kiểm tra bổ sung (review 2026-10-05) | – | ✅ 2026-10-06: **IN100-B** (seed 456, 4/100 trùng IN100) single-shot MAE −87,8% [82; 92], bo30 −65,8% [56; 75] → refine tái lập trên tập mới. **Hội tụ lưới**: ν 50² vs 200² lệch trung vị 0,015 (`p1_6x_mesh_convergence.json`). **Thiết kế cuối trên 200²** (`p1_6x_final_design_eval.json`): R²(ν12) cVAE 0,992 / SIMP full 0,985, MAE 0,0125 / 0,0162 → xếp hạng giữ nguyên; E_x/E0 trung vị ~0,07-0,08 cả 2 (không phải cơ cấu mềm) | – |
| P1.6y | Retrieval verified trong phân phối (review mục 5) + lệch nhãn↔verify | – | ✅ 2026-10-06 (`docs/paper1/scripts/retrieval_in100.py` → `p1_6x_retrieval_in100{,_nofp}.json`): retrieval-1 R²(ν12) 0,941 (không fp) / 0,881 (fp); retrieval best-of-30 0,977 / 0,972, MAE cặp 0,035 / 0,041 - so cVAE bo30 0,986 (cặp 0,048), cVAE+refine 0,998 (cặp 0,016). **`LIMITATIONS.md` #18 "retrieval R²=1,000" đo trên NHÃN, không verify - cần đính chính.** Phân tách lệch nhãn dataset↔verify (150 mẫu test, MAE ν12): khứ hồi 64→50 0,019 · +penal 3 0,024 · +nhị phân 0,032 · +force_periodic 0,055. ⚠️ **Cần quyết định:** `force_periodic` ép cột đầu = cột cuối, nhưng lưới FE dựa trên phần tử + PBC nên mọi ảnh pixel vốn đã lát được; câu "opposite edges must match so that the cell tiles" (bài, dòng ~305-312) có thể sai khái niệm | – |
| P1.6z | Chi phí sinh dataset + điểm hòa vốn (review A5) | – | ✅ 2026-10-06 (ước tính, có khoảng): multi-batch 1-8, 11 = 10 080 lần chạy SIMP, 1,38M vòng lặp (đếm thật từ `outputs/multi_batch/batch_*_results.csv`); phase 1 = 550 lần, 67k vòng; batch 0 (8 409 mẫu, 62% dataset, thư mục raw `phase3_v2_raw` đã xóa) ước ~8 900-10 700 lần × ~145 vòng (EXPERIMENT_LOG: "~10 700 mẫu/~3,5h"). **Tổng ≈ 2,7-3,0M FE-solve (~65-72 CPU-giờ, chưa tính train GPU).** Hòa vốn so SIMP best-of-4 `full` (tiết kiệm 790 FE/target): **~3 500-3 800 target**; so `matched` (237 FE/target): ~11 500-12 700 target → **không claim lợi ích chi phí cho 1 target; chỉ claim khi cần thiết kế hàng nghìn target** | – |
| P1.7 | Thí nghiệm lai cVAE → SIMP (tiêu chí đặt trước, mục P1.7) | P1.6a-c | ❌ KHÔNG ĐẠT 2026-10-07 (trượt tiêu chí 1 do #46 và tiêu chí 2: chế tạo 0,33) → Khung A; xem mục P1.7 | – |
| P1.8 | Xác nhận phân tầng dị hướng trên IN100-C (seed 789, tiêu chí ghi trước) | P1.7 | 🟡 ĐẠT 1/2 2026-10-07: sai số cặp r<10 +26,8% [7,9; 42,4] ✅, ν12 +18,7% [−9,6; 39,7] ❌ → claim "ngang SIMP với 8× ít FE", không claim ν12 hơn; xem mục P1.7 | – |
| P1.6d-f | GS review, công bố mã, thực nghiệm (dài hạn) | P1.6a-c | ⬜ Chưa làm | xem mục P1.6 |
| S1 | Tính chất rẻ suy từ Q (ngoài đường găng Bài #1, xem mục 5.1) | – | ✅ XONG 2026-09-30: backfill A4 + notebook 09; 693/693 test | – |

**Đường găng (cập nhật 2026-10-07):** ~~P1.0 → P1.6c, P1.7, P1.8~~ (xong) → **chốt C1-C5 (tác giả)** →
P1.9 đo lại số chế tạo không fp (nếu C5 = bỏ) → P1.10 viết lại bản thảo EN+VI → P1.6d GS review →
P1.6e công bố mã → P1.11 chuẩn bị nộp → nộp.
**Tổng ETA còn lại:** ~3-3,5 ngày làm việc sau khi tác giả chốt C1-C5 (không tính thời gian chờ GS).

| # | Việc còn lại | Ai | Phụ thuộc | ETA |
|---|---|---|---|---|
| D | Chốt C1, C1b, C2, C3, C4, C5 (bên dưới) | Tác giả | – | ✅ CHỐT 2026-10-07 |
| P1.9 | Chạy lại mọi số của bài không fp (C5) | Claude | C5 | ✅ XONG 2026-10-07 14:43 - xem mục "Kết quả P1.9" ngay dưới bảng | – |
| P1.10 | Viết lại bản thảo EN + VI theo Khung A: bảng SIMP (P1.6a/IN100-B/IN100-C), cột lưới 200², phân tầng dị hướng, P1.7 kết quả âm, retrieval verified, ν21 đầy đủ, 6 hình mới, sửa mục 10-18 review 2026-10-05; tên bài mới | Claude | D, P1.9 | 🟡 2026-10-08: viết lại xong `main_en.tex` + `main_vi.tex` (cấu trúc SMO, số không fp, mục review 10-18; bản cũ ở `docs/paper1/archive/`). **Phát hiện mới:** SIMP trả về ô rỗng (ν=ν0=0,3) ở 3/28 mục tiêu đổi dấu - áp tiêu chí suy biến E<1e-3·E0 cho cả 2 phương pháp (`p1_10_degenerate_check.json`), SIMP vẫn thắng OOD. **Còn (cần cắm sạc, ~30 phút máy):** chạy `final_design_eval.py` không fp → cập nhật số 200² + `fig_stiffness`; `make_figures.py` (đã trỏ file không fp, chưa chạy); `tectonic` 2 bản; kiểm hình bằng mắt; DOI TOuNN |
| P1.6d | Gửi GS Nguyễn Đình Đức: đồng tác giả, tạp chí, góc composite/FGM | Tác giả | P1.10 | – |
| P1.6e | Công bố mã + dữ liệu: dọn repo, README tái lập, gói Zenodo (tác giả tạo DOI) | Claude chuẩn bị, tác giả đăng | P1.10 | ~0,5 ngày |
| P1.11 | Chuẩn bị nộp: định dạng tạp chí, cover letter, CRediT, khai báo dùng AI, data availability | Cả hai | P1.6d, P1.6e | ~0,5 ngày |
| P1.6f | (Dài hạn, tùy chọn) In 3D + đo ν thật / FE thương mại / biến dạng lớn | Tác giả | – | tuần |

**Kết quả P1.9 - pipeline không fp (2026-10-07; `docs/paper1/scripts/p1_9_nofp_summary.py` →
`outputs/phase5/plan_v3/p1_9_nofp_summary.json`, đã kiểm lại độc lập 2 phát hiện chính):**

| Số trong bài | Có fp (cũ) | Không fp (mới) |
|---|---|---|
| Bảng 1 single-shot + refine nhận thức nhị phân hóa: R²(ν12), ΔMAE | 0,998, −91,8% | 0,998 (guarded), −92,5% [90; 94,5] (vốn không fp) |
| Bảng 1 best-of-30 + refine liên tục: ΔMAE | −62,5% (tệ hơn) | −73,8% [−119; −39] (tệ hơn) |
| Bảng 1 best-of-30 + refine nhận thức: R²(ν12), ΔMAE | 0,997, +62,6% | 0,998, +68,3% [59; 76] |
| Chế tạo được (liên thông + nét tối thiểu), best-of-30 guarded | 0,24 | 0,32 (single-shot 0,22 → 0,29) |
| OOD biên độ, refine nhận thức, sai số tương đối trung vị tại −2,0 / −2,25 / −2,5 | 12,5% / 22% / 32% | **1,2%** / 24,7% / 28,8% (guarded 1,2 / 14,2 / 20,0) |
| SIMP, tầng r < 10, sai số cặp (IN100 / B / C) | +27 / +40 / +27% | +41 [26; 54] / +40 [23; 53] / +35 [19; 48]% |
| SIMP, tầng r < 10, ν12 (IN100 / B / C) | +42 / +38 / +19% (C: CI chứa 0) | +39 [19; 54] / +39 [19; 54] / **+37 [14; 54]%** |
| OOD đổi dấu MAE ν12 cVAE vs SIMP (đối xứng / ν21=0,1 / 0,3) | 0,038/0,104/0,089 vs 0,012/0,031/0,005 | 0,056/0,090/0,076 vs cùng SIMP → SIMP vẫn thắng |
| Bảng 2 n=300 R²: chọn độ chính xác / composite; chế tạo TB pool | 0,995 / 0,983; 0,247 | 0,9965 / 0,988; 0,238 |
| Spearman composite ↔ Pareto (TB, trung vị) | 0,693, 0,714 | 0,718, 0,746 |

**Cảnh báo diễn giải (bắt buộc ghi trong bài):**
- C5 được chốt vì lý do khái niệm + ablation C5 (độ chính xác IN100 không đổi), TRƯỚC khi có các số
  trên. Dù vậy nhiều số tốt lên sau khi bỏ fp → báo cả 2 phiên bản ở phụ lục để reviewer thấy.
- **P1.8 giữ kết quả đăng ký trước: ĐẠT 1/2 (pipeline có fp).** Bản không fp đạt cả 2 tiêu chí
  trên IN100-C nhưng là phân tích độ nhạy, không thay kết quả đăng ký trước.
- Tiêu chí P1.1d (≤ 8% tại −2,0) giờ đạt cả bản thường, nhưng phân phối 2 cụm: 6/10 lần ≤ 1,7%, 4/10
  lần 17-31% → báo IQR/từng lần, không chỉ trung vị; −2,25 trở đi vẫn bão hòa (không claim ngoại suy).
- Ngưỡng Spearman 0,7 đặt cho bản có fp vẫn là TRƯỢT; bản không fp 0,718 chỉ báo kèm.

**Quyết định của tác giả (CHỐT 2026-10-07):**
- **C1 = Khung A**: đóng góp chính "objective phải tối ưu đúng thiết kế được verify" (cVAE lẫn SIMP);
  claim cVAE + refine ngang SIMP hội tụ với ~8× ít FE, sai số cặp thấp hơn ~27% khi r < 10; nói rõ
  SIMP thắng chế tạo, OOD, dị hướng cực đoan; không claim ν₁₂ hơn SIMP. **C1b** = báo đủ mẫu +
  phân tầng theo r (mục tiêu r ≥ 10 báo từng ca).
- **C2** = "Optimizing What Is Verified: Binarization-Aware Latent Refinement for Generative Inverse
  Design of Auxetic Metamaterials".
- **C3** = không đưa ν0 (để Bài #2). **C4** = Structural and Multidisciplinary Optimization (SMO).
- **C5** = bỏ `force_periodic` + bỏ kiểm tra cạnh khỏi định nghĩa chế tạo được (= liên thông + nét
  tối thiểu); mọi số trong bài chuyển sang pipeline không fp. Kết quả P1.8 (đăng ký trước trên
  pipeline có fp) vẫn báo như đã ghi; bản không fp chỉ là phân tích độ nhạy.
- **C6 (mới 2026-10-07) - ĐÓNG:** lọc ô suy biến lúc chọn best-of-30 không giúp được - ở #46/#1/#57
  cả 30/30 ứng viên đều đứt (giới hạn phủ dữ liệu dị hướng cực đoan, không phải lỗi chọn).
- Còn lại sau khi chốt: xem bảng "Việc còn lại" ngay trên (P1.9 → P1.10 → P1.6d/e → P1.11).

---

## 5. Bài báo #2 — đóng băng tới khi Bài #1 có bản thảo

Code khung đã có (`pipeline/mno/`, `pipeline/kinn/`, `pipeline/ickans/`) — giữ nguyên, không đầu
tư thêm. Điều kiện tiên quyết cần có trước khi mở lại:

| Task | Điều kiện tiên quyết còn thiếu | KPI v2 (giữ làm mục tiêu tham khảo) |
|---|---|---|
| MNO (Mamba neural operator) | `mamba_ssm` trong env (hiện fallback GRU); **dataset trường dịch chuyển FE** (chưa có) | ≤ 1,5 ms/mẫu; L2 ≤ 1% |
| KINN phi tuyến | JAX-AMG; **solver FE phi tuyến tham chiếu** (chưa có — `simp/core/` là tuyến tính) | Hội tụ < 30 s/mẫu |
| ICKAN composite | GS hướng dẫn duyệt hướng composite/FGM (trùng hướng "composite auxetic" trong `PROJECT_PLAN.md`) | R² ≥ 0,985 OOD; Hessian xác định dương |

Trước khi mở lại: trình GS chọn **1** trong 3 hướng (không làm song song cả 3).

### 5.1 Ứng viên hướng đa tính chất (S1, 2026-09-30)

Đã làm tầng "rẻ": các tính chất là hàm đại số của Q, không cần solver mới (chi tiết:
`EXPERIMENT_LOG.md` mục 2026-09-30, `notebooks/09_cheap_physical_properties.ipynb`).
- **E_x, E_y, B_eff, M_x/M_y (= f1/f2), tốc độ sóng dọc:** 92–97% dự đoán được từ 5 điều kiện
  hiện có, nên thêm làm condition gần như dư thừa.
- **G_xy (R² 0,84):** mang nhiều thông tin mới nhất. Gradient đã có sẵn trong
  `solve_elastic_with_grad`.
- Đánh đổi ν12 ↔ E_x/ρ rất mạnh (~5,7× dọc biên Pareto), nên target (ν, E) phải nằm dưới biên.

Bước tiếp theo nếu mở hướng này (chưa lên lịch, không nằm trên đường găng Bài #1):

| Bước | Việc | ETA |
|---|---|---|
| S2 | Nối `solve_elastic_with_grad` vào refinement nhận thức nhị phân hóa, target (ν, G_xy) | ~1-2 ngày |
| S3 | Solver dẫn nhiệt κ + adjoint (Laplace vô hướng, dùng lại PBC); đo tương quan với volfrac trước khi dùng | ~3-5 ngày |
| – | Permeability, phononic/photonic band gap: đề tài riêng. Optoelectronics: không áp dụng. CTE 1 vật liệu: không điều chỉnh được bằng hình học | – |

---

## 6. Rủi ro

- **P1.1a không đạt trên production** → luận điểm chính yếu đi; đã có điểm dừng quyết định ngay
  sau P1.1.
- **OOD ν*=−2,5 có thể không khả thi vật lý** ở lưới 50×50 → bước P1.1c kiểm trước, tránh đo một
  mục tiêu không tồn tại lời giải.
- **P1.2 hòa** là kết quả có khả năng cao (theo số đo 1.1) — chấp nhận và báo cáo trung thực;
  bài báo không phụ thuộc vào việc KAN thắng.
- **Real-physics chậm** (FE trong vòng train) → ETA P1.2 có thể trượt 30-50%; chạy nền qua đêm.
- **Sàn cứng trong `CLAUDE.md`** tham chiếu `cvae_v2_finetuned.pt` - đúng checkpoint production
  (đính chính 1.1), nhưng nên ghi rõ sàn đo bằng **R²(FE) best-of-N** chứ không phải
  `property_accuracy()` (trên thước đo surrogate checkpoint này chỉ 0,27/0,46, sàn vô nghĩa).

---

## Phụ lục A — Chi tiết kỹ thuật P1.3/P1.4 (giữ từ v2, 2026-08-24, CHƯA CHẠY)

### Phần A — ConvKAN tăng `grid_size` (P1.4)

`SurrogateCNN` hiện hard-code `grid_size=5` (`pipeline/phase4_surrogate/model.py`). Cần: tham số
`kan_grid_size` trong `__init__`, flag `--kan-grid-size` ở `train.py` + lưu vào checkpoint, đọc
`ckpt.get("kan_grid_size", 5)` ở `evaluate.py`/`export_for_phase5.py`/`losses.py::load_frozen_surrogate`/
`bootstrap_ci.py` (tương thích ngược), test. Cấu hình run: `--epochs 60 --batch-size 128 --lr 5e-4
--patience 10 --seed 0 --use-kan --kan-grid-size 8`.

### Phần B — WIRE + Heaviside + real-physics (P1.3)

- Port `simp/core/filter.py::apply_heaviside_projection` sang torch khả vi
  (`pipeline/phase5_cvae/heaviside.py`, tanh, η=0,5, β ramp 1→50), test so khớp bản numpy.
- Flag `heaviside_beta` trên `CVAE`, áp trong `forward()`/`generate()`, lưu vào checkpoint; ramp β
  theo epoch sau KL warmup; real-physics tính trên ảnh **đã chiếu**.
- Recipe 2 giai đoạn (bài học từ KAN): base WIRE (có `enforce_symmetry`) → fine-tune
  `--decoder-type wire --select-by fe_r2 --fe-eval-every 5 --lambda-real-physics 2.0
  --real-physics-every 10 --real-physics-subsample 8 --heaviside-beta-max 50`; `--resume-from`
  phải dùng `--output-name` mới (`train.py` reset `best_val` mỗi lần chạy — `EXPERIMENT_LOG.md`
  2026-09-23).
- Eval `best_of_n_eval.py` ở resolution 128 (WIRE không phụ thuộc resolution).

## Phụ lục B — Lịch sử số liệu v2 (để đối chiếu)

| Checkpoint | Metric | Giá trị | Nguồn |
|---|---|---|---|
| `cvae_kan_realphysics_v2` | R²(FE) best lúc train | 0,889 | `EXPERIMENT_LOG.md` 2026-08-23 |
| `cvae_kan_realphysics_v2` | R² FE(v12), n=24 | 0,581 | Exp 3, 2026-09-11 |
| `exp3_mlp_physics_150` | R² FE(v12), n=24 | −9,834 | Exp 3 (recipe hỏng, xem 1.1) |
| `cvae_wire_v2` / `_v3_round2` | hit-rate single-shot | 4,17% / 0,00% | 2026-08-24 / 2026-09-23 |
| `surrogate_convkan_v2` | R² v12/v21/volfrac | 0,964/0,957/0,916 | 2026-08-24 |
| Refinement (KAN v2, `IN24`, 30 bước) | MAE(FE) v12 | 0,587 → 0,355 | 2026-09-23 |
