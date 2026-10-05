# CLI Guide - tổng hợp lệnh dòng lệnh của dự án

Tài liệu tra cứu nhanh, gom mọi lệnh CLI đang dùng thật trong repo (không lặp lại phần giải thích
lý do/thiết kế - xem [`PIPELINE.md`](PIPELINE.md) cho narrative đầy đủ từng phase, tài liệu này chỉ
liệt kê lệnh + flag quan trọng để copy-paste). Mọi lệnh Python đều giả định chạy từ thư mục gốc repo.

## 0. Kích hoạt môi trường (BẮT BUỘC trước mọi lệnh Python)

`python` mặc định của hệ thống (miniconda `base`) **không có torch** - phải activate env `simp`:

```bash
source /home/tbm/miniconda3/etc/profile.d/conda.sh && conda activate simp
```

## 1. Test suite

```bash
pytest tests/ -q
```

Chạy 1 file / 1 class / 1 test cụ thể lúc debug:

```bash
pytest tests/test_phase5_real_physics.py -q -k TestPerSampleMaterial
```

Quy ước project (CLAUDE.md): mọi thay đổi ảnh hưởng kết quả khoa học (solver SIMP, homogenization,
surrogate, cVAE) phải chạy `pytest tests/ -q` xanh trước khi báo hoàn thành.

## 2. `simp` - chạy 1 lần tối ưu SIMP (entry point `pyproject.toml:48` → `simp/main.py`)

```bash
simp --seed hexagonal --nelx 64 --nely 64 --volfrac 0.4 --nu 0.3 --max_iter 50 --output_dir /tmp/simp_test
```

```bash
simp --help          # toàn bộ flag (nelx/nely/volfrac/penal/rmin/nu/E0/objective_variant/stiffness_mode...)
simp --list-seeds     # liệt kê seed pattern có sẵn (hourglass/hexagonal/reentrant_bowtie/circle/...)
simp --version
```

## 3. `simp-analysis` - phân tích kết quả 1 batch đã chạy (entry point → `analysis/cli.py`)

```bash
python -m analysis.cli report --data-dir outputs/pipeline/phase1/circle/auxetic
python -m analysis.cli image-metrics --dir outputs/pipeline/phase1/circle/auxetic/sample_0000
```

## 4. Phase 1 - LHS Screening

```bash
python3 pipeline/phase1_screening/screening_parallel.py --help
```

```bash
python3 pipeline/phase1_screening/refine_params.py --help
```

## 5. Phase 2 - Multi-Batch Adaptive DOE

```bash
python3 -m pipeline.phase2_multi_batch.main --help
```

Sinh batch production đầy đủ (script chính dùng cho mọi lần sinh dataset lớn, xem A3/A4 trong
`EXPERIMENT_LOG.md`), dispatch optimizer MMA/OC per-seed tự động:

```bash
python3 analysis/scripts/generate_production_batch.py \
    --n-raw 3000 --run-dir outputs/phase3_a4_nu_raw --vary-nu
```

Flag quan trọng: `--only-seed <tên>` (chỉ sinh 1 seed), `--optimizer {auto,gate,mma}`,
`--vary-nu` (sample ν0 trong `PARAM_SPACE`, xem Nhóm 1/A trong `PROJECT_PLAN.md`).

## 6. Phase 3 - Dataset Build

```bash
python3 pipeline/phase3_dataset/build_npz.py --resolution 64
```

```bash
python3 pipeline/phase3_dataset/finalize_dataset.py \
    --resolution 64 --val-frac 0.15 --test-frac 0.15 --seed 42
```

## 7. Phase 4 - CNN Surrogate

Train:

```bash
python3 pipeline/phase4_surrogate/train.py --epochs 60 --batch_size 128 --output-name surrogate_best.pt
```

Với ν0 làm input phụ (A4, cần dataset có field `nu`):

```bash
python3 pipeline/phase4_surrogate/train.py \
    --include-nu0 --train-npz outputs/phase3_a4/train.npz --val-npz outputs/phase3_a4/val.npz \
    --output-name surrogate_a4_nu0.pt
```

Evaluate (đo R² trên test set):

```bash
python3 pipeline/phase4_surrogate/evaluate.py --ckpt outputs/phase4/surrogate_best.pt
```

Bootstrap CI:

```bash
python3 pipeline/phase4_surrogate/bootstrap_ci.py --help
```

## 8. Phase 5 - Conditional VAE

### 8.1. Train

2-stage bắt buộc theo kinh nghiệm dự án (train from-scratch với real-physics thất bại, xem
`EXPERIMENT_LOG.md` 2026-07-25): base trước, fine-tune real-physics sau.

```bash
# Stage 1 - base (không real-physics)
python3 pipeline/phase5_cvae/train.py --epochs 100 --latent-dim 32 --output-name cvae_base.pt

# Stage 2 - fine-tune với differentiable real-physics loss
python3 pipeline/phase5_cvae/train.py \
    --resume-from outputs/phase5/cvae_base.pt --lambda-real-physics 1.0 \
    --epochs 30 --output-name cvae_finetuned.pt
```

Với điều kiện optional (volfrac/void_size_frac, A6/Nhóm 2 trong `PROJECT_PLAN.md`):

```bash
python3 pipeline/phase5_cvae/train.py --extended-condition --lambda-volfrac 1.0 [...]
```

Với ν0 làm condition (A6, cần `--data-dir` trỏ dataset có field `nu`):

```bash
python3 pipeline/phase5_cvae/train.py \
    --include-nu0 --data-dir outputs/phase3_a4 --resume-from outputs/phase5/cvae_a4_control_base.pt \
    --lambda-real-physics 1.0 --output-name cvae_a4_nu0_finetuned.pt
```

Flag quan trọng khác: `--surrogate-path` (property-consistency loss), `--weighted-sampling`,
`--select-by fe_r2` và `--fe-eval-every N`. Pipeline v2 bắt buộc chọn checkpoint
bằng R² FE thật; `val_loss` không còn là lựa chọn hợp lệ.

Kiến trúc KAN + WIRE (2026-08-23): mọi `fc` trong cVAE giờ là `EfficientKANLinear`
(encoder lẫn decoder). Chọn decoder bằng:

```bash
# Decoder conv (tương thích ngược checkpoint cũ)
source /home/tbm/miniconda3/etc/profile.d/conda.sh && conda activate simp
python3 pipeline/phase5_cvae/train.py --decoder-type conv --fe-eval-every 10 --output-name cvae_base.pt

# Decoder WIRE INR (Task 2 - chưa train; cần checkpoint wire riêng)
python3 pipeline/phase5_cvae/train.py --decoder-type wire --fe-eval-every 10 \
    --wire-hidden-dim 128 --wire-omega0 10.0 --wire-s0 10.0 --output-name cvae_wire.pt
```

### 8.2. Sinh mẫu nhanh (KHÔNG verify FE - chỉ xem hình dạng)

```bash
python3 pipeline/phase5_cvae/sample.py --v12 -0.5 --v21 -0.5 --n 8
```

```bash
python3 pipeline/phase5_cvae/sample.py \
    --v12 -0.5 --v21 -0.5 --nu0 0.35 \
    --ckpt outputs/phase5/cvae_a4_nu0_finetuned.pt --n 8
```

Xuất ảnh độ phân giải cao (chỉ tác dụng với checkpoint `decoder_type="wire"`;
decoder conv luôn xuất 64×64):

```bash
python3 pipeline/phase5_cvae/sample.py --v12 -0.6 --v21 -0.6 --n 8 \
    --ckpt outputs/phase5/cvae_wire.pt --resolution 256
```

### 8.3. `best_of_n_eval.py` - quy trình CHÍNH THỨC (verify bằng FE thật, dùng để báo cáo số liệu)

```bash
python3 pipeline/phase5_cvae/best_of_n_eval.py \
    --cvae-ckpt outputs/phase5/cvae_v2_finetuned.pt \
    --v12 -0.5 --v21 -0.5 --n-samples 30 --png-out /tmp/best_sample.png
```

Full benchmark (24 condition mặc định từ `test.npz`):

```bash
python3 pipeline/phase5_cvae/best_of_n_eval.py --cvae-ckpt outputs/phase5/cvae_gamma20.pt --n-samples 30
```

Với checkpoint A4/nu0:

```bash
python3 pipeline/phase5_cvae/best_of_n_eval.py \
    --cvae-ckpt outputs/phase5/cvae_a4_nu0_finetuned.pt \
    --data-dir outputs/phase3_a4 --nu0 0.35 --n-samples 30 --require-manufacturable
```

Flag quan trọng: `--k-fe-verify K` (mô phỏng chi phí triển khai thật - surrogate xếp hạng, chỉ verify
FE top-K, thay vì oracle chấm hết N mẫu), `--require-manufacturable`, `--w-accuracy/--w-manuf/--w-aesthetic`
(mặc định 0.6/0.3/0.1, composite score).

## 10. Roadmap model experiments

Các lệnh sau chạy sau khi đã activate môi trường `simp` ở mục 0:

```bash
python -m pipeline.mno.train --data outputs/mno_dataset.npz --output outputs/mno.pt
python -m pipeline.mno.evaluate --data outputs/mno_dataset.npz --checkpoint outputs/mno.pt
python -m pipeline.kinn.train --epochs 100 --output outputs/kinn.pt
python -m pipeline.ickans.train --epochs 100 --output outputs/ickan.pt
python -m pipeline.ickans.evaluate --checkpoint outputs/ickan.pt
```

MNO cần NPZ có `images` và `displacements` 18 kênh. Khi chưa có `mamba_ssm`,
MNO tự dùng fallback GRU; KINN giữ graph huấn luyện bằng PyTorch khi chưa có
JAX/AMG. Các lệnh trên là điểm bắt đầu experiment, không tự động chứng minh
KPI production.

### 8.4. Self-play (active learning round-trip surrogate ↔ cVAE)

```bash
python3 pipeline/phase5_cvae/self_play.py --rounds 2 --n-conditions 8 --help
```

### 8.5. Bootstrap CI / coverage eval

```bash
python3 pipeline/phase5_cvae/bootstrap_ci.py --help
python3 pipeline/phase5_cvae/coverage_eval.py --help
```

## 9. Analysis scripts (so sánh baseline, kiểm định độc lập)

| Script | Việc | Lệnh mẫu |
|---|---|---|
| `analysis/scripts/baseline_comparison.py` | So retrieval-vs-cVAE (Nhóm 4, `PROJECT_PLAN.md`) | `python3 analysis/scripts/baseline_comparison.py --cvae-ckpt outputs/phase5/cvae_v2_finetuned.pt --n-conditions 24` |
| `analysis/scripts/ood_baseline_comparison.py` | So OOD (sign-flip), xem `EXPERIMENT_LOG.md` mục 2026-08-04 | `python3 analysis/scripts/ood_baseline_comparison.py --help` |
| `analysis/scripts/novelty_diversity_eval.py` | Đo đa dạng/novelty mẫu sinh ra | `python3 analysis/scripts/novelty_diversity_eval.py --help` |
| `analysis/scripts/run_independent_fe_check.py` | Kiểm chứng FE độc lập bằng `scikit-fem` | `python3 analysis/scripts/run_independent_fe_check.py --help` |
| `analysis/scripts/build_design_library.py` | Gom thư viện thiết kế đã sinh | `python3 analysis/scripts/build_design_library.py --help` |
| `analysis/scripts/backfill_f1_f2_npz.py` | Backfill f1/f2 vào npz cũ (Nhóm 2) | `python3 analysis/scripts/backfill_f1_f2_npz.py --help` |

Mọi script trên đều có `--help` đầy đủ - bảng chỉ để biết script nào làm việc gì, không thay thế `--help`.

## 10. Sự cố thường gặp

- `ModuleNotFoundError: No module named 'torch.utils'` / `AttributeError: module 'torch' has no attribute '__version__'` → chưa `conda activate simp` (mục 0).
- `FileNotFoundError` khi chạy `sample.py`/`best_of_n_eval.py` → checkpoint `--ckpt`/`--cvae-ckpt` chưa tồn tại, kiểm tra `outputs/phase5/` hoặc train lại (mục 8.1).
- `best_of_n_eval.py --include-nu0`/`--nu0` báo lỗi thiếu field `nu` → thiếu `--data-dir` trỏ đúng dataset có field đó (vd `outputs/phase3_a4/`), không dùng `outputs/phase3/` mặc định (xem A6, `PROJECT_PLAN.md`).
