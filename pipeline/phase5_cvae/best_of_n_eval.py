"""
Phase 5 - best_of_n_eval.py
============================================================
Thay vì cố ép R2(FE thật) của **1 lần sinh duy nhất** tiến về 0 (self-play,
ensemble - xem README §5, cả 2 đều không đủ trong ngân sách thời gian đã
thử), áp dụng chiến lược "generate-nhiều-rồi-lọc-bằng-FE-thật" giống bài báo
Deep-DRAM (Pahlavani et al., Advanced Materials 2024, DOI
10.1002/adma.202303481): cVAE sinh N ứng viên cho mỗi target, KHÔNG qua
surrogate để quyết định - chạy FE thật trên cả N ứng viên và báo cáo ứng
viên tốt nhất tìm được. Đây không phải "sửa" gradient training, mà đổi tiêu
chí thành công: "tồn tại >=1 trong N mẫu sinh ra thật sự auxetic" thay vì
"mẫu sinh ra (duy nhất) có R2 cao qua surrogate" - né hoàn toàn vấn đề
exploitation vì FE thật luôn là trọng tài cuối cùng, không phải surrogate.

So sánh trực tiếp với hit_rate 1-mẫu đã đo trong self_play.verify_round
(cùng 24 condition, seed=123, tập test.npz) để biết best-of-N cải thiện
được bao nhiêu so với single-shot.

Roadmap 6.2/6.3 (manufacturability.py): đúng Poisson ratio (FE thật) không
đồng nghĩa "sản xuất được" - thêm --require-manufacturable để loại ứng
viên rời rạc/nét quá mảnh/không ghép ô tuần hoàn khỏi việc xếp hạng & chọn
best-of-N. Mặc định TẮT (giữ hành vi gốc, chỉ tối ưu Poisson ratio).

force_periodic() (manufacturability.py, thêm từ nhánh research/auxetic-
breakthrough - xem EXPERIMENT_LOG.md mục "Phase 6"): ép cứng periodicity
bằng 1 phép gán (không cần học) TRƯỚC khi chấm manufacturability/FE - đo
được passes_all 1,7%->19,5% trên chính checkpoint cvae_gamma20.pt, chi phí
sai số ν₁₂ trung bình ~0,02. Mặc định BẬT (--no-force-periodic để tắt, so
sánh với hành vi gốc trước khi có cải tiến này).

Cách chạy:
    python3 pipeline/phase5_cvae/best_of_n_eval.py \\
        --cvae-ckpt outputs/phase5/cvae_gamma20.pt --n-samples 30
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from dataset import CVAEDataset                                # noqa: E402
from verify_fe import FE_PARAMS, resize_to_fe_grid, evaluate_density_field  # noqa: E402
from self_play import load_cvae                                # noqa: E402
from losses import load_frozen_surrogate, SURROGATE_PATH        # noqa: E402
from manufacturability import check_manufacturability, force_periodic  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHASE3_DIR = os.path.join(REPO_ROOT, "outputs", "phase3")
PHASE5_DIR = os.path.join(REPO_ROOT, "outputs", "phase5")


def best_of_n(cvae_ckpt_path: str, n_conditions: int, n_samples: int,
              device: str, seed: int = 123, k_fe_verify: int = None,
              surrogate_path: str = None, require_manufacturable: bool = False,
              min_feature_px: int = 2, periodicity_tol: float = 0.1,
              custom_condition: np.ndarray = None, save_best_png: str = None,
              apply_force_periodic: bool = True):
    """CÙNG tập condition với self_play.verify_round (seed mặc định 123,
    test.npz) để so sánh apples-to-apples. Với mỗi condition, sinh n_samples
    ứng viên.

    k_fe_verify=None (mặc định): chấm TẤT CẢ n_samples bằng FE thật - "oracle"
    cận trên lý tưởng, không quan tâm chi phí.

    k_fe_verify=K (kiểu Deep-DRAM thật, xem docstring đầu file): dùng
    surrogate (rẻ, 1 forward pass batch) xếp hạng n_samples ứng viên theo
    |dự đoán - target|, CHỈ chạy FE thật (đắt) trên top-K gần nhất - mô
    phỏng đúng chi phí thực tế triển khai (K lần FE thay vì N lần).

    custom_condition: nếu truyền vào (vd np.array([v12, v21])), bỏ qua
    test.npz, chỉ chạy best-of-N cho ĐÚNG 1 target này (dùng để test tay 1
    target tuỳ ý qua CLI --v12/--v21, xem main()). n_conditions bị bỏ qua
    trong trường hợp này.

    save_best_png: nếu truyền vào (đường dẫn .png), lưu ảnh ứng viên tốt
    nhất (chọn bởi FE thật) ra file - chỉ có ý nghĩa khi custom_condition
    được dùng (1 condition duy nhất nên "ứng viên tốt nhất" không mơ hồ).

    apply_force_periodic: mặc định True - áp force_periodic() (xem
    manufacturability.py) lên MỌI ứng viên trước khi chấm manufacturability
    và FE, ép cứng periodicity bằng 1 phép gán thay vì phải trông chờ cVAE
    học đúng. Đặt False để tái hiện hành vi gốc (trước khi có cải tiến
    này) hoặc so sánh có/không."""
    torch.manual_seed(seed)
    if custom_condition is not None:
        conditions = [np.asarray(custom_condition, dtype=np.float32)]
    else:
        test_ds = CVAEDataset(os.path.join(PHASE3_DIR, "test.npz"))
        rng = np.random.default_rng(seed)
        idxs = rng.choice(len(test_ds), size=n_conditions, replace=False)
        conditions = [test_ds[i][1].numpy() for i in idxs]

    model = load_cvae(cvae_ckpt_path, device)
    surrogate = None
    n_seeds = None
    if k_fe_verify is not None:
        path = surrogate_path or SURROGATE_PATH
        surrogate, target_names = load_frozen_surrogate(device=device, path=path)
        idx_v12 = target_names.index("v12")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        n_seeds = ckpt["n_seeds"]

    per_condition = []
    n_auxetic_targets = 0
    n_hits_best_of_n = 0
    n_hits_single_shot = 0  # mẫu đầu tiên trong N mẫu - để so sánh apples-to-apples cùng 1 lần chạy
    n_fe_calls_total = 0

    for cond in conditions:
        cond_t = torch.tensor(cond, dtype=torch.float32, device=device)
        is_auxetic_target = cond[0] < 0
        if is_auxetic_target:
            n_auxetic_targets += 1

        imgs = []
        for i in range(n_samples):
            with torch.no_grad():
                img = model.generate(cond_t, n_samples=1, device=device)
            imgs.append(img.squeeze().cpu().numpy().astype(np.float32))

        if apply_force_periodic:
            imgs = [force_periodic(img) for img in imgs]

        # roadmap 6.2/6.3: connectivity + min-feature-size + periodicity -
        # cấu trúc "sản xuất được" là điều kiện tách biệt với việc đạt đúng
        # Poisson ratio, xem manufacturability.py docstring.
        manuf_reports = [
            check_manufacturability((img > 0.5).astype(np.float32),
                                     min_feature_px=min_feature_px,
                                     periodicity_tol=periodicity_tol)
            for img in imgs
        ]
        n_manufacturable = sum(1 for r in manuf_reports if r["passes_all"])
        candidate_pool = range(n_samples)
        if require_manufacturable:
            manufacturable_idx = [i for i, r in enumerate(manuf_reports) if r["passes_all"]]
            if manufacturable_idx:
                candidate_pool = manufacturable_idx
            # nếu KHÔNG có ứng viên nào manufacturable, giữ nguyên cả N mẫu
            # (không loại hết, tránh mất cả condition này khỏi báo cáo) -
            # nhưng frac_manufacturable=0 trong per_condition sẽ phản ánh
            # đúng việc target này chưa có lời giải sản xuất được.

        fe_order = candidate_pool
        if k_fe_verify is not None:
            # dùng seed one-hot mặc định (seed đầu tiên trong seed_classes) vì
            # ảnh generate không có nhãn seed thật - cùng xấp xỉ với
            # property_consistency_loss (xem losses.py docstring).
            imgs_t = torch.tensor(np.stack(imgs), device=device).unsqueeze(1)
            seed_vec = torch.zeros(n_samples, n_seeds, device=device)
            seed_vec[:, 0] = 1.0
            with torch.no_grad():
                pred = surrogate(imgs_t, seed_vec)
            surrogate_v12 = pred[:, idx_v12].cpu().numpy()
            pool = list(candidate_pool)
            order_within_pool = np.argsort(np.abs(surrogate_v12[pool] - cond[0]))[:k_fe_verify]
            fe_order = [pool[j] for j in order_within_pool]

        v12_reals = []
        v12_reals_img_idx = []  # imgs[] index cho từng entry của v12_reals (loop bỏ qua exception nên KHÔNG khớp 1-1 với fe_order)
        for i in fe_order:
            img_bin = (imgs[i] > 0.5).astype(np.float32)
            img_fe = resize_to_fe_grid(img_bin, FE_PARAMS["nely"], FE_PARAMS["nelx"])
            try:
                v12_fe, v21_fe, _ = evaluate_density_field(img_fe, FE_PARAMS)
            except Exception:
                continue
            v12_reals.append(v12_fe)
            v12_reals_img_idx.append(i)
            n_fe_calls_total += 1

        if not v12_reals:
            continue

        v12_reals = np.array(v12_reals)
        best_idx = int(np.argmin(np.abs(v12_reals - cond[0])))
        v12_best = float(v12_reals[best_idx])
        if save_best_png:
            best_img_idx = v12_reals_img_idx[best_idx]
            os.makedirs(os.path.dirname(save_best_png) or ".", exist_ok=True)
            arr = ((imgs[best_img_idx] > 0.5).astype(np.float32) * 255).astype(np.uint8)
            Image.fromarray(arr, mode="L").save(save_best_png)
        img_bin0 = (imgs[0] > 0.5).astype(np.float32)
        img_fe0 = resize_to_fe_grid(img_bin0, FE_PARAMS["nely"], FE_PARAMS["nelx"])
        try:
            v12_first, _, _ = evaluate_density_field(img_fe0, FE_PARAMS)
        except Exception:
            v12_first = float("nan")

        if is_auxetic_target and v12_best < 0:
            n_hits_best_of_n += 1
        if is_auxetic_target and not np.isnan(v12_first) and v12_first < 0:
            n_hits_single_shot += 1

        per_condition.append({
            "target_v12": float(cond[0]),
            "is_auxetic_target": bool(is_auxetic_target),
            "n_valid_samples": len(v12_reals),
            "v12_first": v12_first,
            "v12_best": v12_best,
            "v12_all": v12_reals.tolist(),
            "frac_auxetic_among_samples": float((v12_reals < 0).mean()) if is_auxetic_target else None,
            "n_manufacturable": n_manufacturable,
            "frac_manufacturable": n_manufacturable / n_samples,
        })

    hit_rate_best_of_n = n_hits_best_of_n / n_auxetic_targets if n_auxetic_targets else float("nan")
    hit_rate_single_shot = n_hits_single_shot / n_auxetic_targets if n_auxetic_targets else float("nan")

    targets_best = np.array([c["target_v12"] for c in per_condition])
    preds_best = np.array([c["v12_best"] for c in per_condition])
    ss_res = ((targets_best - preds_best) ** 2).sum()
    ss_tot = ((targets_best - targets_best.mean()) ** 2).sum()
    r2_best_of_n = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    mean_frac_manufacturable = float(np.mean([c["frac_manufacturable"] for c in per_condition]))

    return {
        "n_conditions": len(conditions),
        "n_samples_per_condition": n_samples,
        "k_fe_verify": k_fe_verify if k_fe_verify is not None else n_samples,
        "n_fe_calls_total": n_fe_calls_total,
        "n_auxetic_targets": n_auxetic_targets,
        "hit_rate_single_shot": hit_rate_single_shot,
        "hit_rate_best_of_n": hit_rate_best_of_n,
        "r2_fe_v12_best_of_n": r2_best_of_n,
        "require_manufacturable": require_manufacturable,
        "apply_force_periodic": apply_force_periodic,
        "mean_frac_manufacturable": mean_frac_manufacturable,
        "per_condition": per_condition,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cvae-ckpt", type=str,
                         default=os.path.join(PHASE5_DIR, "cvae_gamma20.pt"))
    parser.add_argument("--n-conditions", type=int, default=24)
    parser.add_argument("--n-samples", type=int, default=30,
                         help="Số ứng viên sinh ra MỖI condition, tất cả đều chấm bằng FE thật.")
    parser.add_argument("--seed", type=int, default=123,
                         help="PHẢI giống seed mặc định của self_play.verify_round (123) "
                              "để so sánh cùng 1 tập condition.")
    parser.add_argument("--out", type=str, default=None,
                         help="Mặc định: outputs/phase5/self_play/best_of_n_result.json, TRỪ "
                              "khi dùng --v12/--v21 (test tay 1 target) thì mặc định đổi sang "
                              "outputs/phase5/self_play/custom_test/ để KHÔNG ghi đè file "
                              "benchmark 24-condition gốc đã commit vào git.")
    parser.add_argument("--k-fe-verify", type=int, default=None,
                         help="Chỉ chạy FE thật trên top-K ứng viên (xếp hạng bởi surrogate, "
                              "rẻ) trong N ứng viên sinh ra - kiểu Deep-DRAM thật (tiết kiệm "
                              "chi phí FE). Bỏ trống = chấm FE trên TẤT CẢ N ứng viên (oracle, "
                              "cận trên lý tưởng, không quan tâm chi phí).")
    parser.add_argument("--surrogate-path", type=str, default=None,
                         help="Surrogate dùng để xếp hạng khi --k-fe-verify được set. "
                              "Bỏ trống = SURROGATE_PATH mặc định trong losses.py.")
    parser.add_argument("--require-manufacturable", action="store_true",
                         help="Roadmap 6.2/6.3: loại ứng viên không liên thông / có nét "
                              "mảnh hơn ngưỡng / không ghép ô tuần hoàn được (xem "
                              "manufacturability.py) khỏi việc xếp hạng và chọn best-of-N. "
                              "Mặc định tắt (chỉ tối ưu đúng Poisson ratio, không lọc "
                              "manufacturability, giống hành vi gốc).")
    parser.add_argument("--min-feature-px", type=int, default=2,
                         help="Ngưỡng độ rộng nét tối thiểu (pixel, lưới 64x64) cho "
                              "--require-manufacturable.")
    parser.add_argument("--periodicity-tol", type=float, default=0.1,
                         help="Tỉ lệ pixel-biên tối đa được phép không khớp khi ghép ô "
                              "tuần hoàn cho --require-manufacturable.")
    parser.add_argument("--no-force-periodic", action="store_true",
                         help="Tắt force_periodic() (xem manufacturability.py) - mặc định "
                              "BẬT, ép cứng periodicity bằng 1 phép gán trước khi chấm "
                              "manufacturability/FE (đo được passes_all 1,7%%->19,5%% trên "
                              "cvae_gamma20.pt, chi phí sai số ν₁₂ trung bình ~0,02 - xem "
                              "EXPERIMENT_LOG.md mục Phase 6). Dùng cờ này để tái hiện hành "
                              "vi gốc trước khi có cải tiến.")
    parser.add_argument("--v12", type=float, default=None,
                         help="Test tay 1 target tuỳ ý thay vì lấy ngẫu nhiên từ test.npz - "
                              "PHẢI đi kèm --v21. Khi dùng, --n-conditions bị bỏ qua "
                              "(chỉ chạy đúng 1 condition này) và ảnh ứng viên tốt nhất "
                              "được lưu ra PNG (xem --png-out).")
    parser.add_argument("--v21", type=float, default=None,
                         help="Đi kèm --v12, xem help của --v12.")
    parser.add_argument("--png-out", type=str, default=None,
                         help="Đường dẫn lưu ảnh ứng viên tốt nhất (chỉ dùng khi --v12/--v21 "
                              "được set). Mặc định: outputs/phase5/self_play/custom_test/"
                              "v12_{v12}_v21_{v21}_best.png")
    args = parser.parse_args()
    if (args.v12 is None) != (args.v21 is None):
        parser.error("--v12 và --v21 phải đi cùng nhau")

    custom_condition = None
    save_best_png = None
    if args.v12 is not None:
        custom_condition = np.array([args.v12, args.v21], dtype=np.float32)
        save_best_png = args.png_out or os.path.join(
            PHASE5_DIR, "self_play", "custom_test",
            f"v12_{args.v12:.3f}_v21_{args.v21:.3f}_best.png")
        out_path = args.out or os.path.join(
            PHASE5_DIR, "self_play", "custom_test",
            f"v12_{args.v12:.3f}_v21_{args.v21:.3f}_result.json")
    else:
        out_path = args.out or os.path.join(PHASE5_DIR, "self_play", "best_of_n_result.json")
    args.out = out_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = best_of_n(args.cvae_ckpt, args.n_conditions, args.n_samples, device, args.seed,
                        k_fe_verify=args.k_fe_verify, surrogate_path=args.surrogate_path,
                        require_manufacturable=args.require_manufacturable,
                        min_feature_px=args.min_feature_px,
                        periodicity_tol=args.periodicity_tol,
                        custom_condition=custom_condition,
                        save_best_png=save_best_png,
                        apply_force_periodic=not args.no_force_periodic)

    print(f"Checkpoint: {args.cvae_ckpt}")
    print(f"N condition: {result['n_conditions']} ({result['n_auxetic_targets']} auxetic target) "
          f"x {result['n_samples_per_condition']} mẫu/condition, k_fe_verify={result['k_fe_verify']} "
          f"({result['n_fe_calls_total']} lần gọi FE thật tổng cộng)")
    print(f"  hit_rate (1 mẫu duy nhất, mẫu đầu)      = {result['hit_rate_single_shot']:.3f}")
    print(f"  hit_rate (best-of-N, chọn bởi FE oracle) = {result['hit_rate_best_of_n']:.3f}")
    if custom_condition is not None and result["per_condition"]:
        c = result["per_condition"][0]
        print(f"  target v12={c['target_v12']:.4f} -> best_v12(FE thật)={c['v12_best']:.4f} "
              f"(sai số {abs(c['v12_best']-c['target_v12']):.4f})")
        print(f"  Ảnh ứng viên tốt nhất đã lưu: {save_best_png}")
    print(f"  R2(FE, best-of-N)                        = {result['r2_fe_v12_best_of_n']:.4f}")
    print(f"  frac manufacturable (6.2/6.3, TB các cond) = {result['mean_frac_manufacturable']:.3f} "
          f"(force_periodic={'BẬT' if result['apply_force_periodic'] else 'TẮT'})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nĐã lưu: {args.out}")


if __name__ == "__main__":
    main()
