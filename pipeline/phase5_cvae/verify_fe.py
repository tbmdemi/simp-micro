"""
Phase 5 - verify_fe.py
============================================================
Kiểm chứng ĐỘC LẬP với surrogate: lấy ảnh cVAE sinh ra, binarize, chạy qua
FE solver + homogenization THẬT (simp/core/solver.py, simp/homogenization/
compute.py) để lấy (v12, v21) THẬT - không qua surrogate CNN, nên không thể
bị decoder "đánh lừa" như property_consistency_loss/property_accuracy vốn
dùng chung 1 surrogate đông cứng cho cả train lẫn eval.

ĐÃ XÁC NHẬN (2026-07-22, xem outputs/phase5/fe_verification_report.json và
README mục Phase 5): R2(FE thật) âm nặng ở mọi gamma đã thử (1-300), gap
surrogate-vs-FE càng doãng khi gamma càng cao - đúng là exploitation.

Lưu ý khi chạy lại:
  - FE_PARAMS['nelx']/['nely'] PHẢI là 50 (không phải 80 như
    pipeline/params.py FIXED_PARAMS - config đó chỉ dùng cho Phase 1, không
    dùng cho dataset thật, xem comment ở params.py).
  - `penal` biến thiên theo mẫu lúc sinh dataset gốc; sanity_check() dùng
    đúng giá trị thật lưu trong test.npz['params'], nhưng run_verification()
    (ảnh cVAE sinh mới, không có "penal gốc") dùng giá trị đại diện cố định
    3.0 - xấp xỉ, không chính xác tuyệt đối.

Cách chạy: sanity-check trước (bắt buộc pass, xác nhận FE_PARAMS/resize
đúng), rồi mới chạy full:
    python3 pipeline/phase5_cvae/verify_fe.py --sanity-check
    python3 pipeline/phase5_cvae/verify_fe.py --gammas 20 100 300 --n-per-condition 3

Output: outputs/phase5/fe_verification_report.json
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import CVAE                     # noqa: E402
from dataset import CVAEDataset            # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHASE3_DIR = os.path.join(REPO_ROOT, "outputs", "phase3")
PHASE5_DIR = os.path.join(REPO_ROOT, "outputs", "phase5")

# ============================================================
# CHỈNH CÁC GIÁ TRỊ NÀY CHO KHỚP VỚI pipeline/params.py FIXED_PARAMS
# CỦA REPO BẠN TRƯỚC KHI CHẠY - xem cảnh báo ở docstring trên.
# ============================================================
FE_PARAMS = {
    "nelx": 50,
    "nely": 50,
    "E0": 199.0,
    "Emin": 1e-9,
    "nu": 0.3,
    "penal": 3.0,     # xấp xỉ - xem ghi chú docstring
    "rho0": 1.0,
}


def _import_simp():
    """Import các hàm simp cần thiết. Tách riêng thành hàm để lỗi import
    (nếu FE_PARAMS/API không khớp phiên bản simp/ đang dùng) báo rõ ràng
    ngay từ đầu, thay vì lỗi mơ hồ giữa chừng."""
    from simp.materials.isotropic import Material
    from simp.core.fem import build_dof_mesh
    from simp.core.pbc import build_pbc
    from simp.core.solver import solve_fe
    from simp.homogenization.compute import compute_homogenized_tensor
    from simp.objectives.auxetic import compute_nu12, compute_nu21
    return (Material, build_dof_mesh, build_pbc, solve_fe,
            compute_homogenized_tensor, compute_nu12, compute_nu21)


def evaluate_density_field(xPhys: np.ndarray, fe_params: dict = FE_PARAMS):
    """1 lần FE solve + homogenization THẬT (không optimize, chỉ forward-eval)
    trên xPhys (nely, nelx) trong [0,1] -> (v12, v21, Q)."""
    (Material, build_dof_mesh, build_pbc, solve_fe,
     compute_homogenized_tensor, compute_nu12, compute_nu21) = _import_simp()

    nely, nelx = xPhys.shape
    material = Material(E0=fe_params["E0"], Emin=fe_params["Emin"], nu=fe_params["nu"])
    nodenrs, edofVec, edofMat, iK, jK = build_dof_mesh(nelx, nely)
    pbc = build_pbc(nelx, nely, nodenrs)

    U, U0 = solve_fe(
        xPhys, material.KE, iK, jK, pbc,
        fe_params["penal"], fe_params["E0"], fe_params["Emin"],
        rho0=fe_params["rho0"],
    )
    # U là trường dao động (fluctuation) - phải cộng U0 trước khi tính Q
    # (xem README "Key Bugfixes" - runner.py từng có bug thiếu bước này).
    U_total = U0 + U
    Q, dQ, _ = compute_homogenized_tensor(
        U_total, U0, xPhys, material.KE, edofMat,
        fe_params["penal"], fe_params["E0"], fe_params["Emin"],
        rho0=fe_params["rho0"],
    )
    v12 = compute_nu12(Q)
    v21 = compute_nu21(Q)
    return v12, v21, Q


def resize_to_fe_grid(img64: np.ndarray, nely: int, nelx: int) -> np.ndarray:
    """Resize ảnh 64x64 (output cVAE, [0,1]) về đúng lưới FE (nely, nelx)
    bằng nearest-neighbor qua PIL (đơn giản, tránh phụ thuộc thêm thư viện;
    nearest giữ ảnh gần-nhị-phân sau khi binarize, không làm mờ lại)."""
    from PIL import Image
    im = Image.fromarray((img64 * 255).astype(np.uint8), mode="L")
    im = im.resize((nelx, nely), resample=Image.NEAREST)
    return np.asarray(im, dtype=np.float32) / 255.0


def sanity_check():
    """Kiểm tra evaluate_density_field() cho ra v12 khớp với v12 đã lưu
    trong test.npz cho chính density field đó (không qua cVAE) - PHẢI PASS
    trước khi tin số liệu run_verification(), nếu không nghĩa là
    FE_PARAMS/resize sai chứ không phải cVAE có vấn đề. Dùng `penal` THẬT
    của từng mẫu (test.npz['params']) thay vì hằng số, để không lẫn sai số
    xấp xỉ penal vào sai số cần cô lập ở bước này."""
    test_path = os.path.join(PHASE3_DIR, "test.npz")
    if not os.path.exists(test_path):
        print(f"[LỖI] Không tìm thấy {test_path}. Chạy script này trên máy "
              f"có sẵn outputs/phase3/test.npz.")
        return

    ds = CVAEDataset(test_path)
    raw = np.load(test_path, allow_pickle=True)
    param_names = list(raw["param_names"])
    penal_idx = param_names.index("penal")
    penal_all = raw["params"][:, penal_idx]

    n_check = min(10, len(ds))
    errors = []
    print(f"Sanity check trên {n_check} mẫu THẬT (không qua cVAE)...")
    print(f"FE_PARAMS đang dùng: {FE_PARAMS} (penal sẽ bị ghi đè bằng penal "
          f"THẬT của từng mẫu bên dưới)")
    for i in range(n_check):
        img, cond, seed_vec, vf = ds[i]
        img64 = img.squeeze(0).numpy()  # (64,64) trong [0,1]
        v12_saved = cond[0].item()
        v21_saved = cond[1].item()
        fe_params_i = dict(FE_PARAMS, penal=float(penal_all[i]))

        img_fe = resize_to_fe_grid(img64, FE_PARAMS["nely"], FE_PARAMS["nelx"])
        v12_fe, v21_fe, _ = evaluate_density_field(img_fe, fe_params_i)

        err12 = abs(v12_fe - v12_saved)
        err21 = abs(v21_fe - v21_saved)
        errors.append((err12, err21))
        print(f"  mẫu {i}: penal={fe_params_i['penal']:.3f} | "
              f"v12 lưu={v12_saved:+.4f} FE_tính_lại={v12_fe:+.4f} "
              f"(sai lệch={err12:.4f}) | v21 lưu={v21_saved:+.4f} "
              f"FE_tính_lại={v21_fe:+.4f} (sai lệch={err21:.4f})")

    mean_err12 = np.mean([e[0] for e in errors])
    mean_err21 = np.mean([e[1] for e in errors])
    print(f"\nSai lệch trung bình: v12={mean_err12:.4f}, v21={mean_err21:.4f}")
    if mean_err12 > 0.05 or mean_err21 > 0.05:
        print("[CẢNH BÁO] Sai lệch > 0.05 - KHÔNG nên tin kết quả verify_fe "
              "ở phần dưới cho tới khi tìm ra nguyên nhân (khả năng cao: "
              "FE_PARAMS['nelx']/['nely']/['penal'] sai, hoặc cách resize/"
              "binarize không khớp cách outputs/phase3 được tạo ra - kiểm "
              "tra lại pipeline/phase3/build_npz.py để xem chính xác cách "
              "PNG gốc -> 64x64 được tạo).")
    else:
        print("[OK] Sai lệch nhỏ - FE_PARAMS và cách resize đáng tin cậy. "
              "Có thể chạy full verification (bỏ --sanity-check).")


def load_cvae_checkpoint(gamma_tag: str, device):
    """Load 1 trong các checkpoint đã có: cvae_gamma{N}.pt hoặc
    cvae_best.pt (xem output ls outputs/phase5/ trong terminal của bạn)."""
    candidates = [
        os.path.join(PHASE5_DIR, f"cvae_gamma{gamma_tag}.pt"),
        os.path.join(PHASE5_DIR, "gamma_sweep_results", f"cvae_gamma{gamma_tag}.pt"),
        os.path.join(PHASE5_DIR, "gamma_sweep_results", f"cvae_best_gamma{gamma_tag}.pt"),
    ]
    ckpt_path = next((p for p in candidates if os.path.exists(p)), None)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint cho gamma={gamma_tag}. Đã thử: {candidates}. "
            f"Sửa lại đường dẫn trong load_cvae_checkpoint() cho khớp cấu trúc "
            f"thư mục thật của bạn (xem output `ls outputs/phase5/`)."
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = CVAE(
        condition_dim=ckpt.get("condition_dim", 2),
        latent_dim=ckpt["latent_dim"],
        resolution=ckpt.get("resolution", 64),
        channels=ckpt.get("channels", (32, 64, 128, 256)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Đã load {ckpt_path} (epoch={ckpt.get('epoch')}, "
          f"val_loss={ckpt.get('val_loss'):.2f}, gamma={ckpt.get('gamma')})")
    return model


def run_verification(gammas, n_conditions=10, n_per_condition=3, device="cpu"):
    """Bước 1.1 đầy đủ: với mỗi gamma, sinh ảnh ở nhiều condition, binarize,
    chạy FE thật, so sánh target vs surrogate (đọc lại từ eval_gamma*.json
    nếu có) vs FE thật."""
    test_ds = CVAEDataset(os.path.join(PHASE3_DIR, "test.npz"))

    rng = np.random.default_rng(42)
    # Lấy n_conditions cặp (v12,v21) THẬT từ test set để condition có ý
    # nghĩa vật lý (nằm trong phân phối train), thay vì bịa số ngẫu nhiên.
    idxs = rng.choice(len(test_ds), size=n_conditions, replace=False)
    conditions = [test_ds[i][1].numpy() for i in idxs]

    report = {"fe_params": FE_PARAMS, "results": {}}

    for gamma in gammas:
        print(f"\n=== gamma={gamma} ===")
        model = load_cvae_checkpoint(str(gamma), device)
        rows = []
        for cond in conditions:
            cond_t = torch.tensor(cond, dtype=torch.float32, device=device)
            for _ in range(n_per_condition):
                with torch.no_grad():
                    img = model.generate(cond_t, n_samples=1, device=device)
                img64 = img.squeeze().cpu().numpy()
                img_bin = (img64 > 0.5).astype(np.float32)
                img_fe = resize_to_fe_grid(img_bin, FE_PARAMS["nely"], FE_PARAMS["nelx"])
                try:
                    v12_fe, v21_fe, _ = evaluate_density_field(img_fe)
                except Exception as e:
                    print(f"  [LỖI FE] condition={cond}: {e}")
                    continue
                rows.append({
                    "target_v12": float(cond[0]), "target_v21": float(cond[1]),
                    "fe_v12": float(v12_fe), "fe_v21": float(v21_fe),
                })
                print(f"  target=({cond[0]:+.3f},{cond[1]:+.3f}) -> "
                      f"FE_thật=({v12_fe:+.3f},{v21_fe:+.3f})")

        if rows:
            targets = np.array([[r["target_v12"], r["target_v21"]] for r in rows])
            fe_preds = np.array([[r["fe_v12"], r["fe_v21"]] for r in rows])
            mae = np.abs(fe_preds - targets).mean(axis=0)
            ss_res = ((targets - fe_preds) ** 2).sum(axis=0)
            ss_tot = ((targets - targets.mean(axis=0)) ** 2).sum(axis=0)
            r2 = 1 - ss_res / ss_tot
            print(f"  --> R2(FE thật) v12={r2[0]:.4f} v21={r2[1]:.4f} | "
                  f"MAE v12={mae[0]:.4f} v21={mae[1]:.4f}")
            report["results"][str(gamma)] = {
                "rows": rows,
                "r2_fe": {"v12": float(r2[0]), "v21": float(r2[1])},
                "mae_fe": {"v12": float(mae[0]), "v21": float(mae[1])},
            }

    os.makedirs(PHASE5_DIR, exist_ok=True)
    out_path = os.path.join(PHASE5_DIR, "fe_verification_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nĐã lưu: {out_path}")

    print("\n=== SO SÁNH R2(surrogate, từ evaluation_report cũ) vs R2(FE thật) ===")
    for gamma in gammas:
        eval_path = os.path.join(PHASE5_DIR, "gamma_sweep_results", f"eval_gamma{gamma}.json")
        if os.path.exists(eval_path) and str(gamma) in report["results"]:
            with open(eval_path) as f:
                surro = json.load(f)["property_accuracy"]["v12"]["r2"]
            fe_r2 = report["results"][str(gamma)]["r2_fe"]["v12"]
            gap = surro - fe_r2
            flag = "  <-- CHÊNH LỆCH LỚN, nghi ngờ exploitation" if gap > 0.15 else ""
            print(f"  gamma={gamma:4d} | R2(surrogate)={surro:.3f} | "
                  f"R2(FE thật)={fe_r2:.3f} | gap={gap:+.3f}{flag}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity-check", action="store_true",
                         help="Chỉ chạy Bước 1.2 (kiểm tra FE_PARAMS đúng chưa)")
    parser.add_argument("--gammas", type=int, nargs="+", default=[20, 100, 300],
                         help="Danh sách gamma cần verify (khớp tên file "
                              "cvae_gamma{N}.pt đang có)")
    parser.add_argument("--n-conditions", type=int, default=10)
    parser.add_argument("--n-per-condition", type=int, default=3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.sanity_check:
        sanity_check()
    else:
        run_verification(args.gammas, args.n_conditions, args.n_per_condition, device)


if __name__ == "__main__":
    main()
