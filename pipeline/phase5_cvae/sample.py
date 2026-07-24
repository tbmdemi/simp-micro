"""
Phase 5 - sample.py
=====================
Sinh geometry mới từ 1 target Poisson ratio (v12, v21) bất kỳ, dùng cVAE
đã train. Đây là "inverse design" thật sự - input chỉ là con số mong muốn,
output là ảnh density field.

CẢNH BÁO QUAN TRỌNG - ĐỌC TRƯỚC KHI DÙNG KẾT QUẢ SCRIPT NÀY:
verify_fe.py đã xác nhận ảnh sinh ra bởi BẤT KỲ checkpoint nào (gamma
1-300) thường KHÔNG đạt đúng Poisson ratio thật khi kiểm bằng FE thật -
R2(FE thật) âm nặng ở mọi gamma đã thử, xem
outputs/phase5/fe_verification_report.json và mục Phase 5 trong README.
Script này chỉ sinh 1 mẫu/lần gọi, KHÔNG lọc qua FE - dùng để xem NHANH
hình dạng generator sinh ra, KHÔNG dùng trực tiếp kết quả cho mục đích
thực tế.

Quy trình CHÍNH THỨC để lấy 1 geometry đáng tin cậy là best_of_n_eval.py
(sinh N ứng viên, chấm điểm bằng FE thật, giữ ứng viên tốt nhất -
R2=+0.44..+0.60, hit rate 100% trên tập test, xem README §5):

    python3 pipeline/phase5_cvae/best_of_n_eval.py \\
        --cvae-ckpt outputs/phase5/cvae_gamma20.pt --n-samples 30

Cách chạy (chỉ để xem nhanh, không phải quy trình chính thức):
    python3 pipeline/phase5_cvae/sample.py --v12 -0.6 --v21 -0.6 --n 8

Output: outputs/phase5/samples/v12_-0.60_v21_-0.60/sample_XX.png
        (ảnh grayscale, trắng = vật liệu, đen = rỗng - đúng convention
        Phase 3 build_npz.py: pixel 1 = vật liệu)

LƯU Ý: chỉ nên request v12 trong khoảng ~[-0.81, 0.37] (phạm vi dataset
train, xem usage_note trong outputs/phase4/surrogate_for_phase5.pt). Ngoài
khoảng này là ngoại suy, surrogate + cVAE đều không đáng tin.
"""
import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from model import CVAE  # noqa: E402
from manufacturability import force_periodic  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHASE5_DIR = os.path.join(REPO_ROOT, "outputs", "phase5")
CKPT_PATH = os.path.join(PHASE5_DIR, "cvae_best.pt")


def load_model(device="cpu", ckpt_path=CKPT_PATH):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = CVAE(condition_dim=ckpt["condition_dim"],
                 latent_dim=ckpt["latent_dim"],
                 resolution=ckpt["resolution"],
                 channels=ckpt.get("channels", (32, 64, 128, 256)))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def save_png(image_tensor: torch.Tensor, path: str, apply_force_periodic: bool = True):
    """apply_force_periodic (mặc định True): ép cứng periodicity bằng 1
    phép gán (xem manufacturability.py::force_periodic) trước khi lưu ảnh -
    đo được passes_all 1,7%->19,5% trên cvae_gamma20.pt (nhánh research/
    auxetic-breakthrough, xem EXPERIMENT_LOG.md mục "Phase 6"), chi phí sai
    số ν₁₂ trung bình ~0,02. Không thay đổi hành vi generate() của model -
    chỉ hậu xử lý ảnh trước khi ghi file."""
    img = image_tensor.squeeze().cpu().numpy()
    if apply_force_periodic:
        img = force_periodic(img)
    arr = (img * 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v12", type=float, required=True)
    parser.add_argument("--v21", type=float, required=True)
    parser.add_argument("--n", type=int, default=8, help="số mẫu sinh ra")
    parser.add_argument("--out", type=str, default=None,
                         help="thư mục output tuỳ chỉnh (mặc định tự đặt theo v12/v21)")
    parser.add_argument("--ckpt", type=str, default=CKPT_PATH,
                         help="checkpoint cVAE (.pt) để load - mặc định outputs/phase5/"
                              "cvae_best.pt. Giống --cvae-ckpt của best_of_n_eval.py.")
    parser.add_argument("--no-force-periodic", action="store_true",
                         help="Tắt force_periodic() (mặc định BẬT - xem manufacturability.py "
                              "và EXPERIMENT_LOG.md mục Phase 6) trước khi lưu ảnh.")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(
            f"Không tìm thấy {args.ckpt} - hãy chạy train.py trước."
        )

    print("=" * 70)
    print("CẢNH BÁO: sample.py sinh 1 MẪU DUY NHẤT mỗi lần gọi, KHÔNG lọc")
    print("qua FE thật. verify_fe.py đã xác nhận ảnh sinh ra bởi cVAE thường")
    print("KHÔNG đạt đúng Poisson ratio mong muốn khi kiểm bằng FE thật, dù")
    print("R2 qua surrogate trông cao (surrogate exploitation - xem README §5,")
    print("outputs/phase5/fe_verification_report.json).")
    print("Script này chỉ nên dùng để xem NHANH hình dạng generator sinh ra.")
    print("Muốn kết quả đáng tin cậy, dùng quy trình CHÍNH THỨC best_of_n_eval.py")
    print("(sinh N ứng viên, chọn bằng FE thật - R2=+0.44..+0.60, hit rate 100%):")
    print("    python3 pipeline/phase5_cvae/best_of_n_eval.py "
          "--cvae-ckpt outputs/phase5/cvae_gamma20.pt --n-samples 30")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device=device, ckpt_path=args.ckpt)

    condition = torch.tensor([args.v12, args.v21], dtype=torch.float32, device=device)
    samples = model.generate(condition, n_samples=args.n, device=device)  # (n,1,64,64)

    out_dir = args.out or os.path.join(
        PHASE5_DIR, "samples", f"v12_{args.v12:.2f}_v21_{args.v21:.2f}"
    )
    os.makedirs(out_dir, exist_ok=True)
    for i in range(args.n):
        save_png(samples[i], os.path.join(out_dir, f"sample_{i:02d}.png"),
                 apply_force_periodic=not args.no_force_periodic)

    print(f"Đã sinh {args.n} mẫu cho target v12={args.v12}, v21={args.v21}")
    print(f"Lưu tại: {out_dir}")
    print("Nhắc lại: đây là mẫu CHƯA qua lọc FE - dùng best_of_n_eval.py để có "
          "kết quả đáng tin cậy trước khi dùng cho mục đích thực tế.")


if __name__ == "__main__":
    main()