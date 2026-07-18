"""
Phase 5 - sample.py
=====================
Sinh geometry mới từ 1 target Poisson ratio (v12, v21) bất kỳ, dùng cVAE
đã train. Đây là "inverse design" thật sự - input chỉ là con số mong muốn,
output là ảnh density field.

Cách chạy:
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

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHASE5_DIR = os.path.join(REPO_ROOT, "outputs", "phase5")
CKPT_PATH = os.path.join(PHASE5_DIR, "cvae_best.pt")


def load_model(device="cpu", ckpt_path=CKPT_PATH):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = CVAE(condition_dim=ckpt["condition_dim"],
                 latent_dim=ckpt["latent_dim"],
                 resolution=ckpt["resolution"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def save_png(image_tensor: torch.Tensor, path: str):
    arr = (image_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v12", type=float, required=True)
    parser.add_argument("--v21", type=float, required=True)
    parser.add_argument("--n", type=int, default=8, help="số mẫu sinh ra")
    parser.add_argument("--out", type=str, default=None,
                         help="thư mục output tuỳ chỉnh (mặc định tự đặt theo v12/v21)")
    args = parser.parse_args()

    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy {CKPT_PATH} - hãy chạy train.py trước."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device=device)

    condition = torch.tensor([args.v12, args.v21], dtype=torch.float32, device=device)
    samples = model.generate(condition, n_samples=args.n, device=device)  # (n,1,64,64)

    out_dir = args.out or os.path.join(
        PHASE5_DIR, "samples", f"v12_{args.v12:.2f}_v21_{args.v21:.2f}"
    )
    os.makedirs(out_dir, exist_ok=True)
    for i in range(args.n):
        save_png(samples[i], os.path.join(out_dir, f"sample_{i:02d}.png"))

    print(f"Đã sinh {args.n} mẫu cho target v12={args.v12}, v21={args.v21}")
    print(f"Lưu tại: {out_dir}")
    print("Bước tiếp theo gợi ý: chạy verify FEA thật (Phase 6, bước 6.5) trên "
          "các mẫu này để kiểm tra cVAE có thật sự đạt Poisson ratio mong muốn, "
          "vì property loss lúc train chỉ dựa trên surrogate (không phải FE thật).")


if __name__ == "__main__":
    main()