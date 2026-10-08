"""Ví dụ minh họa refine liên tục vs refine nhận thức nhị phân hóa (Bài #1,
hình trước/sau - review mục 3).

Tái tạo ĐÚNG z single-shot của harness P1.1a/P1.1e-a (seed 123, IN100) cho
vài condition, chạy cả 2 kiểu refine và lưu ẢNH LIÊN TỤC của decoder (JSON
benchmark chỉ lưu ảnh nhị phân) - để thấy trực quan refine liên tục khớp
target trên vật liệu xám rồi mất khi nhị phân hóa.

Kiểm chứng tự động: ν verified của mẫu baseline phải khớp `baseline_v12`
trong p1_1a JSON (cùng z) - nếu lệch, script dừng (hình sẽ không còn là
các mẫu đã báo cáo trong bài).

Chạy từ gốc repo:
    python docs/paper1/scripts/refine_examples.py
Output: outputs/phase5/plan_v3/p1_6c_refine_examples.npz (+ .json tóm tắt)
"""

import json
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline", "phase5_cvae"))

from adversarial_dataset import load_cvae  # noqa: E402
from benchmark_physics_guided_refinement import (  # noqa: E402
    _verify_v12_v21,
)
from dataset import CVAEDataset  # noqa: E402
from tandem_lbfgs import tandem_inverse_design_lbfgs  # noqa: E402
from verify_fe import FE_PARAMS  # noqa: E402

PLAN_V3 = os.path.join(REPO_ROOT, "outputs", "phase5", "plan_v3")
CKPT = os.path.join(REPO_ROOT, "outputs", "phase5", "cvae_v2_finetuned.pt")
# Condition IN100 có khoảng lệch refine-liên-tục lớn nhưng refine nhận thức
# nhị phân hóa nhỏ (chọn từ p1_1a vs p1_1e_a), trải dải ν12 −0,28 → −0,54.
EXAMPLE_IDX = (8, 35, 82)
SEED, N_COND = 123, 100


def main():
    """Chạy 2 kiểu refine cho EXAMPLE_IDX, kiểm z khớp harness, lưu npz."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Thứ tự RNG giống hệt harness: manual_seed → load_cvae → randn mỗi
    # condition theo thứ tự. Run P1.1a/P1.1e-a (2026-09-25) chạy trên GPU nên z
    # rút từ bộ sinh CUDA - rút trên CPU cho z khác (đã kiểm: ν lệch).
    torch.manual_seed(SEED)
    model = load_cvae(CKPT, device)
    test_ds = CVAEDataset(
        os.path.join(REPO_ROOT, "outputs", "phase3", "test.npz")
    )
    idxs = np.random.default_rng(SEED).choice(
        len(test_ds), N_COND, replace=False
    )
    zs = [
        torch.randn(1, model.latent_dim, device=device) for _ in range(N_COND)
    ]

    with open(os.path.join(PLAN_V3, "p1_1a_v2_finetuned_in100.json")) as f:
        ref_rows = json.load(f)["per_condition"]

    arrays, summary = {}, []
    for k in EXAMPLE_IDX:
        cond = torch.tensor(test_ds[idxs[k]][1].numpy(), device=device)
        z0 = zs[k].to(device)
        with torch.no_grad():
            img0 = model.decoder(z0, cond.unsqueeze(0)).squeeze().cpu().numpy()
        base = _verify_v12_v21(img0, FE_PARAMS)
        expect = ref_rows[k]["baseline_v12"]
        if abs(base[0] - expect) > 1e-6:
            raise RuntimeError(
                f"condition {k}: baseline ν12={base[0]:.6f} ≠ JSON {expect:.6f}"
                " - z không tái tạo đúng harness, dừng."
            )
        row = {
            "index": k,
            "target": [float(cond[0]), float(cond[1])],
            "baseline": list(base),
        }
        arrays[f"c{k}_baseline"] = img0
        for name, kw in (
            ("continuous", dict(steps=30)),
            ("aware", dict(steps=32, projection_betas=[1.0, 4.0, 16.0, 64.0])),
        ):
            res = tandem_inverse_design_lbfgs(
                target_poisson=cond[:2],
                generator_model=model,
                surrogate_model=None,
                condition=cond,
                initial_z=z0,
                learning_rate=0.5,
                guidance_source="real_physics",
                fe_params=FE_PARAMS,
                **kw,
            )
            img = res["image"].squeeze().cpu().numpy()
            arrays[f"c{k}_{name}"] = img
            row[name] = {
                "verified": list(_verify_v12_v21(img, FE_PARAMS)),
                "final_objective": float(res["history"][-1]),
                "grey_fraction": float(((img > 0.05) & (img < 0.95)).mean()),
            }
        summary.append(row)
        print(json.dumps(row))

    out = os.path.join(PLAN_V3, "p1_6c_refine_examples")
    np.savez_compressed(out + ".npz", **arrays)
    with open(out + ".json", "w") as f:
        json.dump(summary, f, indent=1)
    print("saved", out + ".{npz,json}")


if __name__ == "__main__":
    main()
