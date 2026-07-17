"""
Phase 4 - export_for_phase5.py  (roadmap bước 4.5)
=====================================================
Đóng gói surrogate model đã train thành 1 file duy nhất, tự chứa đủ thông
tin để Phase 5 (cVAE) load và dùng làm property-consistency loss mà
KHÔNG cần import lại pipeline/phase4_surrogate/model.py logic thủ công.

Cách chạy:
    python3 pipeline/phase4_surrogate/export_for_phase5.py

Output: outputs/phase4/surrogate_for_phase5.pt
Cách Phase 5 load lại (ví dụ):

    ckpt = torch.load("outputs/phase4/surrogate_for_phase5.pt", weights_only=False)
    model = SurrogateCNN(n_seeds=ckpt["n_seeds"], channels=ckpt["channels"],
                          fc_hidden=ckpt["fc_hidden"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False   # frozen - không train tiếp trong Phase 5
"""
import os
import json
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHASE4_DIR = os.path.join(REPO_ROOT, "outputs", "phase4")


def main():
    src = os.path.join(PHASE4_DIR, "surrogate_best.pt")
    dst = os.path.join(PHASE4_DIR, "surrogate_for_phase5.pt")
    eval_report_path = os.path.join(PHASE4_DIR, "evaluation_report.json")

    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    eval_report = None
    if os.path.exists(eval_report_path):
        with open(eval_report_path) as f:
            eval_report = json.load(f)
    else:
        print("CẢNH BÁO: chưa tìm thấy evaluation_report.json — hãy chạy "
              "evaluate.py trước để ghi lại độ tin cậy của surrogate vào "
              "gói export này (Phase 5 cần biết R2 để đánh giá độ tin cậy).")

    export = {
        "model_state_dict": ckpt["model_state_dict"],
        "n_seeds": ckpt["n_seeds"],
        "seed_classes": ckpt["seed_classes"],
        "channels": ckpt["channels"],
        "fc_hidden": ckpt["fc_hidden"],
        "target_names": ckpt["target_names"],
        "input_spec": {
            "image_shape": [1, 64, 64],
            "image_range": [0.0, 1.0],
            "seed_vec": "one-hot theo thứ tự seed_classes",
        },
        "training_val_loss": ckpt["val_loss"],
        "training_epoch": ckpt["epoch"],
        "evaluation_report": eval_report,
        "usage_note": (
            "Model FROZEN - chỉ dùng để tính property-consistency loss trong "
            "Phase 5, không train tiếp. Chỉ tin cậy trong phân phối train: "
            "seed đã thấy (11 loại) và v12 trong khoảng đã sample "
            "(~[-0.81, 0.37]). Geometry cVAE sinh ra ngoài phân phối này thì "
            "surrogate loss không đáng tin, cần validate lại bằng FE thật."
        ),
    }

    torch.save(export, dst)
    print(f"Đã export: {dst}")
    print(f"  target_names: {export['target_names']}")
    print(f"  seed_classes: {export['seed_classes']}")
    if eval_report:
        print(f"  R2 (test set): "
              f"v12={eval_report['overall']['v12']['r2']:.4f}, "
              f"v21={eval_report['overall']['v21']['r2']:.4f}")


if __name__ == "__main__":
    main()