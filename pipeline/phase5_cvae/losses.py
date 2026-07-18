"""
Phase 5 - losses.py
=====================
3 thành phần loss (bước 5.3):

1. Reconstruction: BCE, không phải MSE - vì density field gần nhị phân
   (0 = rỗng, 1 = vật liệu), BCE phạt đúng bản chất phân loại pixel hơn.

2. KL divergence: có HỆ SỐ BETA tăng dần theo epoch (KL annealing), KHÔNG
   cố định beta=1 như VAE gốc. Lý do: condition vector chỉ 2 chiều (v12,
   v21) - nếu bật KL full ngay từ đầu, decoder rất dễ "bỏ qua" latent z
   (posterior collapse) vì chỉ cần dựa vào condition cũng giảm được
   reconstruction loss kha khá. Bắt đầu beta=0, tăng tuyến tính tới
   beta_max qua warmup_epochs.

3. Property-consistency: đưa ẢNH LIÊN TỤC (không binarize) do decoder sinh
   ra thẳng vào surrogate ĐÃ ĐÓNG BĂNG (outputs/phase4/surrogate_for_phase5.pt)
   để dự đoán (v12, v21), so với target condition bằng MSE. Đây là loss
   BẮT BUỘC decoder phải sinh hình học thật sự đạt Poisson ratio mong
   muốn, không chỉ "trông giống" density field.

   LƯU Ý QUAN TRỌNG (ghi trong usage_note của export_for_phase5.py):
   surrogate chỉ đáng tin trong phân phối train (11 seed đã thấy, v12 in
   khoảng ~[-0.81, 0.37]). Nếu property_loss cho ra giá trị rất nhỏ nhưng
   ảnh sinh ra "lạ" (không giống geometry nào trong 11 seed), ĐỪNG tin
   tuyệt đối con số này - cần verify lại bằng FE thật ở Phase 6 (bước 6.5).
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SURROGATE_PATH = os.path.join(REPO_ROOT, "outputs", "phase4", "surrogate_for_phase5.pt")


def _import_surrogate_cnn():
    """Import SurrogateCNN từ pipeline/phase4_surrogate/model.py bằng
    importlib thay vì sys.path + `import model` - vì phase5_cvae CŨNG có
    file model.py trùng tên, nếu dùng sys.path/import thường sẽ bị đụng
    module đã cache trong sys.modules (import sai file). importlib.util
    load theo đường dẫn tuyệt đối nên tránh được xung đột này."""
    import importlib.util
    path = os.path.join(REPO_ROOT, "pipeline", "phase4_surrogate", "model.py")
    spec = importlib.util.spec_from_file_location("phase4_surrogate_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SurrogateCNN


def load_frozen_surrogate(device="cpu", path=SURROGATE_PATH):
    """Load surrogate đã export ở Phase 4 (export_for_phase5.py), đóng băng
    toàn bộ tham số."""
    SurrogateCNN = _import_surrogate_cnn()

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = SurrogateCNN(
        n_seeds=ckpt["n_seeds"],
        channels=ckpt["channels"],
        fc_hidden=ckpt["fc_hidden"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    target_names = ckpt["target_names"]
    return model, target_names


def kl_beta_schedule(epoch: int, warmup_epochs: int, beta_max: float = 1.0) -> float:
    """Tăng tuyến tính 0 -> beta_max trong warmup_epochs, giữ nguyên sau đó."""
    if warmup_epochs <= 0:
        return beta_max
    return min(beta_max, beta_max * epoch / warmup_epochs)


def reconstruction_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # recon, target: (B, 1, RES, RES) trong [0,1]. reduction="sum" rồi chia
    # theo batch (chuẩn VAE - so_sánh_được giữa các batch size khác nhau).
    return F.binary_cross_entropy(recon, target, reduction="sum") / recon.size(0)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl.mean()


def property_consistency_loss(
    recon: torch.Tensor,
    condition: torch.Tensor,
    seed_vec: torch.Tensor,
    surrogate: nn.Module,
    target_names,
) -> torch.Tensor:
    """So sánh (v12, v21) surrogate dự đoán trên ảnh sinh ra với condition
    target. seed_vec dùng seed one-hot THẬT của mẫu gốc (surrogate cần input
    này) - đây là xấp xỉ hợp lý cho baseline vì ta chưa có nhãn seed cho ảnh
    generate. Nếu sau này muốn tổng quát hơn (không phụ thuộc seed thật),
    có thể thử trung bình dự đoán qua nhiều seed_vec hoặc dùng seed phổ biến
    nhất - để ở TODO cho bản nâng cấp."""
    pred = surrogate(recon, seed_vec)  # (B, 3) = [v12, v21, volfrac]
    idx_v12 = target_names.index("v12")
    idx_v21 = target_names.index("v21")
    pred_cond = torch.stack([pred[:, idx_v12], pred[:, idx_v21]], dim=1)
    return F.mse_loss(pred_cond, condition)


def cvae_loss(
    recon, image, mu, logvar, condition, seed_vec,
    surrogate, target_names, beta: float, gamma: float = 1.0,
):
    """Tổng hợp 3 thành phần. Trả về dict để log riêng từng loss trong train.py.
    gamma: hệ số trọng số cho property-consistency loss (mặc định 1.0, tăng
    lên nếu muốn ưu tiên đúng Poisson ratio hơn là chất lượng ảnh tái tạo)."""
    recon_l = reconstruction_loss(recon, image)
    kl_l = kl_divergence(mu, logvar)
    prop_l = property_consistency_loss(
        recon, condition, seed_vec, surrogate, target_names
    )
    total = recon_l + beta * kl_l + gamma * prop_l
    return {
        "total": total,
        "recon": recon_l.detach(),
        "kl": kl_l.detach(),
        "prop": prop_l.detach(),
        "beta": beta,
    }