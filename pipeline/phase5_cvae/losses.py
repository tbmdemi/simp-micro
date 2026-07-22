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
    # theo batch - GIỮ NGUYÊN thang này vì cân bằng recon/kl hiện tại (kl
    # dừng ổn định ~66 ở beta=1, không collapse) đã hoạt động tốt trong lần
    # train thật đầu tiên - không nên đụng vào để tránh phá cân bằng đó.
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


def tv_loss(recon: torch.Tensor) -> torch.Tensor:
    """Total-variation: phạt biến thiên pixel đột ngột giữa các ô lân cận
    (ngang + dọc). Đây là vai trò tương đương density filter trong SIMP
    truyền thống (lọc trung bình lân cận để loại checkerboard) - ép decoder
    sinh vùng liên tục thay vì nhiễu hạt ngẫu nhiên. Thang giá trị: trung
    bình |chênh lệch pixel| trên toàn batch, nằm trong [0, 1]."""
    dh = torch.abs(recon[:, :, 1:, :] - recon[:, :, :-1, :]).mean()
    dw = torch.abs(recon[:, :, :, 1:] - recon[:, :, :, :-1]).mean()
    return dh + dw


def binarization_loss(recon: torch.Tensor) -> torch.Tensor:
    """Phạt pixel nằm giữa 0 và 1 (cực đại tại 0.5, = 0 tại 0 hoặc 1) - ép
    ảnh về gần nhị phân, đúng bản chất vật lý density field (0=rỗng,
    1=vật liệu), đồng thời gián tiếp chống nhiễu grayscale ngẫu nhiên vì
    nhiễu thường nằm ở vùng giá trị trung gian."""
    return (recon * (1 - recon)).mean()


# CHẠY THẬT ĐẦU TIÊN CHO THẤY: prop ~ O(0.01-0.05) trong khi recon ~ O(1000)
# - chênh nhau ~5 bậc độ lớn. Với gamma=1 mặc định (bản cũ), gamma*prop gần
# như KHÔNG có tiếng nói trong gradient tổng, nên lúc train `prop` giảm đều
# (trông có vẻ tốt) nhưng thực chất model gần như chỉ tối ưu recon; lúc
# evaluate.py sample z~N(0,1) thật sự (không "ăn gian" bằng z suy ra từ ảnh
# gốc như lúc train) thì property accuracy lộ ra rất tệ (R2 âm).
#
# Sửa: nhân prop với hệ số cố định PROP_LOSS_SCALE để đưa nó về gần thang
# recon/kl TRƯỚC khi nhân với gamma. Việc này giữ nguyên ý nghĩa của gamma
# (gamma=1 vẫn là "trọng số cơ sở hợp lý", không phải phải tự dò con số
# hàng trăm/nghìn) và KHÔNG đụng vào cân bằng recon/kl đã ổn định.
PROP_LOSS_SCALE = 1000.0


def cvae_loss(
    recon, image, mu, logvar, condition, seed_vec,
    surrogate, target_names, beta: float, gamma: float = 1.0,
    lambda_tv: float = 0.0, lambda_bin: float = 0.0,
):
    """Tổng hợp các thành phần. Trả về dict để log riêng từng loss trong train.py.
    gamma: hệ số trọng số CHO PHẦN NHÂN THÊM vào property-consistency loss
    (đã được PROP_LOSS_SCALE đưa về cùng thang recon/kl), mặc định 1.0.
    Tăng gamma nếu muốn ưu tiên đúng Poisson ratio hơn chất lượng ảnh tái
    tạo, giảm nếu thấy recon bị "hy sinh" quá nhiều (ảnh sinh ra mờ/nhoè).

    lambda_tv, lambda_bin: mặc định 0.0 (TẮT) để không phá baseline cũ
    (gamma=1..300 đã chạy trước khi có 2 loss này). Bật lên (ví dụ thử
    0.001-0.01 trước) khi nghi ngờ gamma cao đang khiến decoder sinh
    density field nhiễu/không hợp lý vật lý để "đánh lừa" surrogate -
    xem ghi chú trong docstring đầu file."""
    recon_l = reconstruction_loss(recon, image)
    kl_l = kl_divergence(mu, logvar)
    prop_l = property_consistency_loss(
        recon, condition, seed_vec, surrogate, target_names
    )
    tv_l = tv_loss(recon)
    bin_l = binarization_loss(recon)
    total = (recon_l + beta * kl_l + gamma * PROP_LOSS_SCALE * prop_l
             + lambda_tv * tv_l + lambda_bin * bin_l)
    return {
        "total": total,
        "recon": recon_l.detach(),
        "kl": kl_l.detach(),
        "prop": prop_l.detach(),                                  # thang gốc (Poisson-ratio MSE), để dễ hiểu ý nghĩa vật lý
        "prop_weighted": (gamma * PROP_LOSS_SCALE * prop_l).detach(),  # phần thật sự đóng góp vào total, để so sánh với recon/kl
        "tv": tv_l.detach(),
        "binarization": bin_l.detach(),
        "beta": beta,
    }