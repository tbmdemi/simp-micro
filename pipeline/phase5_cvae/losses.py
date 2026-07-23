"""
Phase 5 - losses.py
=====================
Tổng loss = BCE reconstruction + beta·KL (beta tăng tuyến tính theo epoch,
xem kl_beta_schedule) + gamma·PROP_LOSS_SCALE·property-consistency (MSE
giữa surrogate dự đoán trên ảnh sinh ra và condition target) + các
regularizer tuỳ chọn (tv_loss, binarization_loss).

CẢNH BÁO ĐÃ XÁC NHẬN (outputs/phase5/fe_verification_report.json, xem
verify_fe.py): R2 của property-consistency đo qua surrogate KHÔNG đáng tin
ở bất kỳ gamma nào (đã thử 1-300) - R2 đo bằng FE thật luôn âm nặng, và
khoảng cách surrogate-vs-FE càng doãng ra khi gamma càng tăng (decoder học
đánh lừa surrogate, không sinh hình học auxetic thật). Đừng tăng gamma kỳ
vọng cải thiện thật - xem mục Phase 5 trong README.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SURROGATE_PATH = os.path.join(REPO_ROOT, "outputs", "phase4", "surrogate_for_phase5.pt")


def _import_surrogate_cnn():
    """Load bằng importlib theo đường dẫn tuyệt đối, không phải sys.path +
    `import model`, vì phase5_cvae cũng có model.py trùng tên - tránh đụng
    module đã cache sai trong sys.modules."""
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


def load_frozen_surrogate_ensemble(paths, device="cpu"):
    """Load nhiều surrogate độc lập (vd huấn luyện với seed/khởi tạo khác
    nhau) để dùng property_consistency_loss_ensemble - đánh lừa đồng thời
    N mô hình độc lập khó hơn nhiều so với đánh lừa 1 mô hình đông cứng,
    đây là biện pháp khắc phục surrogate-exploitation mang tính cấu trúc,
    bổ sung cho self-play (xem README §5)."""
    models, target_names = [], None
    for p in paths:
        m, tn = load_frozen_surrogate(device=device, path=p)
        models.append(m)
        target_names = tn  # giống nhau giữa các surrogate (cùng target_names schema)
    return models, target_names


def kl_beta_schedule(epoch: int, warmup_epochs: int, beta_max: float = 1.0) -> float:
    """Tăng tuyến tính 0 -> beta_max trong warmup_epochs, giữ nguyên sau đó."""
    if warmup_epochs <= 0:
        return beta_max
    return min(beta_max, beta_max * epoch / warmup_epochs)


def reconstruction_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # sum-per-pixel rồi chia theo batch (không mean toàn bộ) - đổi thang này
    # sẽ đổi luôn cân bằng recon/kl, cần retune beta nếu sửa.
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
    """seed_vec dùng seed one-hot THẬT của mẫu gốc (surrogate cần input này)
    vì ảnh generate chưa có nhãn seed - xấp xỉ, TODO: thử trung bình qua
    nhiều seed_vec hoặc seed phổ biến nhất cho bản tổng quát hơn."""
    pred = surrogate(recon, seed_vec)  # (B, 3) = [v12, v21, volfrac]
    idx_v12 = target_names.index("v12")
    idx_v21 = target_names.index("v21")
    pred_cond = torch.stack([pred[:, idx_v12], pred[:, idx_v21]], dim=1)
    return F.mse_loss(pred_cond, condition)


def property_consistency_loss_ensemble(
    recon: torch.Tensor,
    condition: torch.Tensor,
    seed_vec: torch.Tensor,
    surrogates,
    target_names,
    lambda_disagreement: float = 0.0,
):
    """Bản ensemble của property_consistency_loss(): pred_cond = TRUNG BÌNH
    dự đoán qua N surrogate độc lập (khó bị decoder đánh lừa đồng loạt hơn
    1 surrogate), cộng thêm lambda_disagreement * phương sai giữa các
    surrogate - phạt decoder sinh ảnh rơi vào vùng các surrogate BẤT ĐỒNG
    (dấu hiệu ngoại suy/vùng mù, đúng chỗ exploitation hay xảy ra) ngay cả
    khi trung bình dự đoán trùng target. lambda_disagreement=0 => giống hệt
    hành vi mean-ensemble không phạt bất đồng."""
    idx_v12 = target_names.index("v12")
    idx_v21 = target_names.index("v21")
    preds_cond = []
    for surrogate in surrogates:
        pred = surrogate(recon, seed_vec)
        preds_cond.append(torch.stack([pred[:, idx_v12], pred[:, idx_v21]], dim=1))
    preds_stack = torch.stack(preds_cond, dim=0)  # (N_surrogate, B, 2)
    mean_pred = preds_stack.mean(dim=0)
    mse = F.mse_loss(mean_pred, condition)
    disagreement = preds_stack.var(dim=0, unbiased=False).mean()
    return mse + lambda_disagreement * disagreement, disagreement.detach()


def tv_loss(recon: torch.Tensor) -> torch.Tensor:
    """Phạt biến thiên pixel đột ngột (ngang+dọc) - vai trò tương đương
    density filter trong SIMP truyền thống, chống nhiễu checkerboard."""
    dh = torch.abs(recon[:, :, 1:, :] - recon[:, :, :-1, :]).mean()
    dw = torch.abs(recon[:, :, :, 1:] - recon[:, :, :, :-1]).mean()
    return dh + dw


def binarization_loss(recon: torch.Tensor) -> torch.Tensor:
    """Phạt pixel giữa 0-1 (cực đại tại 0.5) - ép ảnh về gần nhị phân đúng
    bản chất density field."""
    return (recon * (1 - recon)).mean()


# prop_loss ~ O(0.01-0.05) vs recon ~ O(1000), ~5 bậc độ lớn chênh lệch -
# không có scale này thì gamma*prop gần như vô hình trong gradient tổng.
PROP_LOSS_SCALE = 1000.0


def cvae_loss(
    recon, image, mu, logvar, condition, seed_vec,
    surrogate, target_names, beta: float, gamma: float = 1.0,
    lambda_tv: float = 0.0, lambda_bin: float = 0.0,
    lambda_disagreement: float = 0.0,
):
    """Tổng hợp các thành phần, trả dict để log riêng từng loss trong train.py.
    lambda_tv/lambda_bin mặc định 0.0 (tắt, để không phá baseline gamma=1..300
    đã chạy trước khi có 2 loss này) - bật lên (thử 0.001-0.01 trước) để thử
    giảm exploitation ở gamma cao, xem docstring đầu file.

    `surrogate` có thể là 1 model (như cũ, backward-compatible) HOẶC 1
    list/tuple các model (ensemble - xem property_consistency_loss_ensemble,
    load_frozen_surrogate_ensemble) - dùng trung bình dự đoán qua ensemble
    thay vì 1 surrogate đông cứng, cộng thêm lambda_disagreement * phương
    sai giữa các surrogate để phạt decoder sinh ảnh vào vùng các surrogate
    bất đồng (biện pháp cấu trúc chống exploitation, bổ sung cho self-play)."""
    recon_l = reconstruction_loss(recon, image)
    kl_l = kl_divergence(mu, logvar)
    is_ensemble = isinstance(surrogate, (list, tuple))
    if is_ensemble:
        prop_l, disagreement_l = property_consistency_loss_ensemble(
            recon, condition, seed_vec, surrogate, target_names, lambda_disagreement
        )
    else:
        prop_l = property_consistency_loss(
            recon, condition, seed_vec, surrogate, target_names
        )
        disagreement_l = torch.tensor(0.0)
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
        "disagreement": disagreement_l.detach() if torch.is_tensor(disagreement_l) else disagreement_l,
        "beta": beta,
    }