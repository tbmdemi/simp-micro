"""
Phase 5 - model.py
====================
Conditional VAE (cVAE) cho inverse design: target (v12, v21) -> density field.

Condition vector = [v12, v21] (2 chiều), KHÔNG gồm seed one-hot - mục tiêu
inverse design là chỉ cần đưa target Poisson ratio, không cần biết trước
seed nào. seed_onehot vẫn được Dataset trả về (dùng ở evaluate.py để phân
tích latent space theo seed family), không đưa vào forward pass model.
Nếu cần nâng cấp: concat seed_vec vào condition ở cả encoder/decoder,
chỉ cần đổi `condition_dim` lúc khởi tạo CVAE, không cần sửa file này.

Encoder: 4 ConvBlock (32->64->128->256, giống SurrogateCNN Phase 4 nhưng
GIỮ feature map không gian, không GAP, để decoder có đủ thông tin tái tạo)
-> flatten -> concat condition -> EfficientKANLinear -> (mu, logvar).
Decoder: đối xứng ngược bằng ConvTranspose2d, [z, condition] ->
EfficientKANLinear -> reshape -> upsample x2 x4 lần -> Sigmoid.
decoder_type="wire" (Task 2 - WIRE INR): dùng WireContinuousDecoder - sinh
mật độ ρ ∈ [0,1] tại tọa độ liên tục (x,y) ∈ [-1,1]² qua kích hoạt Gabor
Wavelet phức, resolution-agnostic (generate(resolution=...) quét lưới mịn
hơn khi suy luận). Mặc định "conv" để tương thích ngược checkpoint cũ.
"""

import math
import os
import sys

import torch
import torch.nn as nn

# efficient_kan được vendor ở repo root (không có trên PyPI, xem
# efficient_kan/__init__.py). Khi chạy trực tiếp `python3
# pipeline/phase5_cvae/model.py` (hoặc train.py/sample.py/...), sys.path[0]
# là pipeline/phase5_cvae/ nên repo root không nằm trong sys.path -> thêm
# vào để `from efficient_kan import EfficientKANLinear` tìm thấy gói vendor.
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from efficient_kan import EfficientKANLinear  # noqa: E402


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, final=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=4, stride=2, padding=1
            )
        ]
        if final:
            layers.append(nn.Sigmoid())
        else:
            layers += [nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(
        self,
        condition_dim=2,
        latent_dim=32,
        channels=(32, 64, 128, 256),
        resolution=64,
        use_kan=True,
    ):
        super().__init__()
        blocks = []
        in_ch = 1
        for out_ch in channels:
            blocks.append(EncoderBlock(in_ch, out_ch))
            in_ch = out_ch
        self.conv = nn.Sequential(*blocks)

        n_downs = len(channels)
        self.feat_res = resolution // (2**n_downs)  # 64 / 16 = 4
        self.feat_ch = channels[-1]
        flat_dim = self.feat_ch * self.feat_res * self.feat_res

        fc_layer = EfficientKANLinear if use_kan else nn.Linear
        fc_kwargs = {"grid_size": 5, "spline_order": 3} if use_kan else {}
        self.fc_mu = fc_layer(
            flat_dim + condition_dim, latent_dim, **fc_kwargs
        )
        self.fc_logvar = fc_layer(
            flat_dim + condition_dim, latent_dim, **fc_kwargs
        )

    def forward(self, image, condition):
        x = self.conv(image)  # (B, C, feat_res, feat_res)
        x = x.flatten(1)  # (B, C*feat_res*feat_res)
        x = torch.cat([x, condition], dim=1)  # (B, flat_dim + condition_dim)
        return self.fc_mu(x), self.fc_logvar(x)


class Decoder(nn.Module):
    def __init__(
        self,
        condition_dim=2,
        latent_dim=32,
        channels=(256, 128, 64, 32),
        resolution=64,
        use_kan=True,
        enforce_symmetry=True,
    ):
        super().__init__()
        n_ups = len(channels)
        self.feat_res = resolution // (2**n_ups)  # 4
        self.feat_ch = channels[0]  # 256

        fc_layer = EfficientKANLinear if use_kan else nn.Linear
        fc_kwargs = {"grid_size": 5, "spline_order": 3} if use_kan else {}
        self.fc = fc_layer(
            latent_dim + condition_dim,
            self.feat_ch * self.feat_res * self.feat_res,
            **fc_kwargs,
        )
        self.enforce_symmetry = enforce_symmetry

        blocks = []
        in_ch = channels[0]
        for out_ch in channels[1:]:
            blocks.append(DecoderBlock(in_ch, out_ch))
            in_ch = out_ch
        blocks.append(
            DecoderBlock(in_ch, 1, final=True)
        )  # -> (B,1,64,64) in [0,1]
        self.deconv = nn.Sequential(*blocks)

    def forward(self, z, condition):
        x = torch.cat([z, condition], dim=1)
        x = self.fc(x)
        x = x.view(-1, self.feat_ch, self.feat_res, self.feat_res)
        image = self.deconv(x)  # (B, 1, 64, 64)
        if self.enforce_symmetry:
            image = 0.5 * (image + image.transpose(-1, -2))
        return image


class ComplexGaborActivation(nn.Module):
    """Lớp kích hoạt Gabor Wavelet phức (WIRE - Wavelet Implicit Neural
    Representations, Saragadam et al. ICCV 2023): dạng thực của sóng Gabor
    phức, cos(omega0 * p) * exp(-s0^2 * p^2) với p = Wx + b.

    Envelope Gaussian định xứ làm mỗi neuron chỉ phản hồi trong 1 vùng nhỏ
    của input (giúp biên cấu trúc sắc nét, triệt tiêu cấu trúc rác cô lập),
    còn cos tần số omega0 thêm dao động tần số cao. Khởi tạo weight theo
    skeleton kế hoạch Task 2: U(-sqrt(6/in)/s0, sqrt(6/in)/s0) - chia cho
    s0 để s0 * p giữ thang O(1) (đã kiểm chứng act.std ~0.37, không suy
    biến).
    """

    def __init__(self, in_features, out_features, omega0=10.0, s0=10.0):
        super().__init__()
        self.omega0 = omega0
        self.s0 = s0
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            bound = math.sqrt(6.0 / in_features) / s0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        projected = self.linear(x)
        return torch.cos(self.omega0 * projected) * torch.exp(
            -(self.s0**2) * (projected**2)
        )


class WireContinuousDecoder(nn.Module):
    """Decoder INR liên tục (Task 2 - WIRE INR Decoder): thay thế decoder
    ConvTranspose2d, nhận tọa độ liên tục (x, y) in [-1, 1]^2 gộp với latent
    z và condition c, xuất mật độ rho in [0, 1] tại ĐÚNG tọa độ được truy
    vấn - resolution-agnostic (quét lưới mịn hơn khi suy luận mà không cần
    train lại).

    forward(z, condition, resolution=None) GIỮ API (B,1,H,W) của Decoder
    conv (sinh lưới tọa độ nội bộ) để mọi downstream (losses.py,
    evaluate.py, sample.py, ...) không cần đổi. forward_coords(coords, z,
    cond) là API liên tục thuần (B,N,2) -> (B,N,1) dùng cho truy vấn tùy ý
    (vd tandem optimization Task 4).
    """

    def __init__(
        self,
        latent_dim=32,
        cond_dim=2,
        hidden_dim=128,
        resolution=64,
        omega0=10.0,
        s0=10.0,
        enforce_symmetry=True,
    ):
        """Args:
        latent_dim: số chiều latent z.
        cond_dim: số chiều condition vector.
        hidden_dim: số neuron mỗi lớp Gabor ẩn.
        resolution: độ phân giải lưới mặc định khi forward() không
            truyền resolution (train ở giá trị này).
        omega0, s0: tham số tần số/độ định xứ của ComplexGaborActivation.
        enforce_symmetry: nếu True, áp cùng ràng buộc đối xứng gương
            `0.5*(image+image.T)` mà Decoder (conv) đã có sẵn
            (model.py Decoder.forward) - trước đây WireContinuousDecoder
            thiếu hẳn prior hình học này so với Decoder conv, một phần
            nguyên nhân khiến ảnh sinh ra kém khả năng chế tạo hơn hẳn
            (xem EXPERIMENT_LOG.md 2026-08-24, frac_manufacturable thấp).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.resolution = resolution
        self.enforce_symmetry = enforce_symmetry
        input_dim = 2 + latent_dim + cond_dim  # coords (2) + z + condition
        self.layer1 = ComplexGaborActivation(
            input_dim, hidden_dim, omega0=omega0, s0=s0
        )
        self.layer2 = ComplexGaborActivation(
            hidden_dim, hidden_dim, omega0=omega0, s0=s0
        )
        self.output_layer = nn.Linear(hidden_dim, 1)

    def _coords_grid(self, resolution, device):
        """Lưới tọa độ (x, y) in [-1, 1]^2, thứ tự raster (hàng y cố định)
        để reshape về ảnh (1, H, W) đúng hướng."""
        ys = torch.linspace(-1.0, 1.0, resolution, device=device)
        xs = torch.linspace(-1.0, 1.0, resolution, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # (H*W, 2)

    def forward_coords(self, coords, z, cond):
        """Truy vấn mật độ tại tọa độ liên tục.

        Args:
            coords: (B, N, 2) tọa độ (x, y) in [-1, 1]^2.
            z: (B, latent_dim) latent vector.
            cond: (B, cond_dim) condition vector.

        Returns:
            (B, N, 1) mật độ rho in [0, 1] tại từng tọa độ.
        """
        z_exp = z.unsqueeze(1).expand(-1, coords.size(1), -1)
        cond_exp = cond.unsqueeze(1).expand(-1, coords.size(1), -1)
        x = torch.cat([coords, z_exp, cond_exp], dim=-1)
        x = self.layer1(x)
        x = self.layer2(x)
        return torch.sigmoid(self.output_layer(x))

    def forward(self, z, condition, resolution=None):
        """Sinh ảnh mật độ (B, 1, H, W) từ latent z + condition - API tương
        thích với Decoder conv. resolution mặc định = resolution khởi tạo.

        Nếu `enforce_symmetry=True`, áp cùng công thức đối xứng gương
        `0.5*(image+image.T)` mà Decoder (conv) dùng, sau khi reshape về
        lưới vuông - hợp lệ với mọi `resolution` truyền vào vì `_coords_grid`
        luôn sinh lưới H=W."""
        res = resolution or self.resolution
        coords = self._coords_grid(res, z.device)  # (N, 2)
        coords_b = coords.unsqueeze(0).expand(z.size(0), -1, -1)  # (B, N, 2)
        rho = self.forward_coords(coords_b, z, condition)  # (B, N, 1)
        image = rho.view(-1, 1, res, res)
        if self.enforce_symmetry:
            image = 0.5 * (image + image.transpose(-1, -2))
        return image


class CVAE(nn.Module):
    def __init__(
        self,
        condition_dim=2,
        latent_dim=32,
        resolution=64,
        channels=(32, 64, 128, 256),
        decoder_type="conv",
        wire_hidden_dim=128,
        wire_omega0=10.0,
        wire_s0=10.0,
        use_kan=True,
        enforce_symmetry=True,
    ):
        """channels: kênh encoder tăng dần (VD (32,64,128,256)); decoder tự
        dùng đảo ngược. train.py lưu channels vào checkpoint (sample.py đọc
        lại) nên đổi giá trị này sau khi đã có checkpoint cũ sẽ không load
        lại được state_dict cũ.

        decoder_type: "conv" (mặc định, tương thích ngược checkpoint cũ) hay
        "wire" (WireContinuousDecoder, Task 2 - WIRE INR, resolution-
        agnostic). wire_hidden_dim/wire_omega0/wire_s0 chỉ dùng khi
        decoder_type="wire". train.py lưu decoder_type vào checkpoint để
        sample.py/evaluate.py dựng đúng model khi load."""
        super().__init__()
        self.latent_dim = latent_dim
        self.decoder_type = decoder_type
        self.use_kan = use_kan
        self.enforce_symmetry = enforce_symmetry
        self.encoder = Encoder(
            condition_dim,
            latent_dim,
            channels=tuple(channels),
            resolution=resolution,
            use_kan=use_kan,
        )
        if decoder_type == "wire":
            self.decoder = WireContinuousDecoder(
                latent_dim=latent_dim,
                cond_dim=condition_dim,
                hidden_dim=wire_hidden_dim,
                resolution=resolution,
                omega0=wire_omega0,
                s0=wire_s0,
                enforce_symmetry=enforce_symmetry,
            )
        elif decoder_type == "conv":
            self.decoder = Decoder(
                condition_dim,
                latent_dim,
                channels=tuple(reversed(channels)),
                resolution=resolution,
                use_kan=use_kan,
                enforce_symmetry=enforce_symmetry,
            )
        else:
            raise ValueError(
                f"decoder_type={decoder_type!r} không hợp lệ - "
                "chỉ hỗ trợ 'conv' hoặc 'wire'"
            )

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, image, condition, deterministic: bool = False):
        """deterministic=True: z=mu (không sample) - dùng lúc validation để
        loại nhiễu ngẫu nhiên khỏi so sánh giữa các epoch. False (mặc định,
        lúc train): sample z qua reparameterization trick, chuẩn VAE."""
        mu, logvar = self.encoder(image, condition)
        z = mu if deterministic else self.reparameterize(mu, logvar)
        recon = self.decoder(z, condition)
        return recon, mu, logvar

    def generate(self, condition, n_samples=1, device="cpu", resolution=None):
        """Sinh geometry mới chỉ từ condition (dùng ở sample.py).

        resolution (chỉ tác dụng khi decoder_type="wire"): độ phân giải ảnh
        xuất ra - quét lưới tọa độ mịn hơn (128², 256², 512²) khi suy luận
        mà không cần train lại (resolution-agnostic). Decoder conv luôn xuất
        (B,1,64,64) và bỏ qua tham số này."""
        z = torch.randn(n_samples, self.latent_dim, device=device)
        if condition.dim() == 1:
            condition = condition.unsqueeze(0).repeat(n_samples, 1)
        with torch.no_grad():
            if self.decoder_type == "wire":
                return self.decoder(z, condition, resolution=resolution)
            return self.decoder(z, condition)


def cvae_kwargs_from_checkpoint(ckpt: dict) -> dict:
    """Dựng kwargs cho `CVAE(...)` từ 1 checkpoint đã lưu - nguồn suy luận
    kiến trúc DUY NHẤT cho mọi loader (sample.load_model,
    adversarial_dataset.load_cvae, verify_fe).

    Bug đã sửa 2026-09-25: load_cvae() (dùng bởi best_of_n_eval, self_play,
    benchmark refinement, coverage_eval, active_learning, ...) tự dựng CVAE
    mà không đọc `use_kan`/`enforce_symmetry` -> luôn rơi về mặc định
    True/True của CVAE.__init__: checkpoint KAN train TRƯỚC khi có symmetry
    (vd cvae_kan_realphysics_v2.pt) bị ép đối xứng lúc đánh giá, checkpoint
    Linear (vd cvae_realphysics.pt) không load được. Hai loader lệch nhau vì
    mỗi nơi tự chép danh sách field - gom về 1 chỗ để không lệch lần nữa.

    Mặc định cho checkpoint cũ thiếu field (tương thích ngược):
      - use_kan: suy từ state_dict (khóa `spline_weight` chỉ có ở
        EfficientKANLinear).
      - enforce_symmetry: False - checkpoint trước 2026-09-11 train không có
        ràng buộc này; mặc định True của CVAE.__init__ chỉ dành cho train mới.

    Args:
        ckpt: dict đã torch.load, có ít nhất `latent_dim` và
            `model_state_dict`.

    Returns:
        dict kwargs truyền thẳng vào CVAE(**kwargs).
    """
    state = ckpt["model_state_dict"]
    return {
        "condition_dim": ckpt.get("condition_dim", 2),
        "latent_dim": ckpt["latent_dim"],
        "resolution": ckpt.get("resolution", 64),
        "channels": ckpt.get("channels", (32, 64, 128, 256)),
        "decoder_type": ckpt.get("decoder_type", "conv"),
        "wire_hidden_dim": ckpt.get("wire_hidden_dim", 128),
        "wire_omega0": ckpt.get("wire_omega0", 10.0),
        "wire_s0": ckpt.get("wire_s0", 10.0),
        "use_kan": ckpt.get(
            "use_kan", any("spline_weight" in key for key in state)
        ),
        "enforce_symmetry": ckpt.get("enforce_symmetry", False),
    }


if __name__ == "__main__":
    # Self-test: `python3 pipeline/phase5_cvae/model.py`
    model = CVAE(condition_dim=2, latent_dim=32)
    dummy_img = torch.randn(4, 1, 64, 64)
    dummy_cond = torch.tensor([[-0.5, -0.5]] * 4, dtype=torch.float32)
    recon, mu, logvar = model(dummy_img, dummy_cond)
    print(f"recon: {recon.shape}, mu: {mu.shape}, logvar: {logvar.shape}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Số tham số: {n_params:,}")

    gen = model.generate(torch.tensor([-0.6, -0.6]), n_samples=3)
    print(f"generate() output: {gen.shape}  (kỳ vọng: [3, 1, 64, 64])")
