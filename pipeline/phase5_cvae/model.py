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
-> flatten -> concat condition -> FC -> (mu, logvar).
Decoder: đối xứng ngược bằng ConvTranspose2d, [z, condition] -> FC ->
reshape -> upsample x2 x4 lần -> Sigmoid.
"""
import torch
import torch.nn as nn


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
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        ]
        if final:
            layers.append(nn.Sigmoid())
        else:
            layers += [nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, condition_dim=2, latent_dim=32,
                 channels=(32, 64, 128, 256), resolution=64):
        super().__init__()
        blocks = []
        in_ch = 1
        for out_ch in channels:
            blocks.append(EncoderBlock(in_ch, out_ch))
            in_ch = out_ch
        self.conv = nn.Sequential(*blocks)

        n_downs = len(channels)
        self.feat_res = resolution // (2 ** n_downs)   # 64 / 16 = 4
        self.feat_ch = channels[-1]
        flat_dim = self.feat_ch * self.feat_res * self.feat_res

        self.fc_mu = nn.Linear(flat_dim + condition_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim + condition_dim, latent_dim)

    def forward(self, image, condition):
        x = self.conv(image)                 # (B, C, feat_res, feat_res)
        x = x.flatten(1)                      # (B, C*feat_res*feat_res)
        x = torch.cat([x, condition], dim=1)  # (B, flat_dim + condition_dim)
        return self.fc_mu(x), self.fc_logvar(x)


class Decoder(nn.Module):
    def __init__(self, condition_dim=2, latent_dim=32,
                 channels=(256, 128, 64, 32), resolution=64):
        super().__init__()
        n_ups = len(channels)
        self.feat_res = resolution // (2 ** n_ups)     # 4
        self.feat_ch = channels[0]                     # 256

        self.fc = nn.Linear(
            latent_dim + condition_dim,
            self.feat_ch * self.feat_res * self.feat_res,
        )

        blocks = []
        in_ch = channels[0]
        for out_ch in channels[1:]:
            blocks.append(DecoderBlock(in_ch, out_ch))
            in_ch = out_ch
        blocks.append(DecoderBlock(in_ch, 1, final=True))  # -> (B,1,64,64) in [0,1]
        self.deconv = nn.Sequential(*blocks)

    def forward(self, z, condition):
        x = torch.cat([z, condition], dim=1)
        x = self.fc(x)
        x = x.view(-1, self.feat_ch, self.feat_res, self.feat_res)
        return self.deconv(x)                # (B, 1, 64, 64)


class CVAE(nn.Module):
    def __init__(self, condition_dim=2, latent_dim=32, resolution=64,
                 channels=(32, 64, 128, 256)):
        """channels: kênh encoder tăng dần (VD (32,64,128,256)); decoder tự
        dùng đảo ngược. train.py lưu channels vào checkpoint (sample.py đọc
        lại) nên đổi giá trị này sau khi đã có checkpoint cũ sẽ không load
        lại được state_dict cũ."""
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(condition_dim, latent_dim,
                                channels=tuple(channels), resolution=resolution)
        self.decoder = Decoder(condition_dim, latent_dim,
                                channels=tuple(reversed(channels)), resolution=resolution)

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

    def generate(self, condition, n_samples=1, device="cpu"):
        """Sinh geometry mới chỉ từ condition (dùng ở sample.py)."""
        z = torch.randn(n_samples, self.latent_dim, device=device)
        if condition.dim() == 1:
            condition = condition.unsqueeze(0).repeat(n_samples, 1)
        with torch.no_grad():
            return self.decoder(z, condition)


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