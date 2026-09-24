"""Tests for pipeline/phase5_cvae/build_latent_dataset.py."""

import numpy as np
import torch

from pipeline.phase5_cvae.build_latent_dataset import encode_dataset


class TestEncodeDataset:
    def test_shapes_and_keys(self, make_phase3_npz, make_cvae_checkpoint):
        npz_path = make_phase3_npz("val.npz", n_samples=9)
        ckpt_path = make_cvae_checkpoint(
            latent_dim=8, condition_dim=2, resolution=64
        )
        result = encode_dataset(ckpt_path, npz_path, device="cpu")
        assert set(result.keys()) == {
            "mu",
            "logvar",
            "condition",
            "v12",
            "v21",
        }
        assert result["mu"].shape == (9, 8)
        assert result["logvar"].shape == (9, 8)
        assert result["condition"].shape == (9, 2)
        assert result["v12"].shape == (9,)
        assert result["v21"].shape == (9,)
        assert isinstance(result["mu"], np.ndarray)

    def test_deterministic_across_calls(
        self, make_phase3_npz, make_cvae_checkpoint
    ):
        """Encoder trả về (mu, logvar) trực tiếp, không qua
        reparameterize() - gọi lại 2 lần trên cùng input phải cho cùng mu
        hệt nhau (không có nhiễu ngẫu nhiên nào lọt vào)."""
        npz_path = make_phase3_npz("val.npz", n_samples=5)
        ckpt_path = make_cvae_checkpoint(latent_dim=6, condition_dim=2)
        result1 = encode_dataset(ckpt_path, npz_path, device="cpu")
        result2 = encode_dataset(ckpt_path, npz_path, device="cpu")
        np.testing.assert_array_equal(result1["mu"], result2["mu"])

    def test_batching_preserves_sample_order(
        self, make_phase3_npz, make_cvae_checkpoint
    ):
        """batch_size nhỏ hơn n_samples (buộc DataLoader chia >=2 batch) -
        mu ghép lại phải khớp CHÍNH XÁC với gọi encoder trực tiếp từng mẫu
        một, không bị xáo trộn thứ tự khi nối các batch."""
        from pipeline.phase5_cvae.dataset import CVAEDataset
        from pipeline.phase5_cvae.model import CVAE

        npz_path = make_phase3_npz("val.npz", n_samples=7)
        ckpt_path = make_cvae_checkpoint(
            latent_dim=4, condition_dim=2, resolution=64, channels=(4, 8)
        )
        result = encode_dataset(
            ckpt_path, npz_path, device="cpu", batch_size=3
        )

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = CVAE(
            condition_dim=ckpt["condition_dim"],
            latent_dim=ckpt["latent_dim"],
            resolution=ckpt["resolution"],
            channels=ckpt["channels"],
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        ds = CVAEDataset(npz_path)
        with torch.no_grad():
            for i in range(len(ds)):
                image, condition, _seed_vec, _volfrac = ds[i]
                mu_i, _ = model.encoder(
                    image.unsqueeze(0), condition.unsqueeze(0)
                )
                np.testing.assert_allclose(
                    result["mu"][i], mu_i.squeeze(0).numpy(), atol=1e-6
                )
