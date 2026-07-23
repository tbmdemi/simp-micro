"""
Tests for pipeline/phase5_cvae/dataset.py — CVAEDataset.
"""
import torch

from pipeline.phase5_cvae.dataset import CVAEDataset


class TestCVAEDataset:
    def test_len(self, make_phase3_npz):
        path = make_phase3_npz("val.npz", n_samples=9)
        ds = CVAEDataset(path)
        assert len(ds) == 9

    def test_resolution_property(self, make_phase3_npz):
        path = make_phase3_npz("val.npz", n_samples=2, resolution=32)
        ds = CVAEDataset(path)
        assert ds.resolution == 32

    def test_getitem_shapes(self, phase3_npz_path):
        ds = CVAEDataset(phase3_npz_path)
        image, condition, seed_vec, volfrac = ds[0]
        assert image.shape == (1, 64, 64)
        assert condition.shape == (2,)
        assert seed_vec.shape == (ds.n_seeds,)
        assert volfrac.dim() == 0  # scalar tensor

    def test_condition_is_v12_v21(self, make_phase3_npz):
        path = make_phase3_npz("val.npz", n_samples=1)
        ds = CVAEDataset(path)
        _, condition, _, _ = ds[0]
        assert condition[0].item() == ds.v12[0]
        assert condition[1].item() == ds.v21[0]

    def test_dataloader_batching(self, phase3_npz_path):
        from torch.utils.data import DataLoader
        ds = CVAEDataset(phase3_npz_path)
        loader = DataLoader(ds, batch_size=5)
        image, condition, seed_vec, volfrac = next(iter(loader))
        assert image.shape[0] == 5
        assert condition.shape == (5, 2)
        assert isinstance(image, torch.Tensor)
