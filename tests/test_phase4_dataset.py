"""
Tests for pipeline/phase4_surrogate/dataset.py — AuxeticDataset.
"""
import numpy as np
import torch

from pipeline.phase4_surrogate.dataset import AuxeticDataset


class TestAuxeticDataset:
    def test_len(self, make_phase3_npz):
        path = make_phase3_npz("train.npz", n_samples=17)
        ds = AuxeticDataset(path)
        assert len(ds) == 17

    def test_n_seeds_property(self, make_phase3_npz):
        path = make_phase3_npz(
            "train.npz", seed_classes=np.array(["a", "b", "c"], dtype=object)
        )
        ds = AuxeticDataset(path)
        assert ds.n_seeds == 3

    def test_getitem_shapes_and_dtypes(self, phase3_npz_path):
        ds = AuxeticDataset(phase3_npz_path)
        image, seed_vec, targets = ds[0]
        assert image.shape == (1, 64, 64)
        assert image.dtype == torch.float32
        assert seed_vec.shape == (ds.n_seeds,)
        assert targets.shape == (3,)
        assert targets.dtype == torch.float32

    def test_targets_order_is_v12_v21_volfrac(self, make_phase3_npz):
        path = make_phase3_npz("train.npz", n_samples=1)
        ds = AuxeticDataset(path)
        _, _, targets = ds[0]
        assert targets[0].item() == ds.v12[0]
        assert targets[1].item() == ds.v21[0]
        assert targets[2].item() == ds.volfrac_achieved[0]

    def test_seed_vec_is_onehot(self, phase3_npz_path):
        ds = AuxeticDataset(phase3_npz_path)
        for i in range(len(ds)):
            _, seed_vec, _ = ds[i]
            assert torch.isclose(seed_vec.sum(), torch.tensor(1.0))
            assert seed_vec.max().item() == 1.0

    def test_dataloader_batching(self, phase3_npz_path):
        from torch.utils.data import DataLoader
        ds = AuxeticDataset(phase3_npz_path)
        loader = DataLoader(ds, batch_size=4)
        image, seed_vec, targets = next(iter(loader))
        assert image.shape[0] == 4
        assert image.shape[1:] == (1, 64, 64)
