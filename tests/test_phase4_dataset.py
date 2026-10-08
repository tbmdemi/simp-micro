"""
Tests for pipeline/phase4_surrogate/dataset.py - AuxeticDataset.
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


class TestAuxeticDatasetNu0:
    """include_nu0=True (Giai đoạn A/A4, docs/archive/PROJECT_PLAN.md Nhóm 1) -
    nu là
    INPUT PHỤ, không phải target, nên không được lẫn vào tensor targets."""

    def test_default_getitem_has_3_elements(self, make_phase3_npz):
        """include_nu0=False (mặc định) phải giữ đúng hành vi cũ - 3 phần tử,
        kể cả khi npz có sẵn field 'nu' (không đọc trừ khi được yêu cầu)."""
        path = make_phase3_npz("train.npz", n_samples=5, nu_range=(0.2, 0.4))
        ds = AuxeticDataset(path)
        item = ds[0]
        assert len(item) == 3

    def test_include_nu0_adds_4th_element(self, make_phase3_npz):
        path = make_phase3_npz("train.npz", n_samples=5, nu_range=(0.2, 0.4))
        ds = AuxeticDataset(path, include_nu0=True)
        image, seed_vec, targets, nu0 = ds[0]
        assert image.shape == (1, 64, 64)
        assert targets.shape == (3,)  # nu KHÔNG lẫn vào targets
        assert nu0.dtype == torch.float32
        assert nu0.ndim == 0

    def test_nu0_value_matches_source_array(self, make_phase3_npz):
        path = make_phase3_npz("train.npz", n_samples=5, nu_range=(0.2, 0.4))
        ds = AuxeticDataset(path, include_nu0=True)
        for i in range(len(ds)):
            _, _, _, nu0 = ds[i]
            assert nu0.item() == ds.nu[i]

    def test_missing_nu_field_raises(self, phase3_npz_path):
        """npz không có 'nu' (schema cũ) + include_nu0=True phải báo lỗi rõ,
        không âm thầm giả định 1 giá trị mặc định nào."""
        import pytest
        with pytest.raises(KeyError):
            AuxeticDataset(phase3_npz_path, include_nu0=True)
