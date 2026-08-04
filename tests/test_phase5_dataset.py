"""
Tests for pipeline/phase5_cvae/dataset.py - CVAEDataset.
"""
import numpy as np
import torch

from pipeline.phase5_cvae.dataset import (
    CVAEDataset, build_condition_vector, compute_v12_bin_weights,
    compute_v12_sample_weights,
)


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


class TestExtendedCondition:
    """extended_condition=True: condition grows từ 2 -> 6 chiều
    [v12,v21,volfrac,volfrac_mask,void_size_frac,void_size_frac_mask], mask
    LUÔN=1 ở tầng dataset (dữ liệu thật, có sẵn) - condition-dropout để mô
    phỏng "không chỉ định" là việc của train.py, không phải dataset.py."""

    def test_default_condition_dim_is_2(self, phase3_npz_path):
        ds = CVAEDataset(phase3_npz_path)
        assert ds.condition_dim == 2
        _, condition, _, _ = ds[0]
        assert condition.shape == (2,)

    def test_extended_condition_dim_is_6(self, phase3_npz_path):
        ds = CVAEDataset(phase3_npz_path, extended_condition=True)
        assert ds.condition_dim == 6
        _, condition, _, _ = ds[0]
        assert condition.shape == (6,)

    def test_extended_condition_values_and_masks(self, make_phase3_npz):
        path = make_phase3_npz("val.npz", n_samples=3)
        ds = CVAEDataset(path, extended_condition=True)
        _, condition, _, _ = ds[0]
        assert condition[0].item() == ds.v12[0]
        assert condition[1].item() == ds.v21[0]
        assert condition[2].item() == ds.volfrac_achieved[0]
        assert condition[3].item() == 1.0  # volfrac mask
        assert condition[4].item() == ds.void_size_frac[0]
        assert condition[5].item() == 1.0  # void_size_frac mask

    def test_extended_condition_dataloader_batching(self, phase3_npz_path):
        from torch.utils.data import DataLoader
        ds = CVAEDataset(phase3_npz_path, extended_condition=True)
        loader = DataLoader(ds, batch_size=5)
        _, condition, _, _ = next(iter(loader))
        assert condition.shape == (5, 6)
        assert torch.all(condition[:, 3] == 1.0)
        assert torch.all(condition[:, 5] == 1.0)


class TestBuildConditionVector:
    def test_condition_dim_2_ignores_optional_args(self):
        cond = build_condition_vector(-0.5, -0.3, condition_dim=2, volfrac=0.4)
        np.testing.assert_allclose(cond, [-0.5, -0.3])

    def test_condition_dim_6_unset_optional_gives_zero_mask(self):
        cond = build_condition_vector(-0.5, -0.3, condition_dim=6)
        np.testing.assert_allclose(cond, [-0.5, -0.3, 0.0, 0.0, 0.0, 0.0])

    def test_condition_dim_6_with_volfrac_only(self):
        cond = build_condition_vector(-0.5, -0.3, condition_dim=6, volfrac=0.35)
        np.testing.assert_allclose(cond, [-0.5, -0.3, 0.35, 1.0, 0.0, 0.0])

    def test_condition_dim_6_with_both_optional(self):
        cond = build_condition_vector(-0.5, -0.3, condition_dim=6,
                                       volfrac=0.35, void_size_frac=0.4)
        np.testing.assert_allclose(cond, [-0.5, -0.3, 0.35, 1.0, 0.4, 1.0])

    def test_invalid_condition_dim_raises(self):
        import pytest
        with pytest.raises(ValueError):
            build_condition_vector(-0.5, -0.3, condition_dim=4)


class TestV12BinWeights:
    def test_sparse_bin_gets_higher_weight_than_dense_bin(self):
        # bin [0,1) mật độ cao (900 mẫu), bin [1,2) thưa (10 mẫu).
        v12 = np.concatenate([np.full(900, 0.5), np.full(10, 1.5)])
        bin_edges = np.array([0.0, 1.0, 2.0])
        w = compute_v12_bin_weights(v12, bin_edges, alpha=0.5)
        assert w[1] > w[0]

    def test_alpha_zero_gives_uniform_weight(self):
        v12 = np.concatenate([np.full(900, 0.5), np.full(10, 1.5)])
        bin_edges = np.array([0.0, 1.0, 2.0])
        w = compute_v12_bin_weights(v12, bin_edges, alpha=0.0)
        np.testing.assert_allclose(w, [1.0, 1.0])

    def test_mean_weight_is_one(self):
        v12 = np.random.RandomState(0).uniform(-0.5, 0.3, size=500)
        bin_edges = np.arange(-0.5, 0.35, 0.05)
        counts, _ = np.histogram(v12, bins=bin_edges)
        w = compute_v12_bin_weights(v12, bin_edges, alpha=0.5)
        # mean cân theo count per-bin (không phải mean thô trên các bin) vì
        # per-sample weight thực tế được lặp lại count[i] lần cho bin i.
        weighted_mean = (w * counts).sum() / counts.sum()
        assert abs(weighted_mean - 1.0) < 0.05

    def test_sample_weights_shape_and_fallback_no_file(self, tmp_path):
        v12 = np.array([-0.6, -0.4, -0.4, -0.4, 0.1], dtype=np.float32)
        missing_path = str(tmp_path / "does_not_exist.json")
        weights = compute_v12_sample_weights(v12, weights_path=missing_path, alpha=0.5)
        assert weights.shape == v12.shape
        assert weights.dtype == np.float32
        assert (weights > 0).all()

    def test_sample_weights_from_saved_json(self, tmp_path):
        import json
        weights_path = tmp_path / "weights.json"
        weights_path.write_text(json.dumps({
            "bin_edges": [0.0, 1.0, 2.0],
            "bin_weight": [0.5, 2.0],
        }))
        v12 = np.array([0.2, 1.5], dtype=np.float32)
        weights = compute_v12_sample_weights(v12, weights_path=str(weights_path))
        assert weights[0] == 0.5
        assert weights[1] == 2.0
