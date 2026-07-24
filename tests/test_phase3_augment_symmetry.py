"""
Tests for pipeline/phase3_dataset/augment_symmetry.py — the physics-aware
symmetry augmentation that gives the train split its x6 multiplier
(README: "Train: 5.520 -> 33.120 mẫu"). Getting the v12<->v21 swap under
90/270-degree rotation wrong here would silently mislabel a third of the
augmented training set, so this is worth protecting even though the
function is simple.
"""
import numpy as np

from pipeline.phase3_dataset.augment_symmetry import augment_dataset, augment_sample


class TestAugmentSample:
    def test_returns_six_variants(self):
        image = np.random.default_rng(0).random((8, 8)).astype(np.float32)
        variants = augment_sample(image, v12=-0.4, v21=-0.6)
        assert len(variants) == 6

    def test_90_and_270_degree_rotations_swap_v12_and_v21(self):
        image = np.random.default_rng(0).random((8, 8)).astype(np.float32)
        variants = augment_sample(image, v12=-0.4, v21=-0.6)
        _, v12_r90, v21_r90 = variants[1]
        _, v12_r270, v21_r270 = variants[3]
        assert (v12_r90, v21_r90) == (-0.6, -0.4)
        assert (v12_r270, v21_r270) == (-0.6, -0.4)

    def test_180_rotation_and_flips_preserve_v12_v21(self):
        image = np.random.default_rng(0).random((8, 8)).astype(np.float32)
        variants = augment_sample(image, v12=-0.4, v21=-0.6)
        for idx in (0, 2, 4, 5):  # original, rot180, flip_h, flip_v
            _, v12_v, v21_v = variants[idx]
            assert (v12_v, v21_v) == (-0.4, -0.6)

    def test_rot90_twice_equals_rot180(self):
        image = np.random.default_rng(1).random((8, 8)).astype(np.float32)
        variants = augment_sample(image, v12=-0.4, v21=-0.6)
        img_r90 = variants[1][0]
        img_r180 = variants[2][0]
        assert np.allclose(np.rot90(img_r90, k=1), img_r180)

    def test_images_are_actually_distinct_transforms(self):
        # Guard against accidentally returning the same array 6 times.
        image = np.random.default_rng(2).random((8, 8)).astype(np.float32)
        variants = augment_sample(image, v12=-0.4, v21=-0.6)
        images = [v[0] for v in variants]
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                assert not np.array_equal(images[i], images[j]), (i, j)


class TestAugmentDataset:
    def test_multiplies_dataset_size_by_six(self):
        rng = np.random.default_rng(0)
        n = 5
        images = rng.random((n, 8, 8)).astype(np.float32)
        v12 = rng.uniform(-0.8, -0.1, size=n).astype(np.float32)
        v21 = rng.uniform(-0.8, -0.1, size=n).astype(np.float32)
        extra = {"seed_onehot": rng.random((n, 3)).astype(np.float32)}

        result = augment_dataset(images, v12, v21, extra)

        assert result["images"].shape == (n * 6, 8, 8)
        assert result["v12"].shape == (n * 6,)
        assert result["seed_onehot"].shape == (n * 6, 3)

    def test_max_variants_caps_multiplier(self):
        rng = np.random.default_rng(0)
        n = 4
        images = rng.random((n, 8, 8)).astype(np.float32)
        v12 = rng.uniform(-0.8, -0.1, size=n).astype(np.float32)
        v21 = rng.uniform(-0.8, -0.1, size=n).astype(np.float32)

        result = augment_dataset(images, v12, v21, extra_arrays={}, max_variants=2)
        assert result["images"].shape[0] == n * 2

    def test_extra_arrays_stay_aligned_with_source_sample(self):
        """Each of the 6 variants of sample i must carry sample i's extra
        metadata (e.g. seed_onehot), not some other sample's."""
        rng = np.random.default_rng(0)
        n = 3
        images = rng.random((n, 8, 8)).astype(np.float32)
        v12 = rng.uniform(-0.8, -0.1, size=n).astype(np.float32)
        v21 = rng.uniform(-0.8, -0.1, size=n).astype(np.float32)
        sample_ids = np.arange(n)
        extra = {"sample_id": sample_ids}

        result = augment_dataset(images, v12, v21, extra, max_variants=6)

        # First 6 rows must all belong to sample 0, next 6 to sample 1, etc.
        for i in range(n):
            block = result["sample_id"][i * 6:(i + 1) * 6]
            assert (block == i).all()
