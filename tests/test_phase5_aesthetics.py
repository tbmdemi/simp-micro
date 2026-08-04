"""
Tests for pipeline/phase5_cvae/aesthetics.py - symmetry_score, smoothness_score,
aesthetic_score. Cheap geometric proxy for the "aesthetic" axis in
best_of_n_eval.py's composite scoring (see docstring there) - deliberately
simple/no-model, so tests check clear, hand-verifiable extremes rather than
tolerances against a learned reference.
"""
import numpy as np

from pipeline.phase5_cvae.aesthetics import (
    aesthetic_score, smoothness_score, symmetry_score,
)


class TestSymmetryScore:
    def test_perfectly_symmetric_image_scores_one(self):
        img = np.zeros((8, 8))
        img[2:6, 2:6] = 1.0  # centered solid square - symmetric both axes
        assert symmetry_score(img) == 1.0

    def test_asymmetric_image_scores_lower_than_symmetric(self):
        sym = np.zeros((8, 8))
        sym[2:6, 2:6] = 1.0
        asym = np.zeros((8, 8))
        asym[0:2, 0:2] = 1.0  # corner block - not symmetric under either flip
        assert symmetry_score(asym) < symmetry_score(sym)

    def test_random_noise_scores_near_half(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 2, size=(64, 64)).astype(np.float64)
        # independent random 0/1 pixels: E[|a-b|] with a,b iid Bernoulli(0.5) = 0.5
        assert abs(symmetry_score(img) - 0.5) < 0.05

    def test_score_bounded_in_unit_interval(self):
        rng = np.random.default_rng(1)
        img = rng.random((16, 16))
        s = symmetry_score(img)
        assert 0.0 <= s <= 1.0


class TestSmoothnessScore:
    def test_uniform_solid_scores_one(self):
        img_bin = np.ones((8, 8))
        assert smoothness_score(img_bin) == 1.0

    def test_uniform_void_scores_one(self):
        img_bin = np.zeros((8, 8))
        assert smoothness_score(img_bin) == 1.0

    def test_checkerboard_scores_near_zero(self):
        img_bin = np.indices((8, 8)).sum(axis=0) % 2  # alternating 0/1
        assert smoothness_score(img_bin) < 0.05

    def test_thick_shape_smoother_than_checkerboard(self):
        solid_block = np.zeros((16, 16))
        solid_block[4:12, 4:12] = 1.0  # 1 thick square - low perimeter/area
        checkerboard = np.indices((16, 16)).sum(axis=0) % 2
        assert smoothness_score(solid_block) > smoothness_score(checkerboard)

    def test_score_bounded_in_unit_interval(self):
        rng = np.random.default_rng(2)
        img_bin = (rng.random((16, 16)) > 0.5).astype(np.float64)
        s = smoothness_score(img_bin)
        assert 0.0 <= s <= 1.0


class TestAestheticScore:
    def test_combines_symmetry_and_smoothness_equally(self):
        img = np.zeros((8, 8))
        img[2:6, 2:6] = 1.0  # symmetric AND smooth (thick block)
        img_bin = (img > 0.5).astype(np.float32)
        expected = 0.5 * symmetry_score(img) + 0.5 * smoothness_score(img_bin)
        assert aesthetic_score(img, img_bin) == expected

    def test_binarizes_internally_when_img_bin_not_given(self):
        img = np.zeros((8, 8))
        img[2:6, 2:6] = 0.9  # > 0.5 threshold
        score_explicit = aesthetic_score(img, (img > 0.5).astype(np.float32))
        score_implicit = aesthetic_score(img)
        assert score_explicit == score_implicit

    def test_score_bounded_in_unit_interval(self):
        rng = np.random.default_rng(3)
        img = rng.random((32, 32))
        s = aesthetic_score(img)
        assert 0.0 <= s <= 1.0
