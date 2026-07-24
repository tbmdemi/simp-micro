"""
Tests for pipeline/phase3_dataset/build_npz.py — load_and_resize(), the
box-filter downsample used to turn raw SIMP PNG renders into the 64x64
(or any RESOLUTION) density fields stored in dataset_{RES}.npz.
"""
import numpy as np
from PIL import Image

from pipeline.phase3_dataset.build_npz import load_and_resize


class TestLoadAndResize:
    def test_output_shape_matches_requested_resolution(self, tmp_path):
        img = Image.fromarray((np.random.default_rng(0).random((200, 200)) * 255).astype("uint8"))
        path = tmp_path / "sample.png"
        img.save(path)

        arr = load_and_resize(str(path), resolution=64)
        assert arr.shape == (64, 64)

    def test_output_is_normalized_to_zero_one(self, tmp_path):
        white = Image.fromarray(np.full((100, 100), 255, dtype="uint8"))
        black = Image.fromarray(np.zeros((100, 100), dtype="uint8"))
        white_path, black_path = tmp_path / "white.png", tmp_path / "black.png"
        white.save(white_path)
        black.save(black_path)

        white_arr = load_and_resize(str(white_path), resolution=32)
        black_arr = load_and_resize(str(black_path), resolution=32)

        assert np.allclose(white_arr, 1.0, atol=1e-3)
        assert np.allclose(black_arr, 0.0, atol=1e-3)

    def test_rgb_image_is_converted_to_grayscale(self, tmp_path):
        rgb = Image.fromarray(np.full((50, 50, 3), 128, dtype="uint8"), mode="RGB")
        path = tmp_path / "rgb.png"
        rgb.save(path)

        arr = load_and_resize(str(path), resolution=16)
        assert arr.shape == (16, 16)  # single channel, not (16, 16, 3)

    def test_downsampling_averages_rather_than_subsamples(self, tmp_path):
        """BOX resize should average a checkerboard toward mid-gray rather
        than aliasing to pure black/white — this is the whole reason the
        module docstring specifies Image.BOX over nearest/bilinear."""
        checkerboard = np.indices((64, 64)).sum(axis=0) % 2 * 255
        img = Image.fromarray(checkerboard.astype("uint8"))
        path = tmp_path / "checker.png"
        img.save(path)

        arr = load_and_resize(str(path), resolution=8)
        assert 0.3 < arr.mean() < 0.7
