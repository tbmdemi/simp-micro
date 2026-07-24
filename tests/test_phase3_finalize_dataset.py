"""
Tests for pipeline/phase3_dataset/finalize_dataset.py — `_seed_only_stratify`,
the fixed-bin stratification helper that fixes a data-leakage bug (percentile
bins computed over the whole dataset would leak test-set information into
the train split; see the function's own docstring).

Import is lazy inside each test: finalize_dataset.py does
`sys.path.insert(0, os.path.dirname(__file__))` + bare
`from augment_symmetry import augment_dataset` at import time. That name
isn't in tests/conftest.py's _BARE_MODULE_NAMES collision list (it's unique
to phase3_dataset/, not shared with phase4_surrogate/phase5_cvae), so this
is lower-risk than that landmine, but importing lazily costs nothing and
avoids the module living in sys.path for the rest of the session.
"""
import numpy as np


class TestSeedOnlyStratify:
    def test_combines_seed_and_v12_bin(self):
        from pipeline.phase3_dataset.finalize_dataset import _seed_only_stratify
        seed_names = np.array(["circle"] * 10 + ["square"] * 10)
        v12 = np.concatenate([
            np.full(10, -0.9),  # bin 0 (lowest)
            np.full(10, 0.4),   # bin 4 (highest)
        ])
        labels = _seed_only_stratify(seed_names, v12, n_bins=5)
        assert len(set(labels[:10])) == 1
        assert len(set(labels[10:])) == 1
        assert labels[0] != labels[10]

    def test_falls_back_to_seed_only_when_a_combined_class_is_singleton(self):
        """If seed+bin produces a class with only 1 member, train_test_split
        would raise — the function must fall back to seed-only labels."""
        from pipeline.phase3_dataset.finalize_dataset import _seed_only_stratify
        seed_names = np.array(["circle"] * 9 + ["square"])  # square has n=1
        v12 = np.random.default_rng(0).uniform(-0.9, 0.4, size=10)
        labels = _seed_only_stratify(seed_names, v12, n_bins=5)
        # Fallback means labels are exactly the seed names.
        assert list(labels) == list(seed_names.astype(str))

    def test_out_of_domain_v12_is_clipped_not_erroring(self):
        from pipeline.phase3_dataset.finalize_dataset import _seed_only_stratify
        seed_names = np.array(["circle"] * 6)
        v12 = np.array([-5.0, -1.0, 0.0, 0.5, 2.0, 10.0])  # way outside [-1, 0.5]
        labels = _seed_only_stratify(seed_names, v12, n_bins=5)
        assert len(labels) == 6  # must not raise

    def test_every_sample_gets_a_label(self):
        from pipeline.phase3_dataset.finalize_dataset import _seed_only_stratify
        rng = np.random.default_rng(0)
        n = 40
        seed_names = rng.choice(["circle", "square", "hexagonal"], size=n)
        v12 = rng.uniform(-0.9, 0.4, size=n)
        labels = _seed_only_stratify(seed_names, v12, n_bins=5)
        assert len(labels) == n
        assert not any(l is None for l in labels)
