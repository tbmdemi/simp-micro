"""
Shared fixtures for pipeline/phase4_surrogate and pipeline/phase5_cvae tests.
"""
import sys

import numpy as np
import pytest

# pipeline/phase4_surrogate/ and pipeline/phase5_cvae/ each define sibling
# modules with the SAME basenames (dataset.py, model.py, evaluate.py,
# train.py, ...) and import each other via `sys.path.insert(0, <own dir>)`
# + bare `from dataset import X` rather than package-relative imports (see
# losses.py's _import_surrogate_cnn() docstring for why: importlib is used
# there specifically to avoid this). Whichever phase's script happens to
# import its bare-named sibling FIRST in a test session wins the
# sys.modules["dataset"]/["model"]/... cache slot; the other phase's
# `from dataset import CVAEDataset` (etc.) then silently resolves to the
# wrong module and raises ImportError/AttributeError depending on test
# order. Reset the cache around every test so file/order never matters.
_BARE_MODULE_NAMES = (
    "dataset", "model", "evaluate", "train", "losses", "sample",
    "self_play", "adversarial_dataset", "verify_fe", "best_of_n_eval",
    "export_for_phase5",
)


@pytest.fixture(autouse=True)
def _isolate_pipeline_bare_imports():
    saved_path = list(sys.path)
    for name in _BARE_MODULE_NAMES:
        sys.modules.pop(name, None)
    yield
    for name in _BARE_MODULE_NAMES:
        sys.modules.pop(name, None)
    sys.path[:] = saved_path


SEED_CLASSES = np.array(
    ["circle", "square", "reentrant_bowtie", "hexagonal"], dtype=object
)


def _make_phase3_npz(path, n_samples=12, resolution=64, seed=0,
                      seed_classes=SEED_CLASSES, v12_range=(-0.8, 0.35)):
    """Write a synthetic .npz matching outputs/phase3/{train,val,test}.npz's
    schema (images/v12/v21/volfrac_achieved/seed_onehot/seed_classes), small
    enough to load instantly, so tests never depend on the real (gitignored,
    multi-hundred-MB) Phase 3 dataset."""
    rng = np.random.default_rng(seed)
    n_seeds = len(seed_classes)
    images = rng.random((n_samples, resolution, resolution)).astype(np.float32)
    v12 = rng.uniform(v12_range[0], v12_range[1], size=n_samples).astype(np.float32)
    v21 = rng.uniform(v12_range[0], v12_range[1], size=n_samples).astype(np.float32)
    volfrac = rng.uniform(0.2, 0.6, size=n_samples).astype(np.float32)
    seed_idx = rng.integers(0, n_seeds, size=n_samples)
    seed_onehot = np.zeros((n_samples, n_seeds), dtype=np.float32)
    seed_onehot[np.arange(n_samples), seed_idx] = 1.0

    np.savez(
        path,
        images=images,
        v12=v12,
        v21=v21,
        volfrac_achieved=volfrac,
        seed_onehot=seed_onehot,
        seed_classes=seed_classes,
    )
    return path


@pytest.fixture
def make_phase3_npz(tmp_path):
    """Factory fixture: make_phase3_npz(name="train.npz", n_samples=12, ...)
    -> str path to a freshly-written synthetic Phase-3-schema .npz file."""
    def _factory(name="data.npz", **kwargs):
        return str(_make_phase3_npz(tmp_path / name, **kwargs))
    return _factory


@pytest.fixture
def phase3_npz_path(make_phase3_npz):
    """A single ready-to-use synthetic Phase-3-schema .npz path."""
    return make_phase3_npz("data.npz", n_samples=12)
