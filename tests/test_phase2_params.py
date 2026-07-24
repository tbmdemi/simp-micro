"""
Tests for pipeline/phase2_multi_batch/params.py — BatchConfig/PipelineConfig
dataclasses, JSON round-trip, and the Phase-1-summary merging helper.

No bare sys.path imports here (plain `from pipeline.phase2_multi_batch...`
throughout the module), so top-level imports are safe — unlike
phase4_surrogate/phase5_cvae's bare-import landmine (see tests/conftest.py).
"""
import json

from pipeline.phase2_multi_batch.params import (
    BatchConfig,
    BatchMode,
    PipelineConfig,
    SamplingStrategy,
    default_config,
    load_phase1_params,
    prepare_output,
)


class TestBatchConfig:
    def test_defaults(self):
        b = BatchConfig(batch_id=1)
        assert b.n_samples == 120
        assert b.strategy == SamplingStrategy.SOBOL
        assert b.mode == BatchMode.EXPLORE
        assert b.objectives == ["auxetic"]
        assert len(b.seeds) == 11

    def test_get_output_dir_default_derives_from_batch_id(self):
        b = BatchConfig(batch_id=2)
        assert b.get_output_dir("outputs/pipeline") == "outputs/pipeline/phase3"

    def test_get_output_dir_override_wins(self):
        b = BatchConfig(batch_id=2, output_dir="custom/dir")
        assert b.get_output_dir("outputs/pipeline") == "custom/dir"


class TestPipelineConfigRoundTrip:
    def test_save_load_preserves_batches_and_enums(self, tmp_path):
        config = PipelineConfig()
        config.add_batch(BatchConfig(
            batch_id=1, n_samples=50, strategy=SamplingStrategy.LHS,
            mode=BatchMode.REFINE,
        ))
        path = tmp_path / "pipeline_config.json"
        config.save(str(path))

        loaded = PipelineConfig.load(str(path))
        assert len(loaded.batches) == 1
        assert loaded.batches[0].batch_id == 1
        assert loaded.batches[0].n_samples == 50
        # Enums must round-trip as enum instances, not raw strings.
        assert loaded.batches[0].strategy == SamplingStrategy.LHS
        assert loaded.batches[0].mode == BatchMode.REFINE

    def test_save_produces_valid_json_with_expected_top_level_keys(self, tmp_path):
        config = default_config()
        path = tmp_path / "config.json"
        config.save(str(path))

        with open(path) as f:
            data = json.load(f)
        assert set(data) >= {"fixed", "active", "batches", "base_output_dir"}
        assert len(data["batches"]) == 2


class TestLoadPhase1Params:
    def test_keeps_widest_range_across_summaries(self):
        summaries = [
            {"parameters": {"volfrac": {"range": [0.3, 0.6]}}},
            {"parameters": {"volfrac": {"range": [0.2, 0.5]}}},
        ]
        result = load_phase1_params(summaries)
        assert result["volfrac"] == (0.2, 0.6)

    def test_ignores_malformed_entries(self):
        summaries = [
            {"parameters": {"volfrac": {"range": [0.3, 0.6]}}},
            {"parameters": "not-a-dict"},
            {"parameters": {"rmin": {"no_range_key": True}}},
            {"no_parameters_key": True},
        ]
        result = load_phase1_params(summaries)
        assert result == {"volfrac": (0.3, 0.6)}
        assert "rmin" not in result


class TestDefaultConfig:
    def test_has_two_batches_sobol_then_lhs(self):
        config = default_config()
        assert len(config.batches) == 2
        assert config.batches[0].strategy == SamplingStrategy.SOBOL
        assert config.batches[1].strategy == SamplingStrategy.LHS


class TestPrepareOutput:
    def test_creates_directory(self, tmp_path):
        out = prepare_output(str(tmp_path), batch_id=3, suffix="_results")
        assert out == str(tmp_path / "batch_3_results")
        assert (tmp_path / "batch_3_results").is_dir()
