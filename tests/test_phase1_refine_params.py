"""
Tests for pipeline/phase1_screening/refine_params.py — the ACTIVE/FIXED
parameter decision logic that feeds Phase 2's sampling ranges. A wrong
decision here means Phase 2 either wastes samples on a parameter that
doesn't matter, or fixes one that actually does (see README: `volfrac`
found to dominate with r=0.87-0.96, `move`/`rmin`/`void_size_frac` not
significant).
"""
import json

import pytest

from pipeline.phase1_screening.refine_params import (
    P_THRESHOLD,
    build_refined_parameters,
    decide_active_params,
    main,
)


def _correlations(param_names, configs):
    return {"param_names": param_names, "configs": configs}


def _config(seed, objective, pvals):
    return {"seed": seed, "objective": objective, "pval": pvals}


class TestDecideActiveParams:
    def test_param_below_threshold_on_any_seed_is_active(self):
        correlations = _correlations(
            ["volfrac", "rmin"],
            [
                _config("circle", "auxetic", [1e-8, 0.5]),
                _config("square", "auxetic", [0.3, 0.6]),
            ],
        )
        decisions = decide_active_params(correlations)
        assert decisions["volfrac"]["active"] is True
        assert decisions["rmin"]["active"] is False

    def test_takes_minimum_pvalue_across_seeds(self):
        correlations = _correlations(
            ["penal"],
            [
                _config("circle", "auxetic", [0.2]),
                _config("reentrant_bowtie", "auxetic", [0.06]),
                _config("square", "auxetic", [0.5]),
            ],
        )
        decisions = decide_active_params(correlations)
        assert decisions["penal"]["min_pval"] == pytest.approx(0.06)
        assert decisions["penal"]["seed"] == "reentrant_bowtie"

    def test_borderline_between_005_and_threshold_gets_a_note(self):
        assert P_THRESHOLD == 0.10  # this test assumes the documented default
        correlations = _correlations(["x"], [_config("circle", "auxetic", [0.07])])
        decisions = decide_active_params(correlations)
        assert decisions["x"]["active"] is True
        assert "note" in decisions["x"]

    def test_clearly_significant_param_gets_no_borderline_note(self):
        correlations = _correlations(["volfrac"], [_config("circle", "auxetic", [1e-8])])
        decisions = decide_active_params(correlations)
        assert "note" not in decisions["volfrac"]


class TestBuildRefinedParameters:
    def test_active_params_keep_full_param_space_range(self):
        correlations = _correlations(["volfrac"], [_config("circle", "auxetic", [1e-8])])
        refined = build_refined_parameters(correlations)
        assert "volfrac" in refined["active_parameters"]
        assert refined["active_parameters"]["volfrac"]["range"] == [0.45, 0.70]
        assert "volfrac" not in refined["fixed_parameters"]

    def test_fixed_params_get_midpoint_of_param_space(self):
        correlations = _correlations(["rmin"], [_config("circle", "auxetic", [0.9])])
        refined = build_refined_parameters(correlations)
        assert refined["fixed_parameters"]["rmin"] == pytest.approx((1.0 + 2.5) / 2, abs=1e-4)
        assert "rmin" not in refined["active_parameters"]

    def test_collects_unique_sorted_seeds_and_objectives(self):
        correlations = _correlations(
            ["volfrac"],
            [
                _config("square", "auxetic", [1e-8]),
                _config("circle", "auxetic", [1e-8]),
                _config("circle", "auxetic", [1e-8]),  # duplicate
            ],
        )
        refined = build_refined_parameters(correlations)
        assert refined["active_seeds"] == ["circle", "square"]
        assert refined["active_objectives"] == ["auxetic"]


class TestMainEndToEnd:
    def test_writes_expected_json_structure(self, tmp_path):
        correlations = _correlations(
            ["volfrac", "rmin"],
            [_config("circle", "auxetic", [1e-8, 0.8])],
        )
        src = tmp_path / "_all_correlations.json"
        src.write_text(json.dumps(correlations))
        dst = tmp_path / "refined_parameters.json"

        main(correlations_path=str(src), output_path=str(dst))

        assert dst.exists()
        with open(dst) as f:
            data = json.load(f)
        assert set(data) >= {
            "fixed_parameters", "active_parameters", "active_seeds",
            "active_objectives", "_decision_log", "_meta",
        }

    def test_missing_correlations_file_raises_with_helpful_message(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Chạy Phase 1 aggregate"):
            main(correlations_path=str(tmp_path / "missing.json"),
                 output_path=str(tmp_path / "out.json"))
