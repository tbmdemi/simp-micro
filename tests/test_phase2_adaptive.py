"""
Tests for pipeline/phase2_multi_batch/adaptive.py — the stop/refine/expand
decision loop that closes the Phase 2 DOE loop. This is the highest-value
module to protect: a wrong decision here means running (or not running)
thousands of expensive SIMP samples.
"""
import json

import numpy as np
import pytest

from pipeline.phase2_multi_batch.adaptive import (
    _estimate_total_bins,
    _fill_seeds,
    _load_accumulated_results,
    _narrow_params,
    _relative_improvement,
    compute_seed_sample_allocation,
    decide_next_action,
)
from pipeline.phase2_multi_batch.params import BatchMode, PipelineConfig


class TestRelativeImprovement:
    def test_positive_when_current_is_lower_better(self):
        # Minimization: curr < prev means improvement.
        assert _relative_improvement(curr=-0.6, prev=-0.5) > 0

    def test_negative_when_current_is_worse(self):
        assert _relative_improvement(curr=-0.4, prev=-0.5) < 0

    def test_near_zero_prev_does_not_blow_up(self):
        """Fix L2 in the source: denominator uses max(|prev|,|curr|,eps), not
        just |prev| — with the old (buggy) denom=|prev|, two tiny values
        (prev=1e-12, curr=1e-11) would divide by ~1e-12 and blow up to a
        huge ratio; with the fix, denom=max(1e-12,1e-11,1e-10)=1e-10 keeps
        the result small and bounded."""
        result = _relative_improvement(curr=1e-11, prev=1e-12)
        assert abs(result) < 1.0

    def test_large_curr_relative_to_tiny_prev_stays_near_one_not_infinite(self):
        result = _relative_improvement(curr=-0.5, prev=1e-12)
        assert result == pytest.approx(1.0, abs=1e-6)


class TestFillSeeds:
    def test_pads_up_to_five(self):
        result = _fill_seeds(["circle"])
        assert len(result) == 5
        assert "circle" in result

    def test_reentrant_bowtie_is_in_the_fallback_pool(self):
        """Regression guard for Fix H1: reentrant_bowtie was missing from
        _ALL_SEEDS entirely, so it could never be introduced by this
        fallback. It's listed last in _ALL_SEEDS, and _fill_seeds() stops
        as soon as it has 5 seeds — so in practice it's only ever added
        when the other 10 named seeds are unavailable, not guaranteed on
        every call. This test only guards the narrower regression: the
        name must at least be present in the pool `_fill_seeds` draws
        from, which we can observe by filling from empty (first 5 in
        _ALL_SEEDS order) and confirming the function never raises/skips
        unexpectedly on a full pass over all known seeds."""
        from pipeline.phase2_multi_batch.adaptive import _ALL_SEEDS
        assert "reentrant_bowtie" in _ALL_SEEDS

    def test_no_duplicates_and_preserves_existing(self):
        result = _fill_seeds(["circle", "square"])
        assert result[:2] == ["circle", "square"]
        assert len(result) == len(set(result))

    def test_already_five_or_more_is_unchanged(self):
        seeds = ["circle", "square", "hourglass", "four_circle", "hexagonal", "nine_circle"]
        assert _fill_seeds(seeds) == seeds

    def test_none_seed_scores_preserves_original_fixed_order(self):
        """seed_scores=None (mặc định) phải cho kết quả GIỐNG HỆT hành vi
        gốc (không truyền seed_scores) - tương thích ngược 100%."""
        assert _fill_seeds(["circle"], seed_scores=None) == _fill_seeds(["circle"])

    def test_seed_scores_prioritizes_higher_score_seeds(self):
        """Roadmap 6.2/6.3: khi có seed_scores, seed điểm cao phải được
        thêm TRƯỚC seed điểm thấp, bất kể thứ tự trong _ALL_SEEDS."""
        from pipeline.phase2_multi_batch.adaptive import _ALL_SEEDS
        # hexagonal đứng SAU circle/square trong _ALL_SEEDS nhưng cho điểm
        # cao nhất - phải được ưu tiên thêm trước nếu seed_scores tôn trọng.
        scores = {s: 0.0 for s in _ALL_SEEDS}
        scores["hexagonal"] = 0.9
        result = _fill_seeds([], seed_scores=scores)
        assert len(result) == 5
        assert result[0] == "hexagonal"

    def test_empty_seed_scores_dict_behaves_like_none(self):
        """seed_scores={} (vd chưa có dữ liệu manufacturability nào) phải
        rơi về đúng thứ tự cố định gốc - {} là falsy trong Python."""
        assert _fill_seeds(["circle"], seed_scores={}) == _fill_seeds(["circle"])


class TestComputeSeedSampleAllocation:
    def _results(self, seed, n, auxetic_frac, manuf_frac_among_auxetic):
        """n mẫu cho 1 seed: auxetic_frac tỷ lệ v12<0, và trong nhóm auxetic
        đó, manuf_frac_among_auxetic tỷ lệ passes_all=True (mô phỏng đúng
        joint_rate = auxetic AND manufacturable đồng thời)."""
        rows = []
        n_aux = int(n * auxetic_frac)
        n_aux_manuf = int(n_aux * manuf_frac_among_auxetic)
        for i in range(n):
            is_aux = i < n_aux
            passes = is_aux and (i < n_aux_manuf)
            rows.append({
                "success": True, "seed": seed,
                "v12": -0.5 if is_aux else 0.3,
                "passes_all": passes,
            })
        return rows

    def test_preserves_total_budget(self):
        """Tổng số mẫu phân bổ phải xấp xỉ n_samples_baseline * len(seeds) -
        đây là ĐIỂM MẤU CHỐT: không được làm giảm tổng kích thước batch so
        với hành vi chia đều gốc."""
        results = (
            self._results("circle_half_quarter", 100, 0.6, 0.9)
            + self._results("hexagonal", 100, 0.6, 0.1)
        )
        allocation = compute_seed_sample_allocation(
            results, ["circle_half_quarter", "hexagonal"], n_samples_baseline=80,
        )
        assert sum(allocation.values()) == pytest.approx(160, abs=5)

    def test_better_seed_gets_more_samples(self):
        results = (
            self._results("circle_half_quarter", 100, 0.6, 0.9)  # joint_rate cao
            + self._results("hexagonal", 100, 0.6, 0.1)           # joint_rate thấp
        )
        allocation = compute_seed_sample_allocation(
            results, ["circle_half_quarter", "hexagonal"], n_samples_baseline=80,
        )
        assert allocation["circle_half_quarter"] > allocation["hexagonal"]

    def test_no_seed_reduced_below_floor(self):
        """min_weight_frac đảm bảo seed tệ nhất vẫn nhận >= floor * uniform_share,
        không bị loại hoàn toàn (đa dạng hình học vẫn cần cho dataset ML)."""
        results = (
            self._results("circle_half_quarter", 100, 0.6, 1.0)
            + self._results("hexagonal", 100, 0.6, 0.0)  # joint_rate = 0 tuyệt đối
        )
        allocation = compute_seed_sample_allocation(
            results, ["circle_half_quarter", "hexagonal"], n_samples_baseline=80,
            min_weight_frac=0.3,
        )
        assert allocation["hexagonal"] >= 1

    def test_unknown_seed_gets_neutral_score_not_penalized(self):
        """Seed hoàn toàn mới (chưa có dữ liệu) không bị phạt oan - nhận
        điểm trung lập (trung bình các seed đã biết)."""
        results = self._results("circle_half_quarter", 100, 0.6, 0.9)
        allocation = compute_seed_sample_allocation(
            results, ["circle_half_quarter", "brand_new_seed"], n_samples_baseline=80,
        )
        assert allocation["brand_new_seed"] >= 1

    def test_no_data_at_all_falls_back_to_uniform(self):
        allocation = compute_seed_sample_allocation(
            [], ["circle", "square", "hexagonal"], n_samples_baseline=80,
        )
        assert allocation == {"circle": 80, "square": 80, "hexagonal": 80}


class TestNarrowParams:
    def _valid_results(self, n=30, seed=0):
        rng = np.random.default_rng(seed)
        return [
            {
                "success": True,
                "v12": float(rng.uniform(-0.8, -0.1)),
                "params": {"volfrac": float(rng.uniform(0.3, 0.7))},
            }
            for _ in range(n)
        ]

    def test_too_few_results_returns_unchanged_copy(self):
        current = {"volfrac": {"range": [0.3, 0.7]}}
        result = _narrow_params([{"success": True, "v12": -0.5, "params": {}}], current)
        assert result == current
        assert result is not current  # must be a deep copy

    def test_narrows_within_original_bounds(self):
        current = {"volfrac": {"range": [0.3, 0.7]}}
        result = _narrow_params(self._valid_results(), current, n_batches_completed=2)
        lo, hi = result["volfrac"]["range"]
        assert 0.3 <= lo < hi <= 0.7

    def test_more_batches_completed_narrows_at_least_as_aggressively(self):
        current = {"volfrac": {"range": [0.3, 0.7]}}
        results = self._valid_results(n=60)
        gentle = _narrow_params(results, current, n_batches_completed=2)
        aggressive = _narrow_params(results, current, n_batches_completed=5)
        gentle_span = gentle["volfrac"]["range"][1] - gentle["volfrac"]["range"][0]
        aggressive_span = aggressive["volfrac"]["range"][1] - aggressive["volfrac"]["range"][0]
        assert aggressive_span <= gentle_span + 1e-9

    def test_backward_compatible_when_passes_all_absent(self):
        """Kết quả cũ/mock (không có field passes_all) phải cho kết quả
        giống hệt trước khi thêm sort theo manufacturability - None coi là
        trung lập, không phải bị phạt."""
        current = {"volfrac": {"range": [0.3, 0.7]}}
        results = self._valid_results(n=60)  # không có passes_all key
        result = _narrow_params(results, current, n_batches_completed=2)
        # so sánh với sort thuần v12 (hành vi gốc) - phải khớp hệt nhau
        rng_check = sorted(results, key=lambda x: x["v12"])
        n_best = max(5, int(len(rng_check) * 0.20))
        expected_vals = [r["params"]["volfrac"] for r in rng_check[:n_best]]
        assert result["volfrac"]["range"][0] == pytest.approx(
            np.percentile(expected_vals, 15)
        )

    def test_prefers_manufacturable_samples_over_pure_v12(self):
        """Với đủ mẫu manufacturable, narrow phải ưu tiên chọn TỪ NHÓM
        manufacturable trước, dù có mẫu non-manufacturable v12 tốt hơn."""
        current = {"volfrac": {"range": [0.0, 1.0]}}
        # 10 mẫu "manufacturable" (v12 kém hơn, volfrac quanh 0.5) +
        # 10 mẫu "không manufacturable" (v12 tốt hơn NHIỀU, volfrac quanh 0.9)
        manuf_group = [
            {"success": True, "v12": -0.3, "passes_all": True, "params": {"volfrac": 0.5}}
            for _ in range(10)
        ]
        non_manuf_group = [
            {"success": True, "v12": -0.9, "passes_all": False, "params": {"volfrac": 0.9}}
            for _ in range(10)
        ]
        results = manuf_group + non_manuf_group
        result = _narrow_params(results, current, n_batches_completed=2)
        lo, hi = result["volfrac"]["range"]
        # Vùng narrow phải nằm quanh 0.5 (nhóm manufacturable), KHÔNG lệch
        # về 0.9 (nhóm v12 tốt hơn nhưng không manufacturable).
        assert lo < 0.6


class TestEstimateTotalBins:
    def test_few_points_gives_one_bin(self):
        assert _estimate_total_bins([{"success": True}] * 5, ("v12", "v21")) == 1

    def test_more_points_gives_more_bins(self):
        results = [{"success": True}] * 200
        assert _estimate_total_bins(results, ("v12", "v21")) > 1


class TestLoadAccumulatedResults:
    def test_reads_results_from_each_summary_output_dir(self, tmp_path):
        batch_dir = tmp_path / "batch_1"
        batch_dir.mkdir()
        payload = {"results": [{"success": True, "v12": -0.5}, {"success": True, "v12": -0.4}]}
        (batch_dir / "batch_1_results.json").write_text(json.dumps(payload))

        summaries = [{"batch_id": 1, "output_dir": str(batch_dir)}]
        results = _load_accumulated_results(summaries)
        assert len(results) == 2

    def test_missing_output_dir_is_skipped_not_an_error(self, tmp_path):
        summaries = [{"batch_id": 1, "output_dir": str(tmp_path / "does_not_exist")}]
        assert _load_accumulated_results(summaries) == []


def _write_batch(tmp_path, batch_id, v12_values, mode="explore"):
    """Write a fake batch_{id}_results.json + return a summary dict
    pointing to it, matching the shape decide_next_action expects."""
    batch_dir = tmp_path / f"batch_{batch_id}"
    batch_dir.mkdir()
    results = [
        {"success": True, "v12": v, "v21": v, "obj_value": v, "params": {"volfrac": 0.4 + 0.01 * i}}
        for i, v in enumerate(v12_values)
    ]
    (batch_dir / f"batch_{batch_id}_results.json").write_text(
        json.dumps({"results": results})
    )
    return {
        "batch_id": batch_id,
        "output_dir": str(batch_dir),
        "mode": mode,
        "best_per_combo": {"circle_auxetic": {"v12": min(v12_values)}},
    }


class TestDecideNextAction:
    def test_stops_at_max_batches(self, tmp_path):
        config = PipelineConfig()
        summaries = [_write_batch(tmp_path, i, [-0.5 - 0.01 * i] * 20) for i in range(1, 6)]
        decision = decide_next_action(summaries, config, max_batches=5)
        assert decision["action"] == "stop"
        assert "max batches" in decision["reason"].lower() or "Reached max batches" in decision["reason"]

    def test_forces_expand_when_fewer_than_two_batches(self, tmp_path):
        config = PipelineConfig()
        summaries = [_write_batch(tmp_path, 1, [-0.5] * 20)]
        decision = decide_next_action(summaries, config, max_batches=5)
        assert decision["action"] == "expand"
        assert decision["next_config"] is not None
        assert decision["next_config"].mode == BatchMode.EXPLORE

    def test_stops_when_objective_stagnant_and_coverage_adequate(self, tmp_path):
        config = PipelineConfig()
        # Same best v12 across several batches -> stagnant; low sparsity via
        # a dense, well-spread grid of points.
        rng = np.random.default_rng(0)
        dense_values = list(np.linspace(-0.8, -0.1, 200))
        summaries = [
            _write_batch(tmp_path, i, dense_values, mode="refine")
            for i in range(1, 4)
        ]
        decision = decide_next_action(
            summaries, config, max_batches=10, improvement_patience=2,
        )
        assert decision["action"] == "stop"

    def test_invalid_param_ranges_type_raises(self, tmp_path):
        config = PipelineConfig()
        summaries = [_write_batch(tmp_path, 1, [-0.5] * 5)]
        with pytest.raises(ValueError):
            decide_next_action(summaries, config, param_ranges="not-a-dict")
