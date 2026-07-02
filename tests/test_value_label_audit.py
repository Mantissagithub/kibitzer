from __future__ import annotations

import pytest

from scripts.audit_value_labels import (
    census,
    choose_indices,
    route_result,
    summarize_by_bin,
    summarize_pairs,
)


def test_audit_sampling_is_deterministic_and_unique() -> None:
    first = choose_indices(100, 20, 42)
    assert first == choose_indices(100, 20, 42)
    assert len(first) == len(set(first)) == 20


def test_audit_metrics_report_sign_disagreement_and_bins() -> None:
    cached = [0.02, 0.10, 0.30, -0.80]
    oracle = [0.03, -0.12, 0.40, 0.90]

    overall = summarize_pairs(cached, oracle)
    by_oracle_bin = summarize_by_bin(cached, oracle, bin_source="oracle")

    assert overall["mae"] == pytest.approx(0.5075)
    assert overall["sign_disagreement"] == pytest.approx(0.5)
    assert by_oracle_bin["won"]["sign_disagreement"] == 1.0
    assert census(cached) == {"quiet": 1, "edge": 1, "decisive": 1, "won": 1}


def test_audit_decision_thresholds_match_decision_log() -> None:
    assert route_result(0.15, 0.01) == "label_limited"
    assert route_result(0.09, 0.08) == "cached_labels_have_headroom"
    assert route_result(0.12, 0.10) == "borderline_head_only_probe"
