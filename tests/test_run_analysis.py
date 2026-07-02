from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from kibitzer.run_analysis import (
    EvidenceLog,
    assert_validation_split,
    collect_joint_stage_epochs,
    collect_match_result,
    collect_oracle_bin_metrics,
    collect_search_strategies,
    collect_value_repair_epochs,
    collect_value_stage_epochs,
    load_checkpoint_config,
    load_checkpoint_eval_metrics,
    load_json,
    parse_markdown_epoch_table,
    parse_strategy_key,
)


SAMPLE_VALUE_TABLE = """
Some prose before.

| epoch | MSE ↓ | MAE ↓ | Pearson ↑ | sign accuracy ↑ | R² ↑ |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0654 | 0.1671 | 0.5065 | 65.08% | 0.2552 |
| 2 | 0.0645 | 0.1653 | 0.5153 | 64.93% | 0.2647 |
| 3 | 0.0638 | 0.1639 | 0.5225 | 65.60% | 0.2730 |
| 4 | **0.0637** | 0.1639 | **0.5235** | 66.28% | **0.2736** |
| 5 | 0.0640 | **0.1638** | 0.5207 | **66.58%** | 0.2709 |

Some prose after.
"""


def test_parse_markdown_epoch_table_reads_five_rows_and_strips_bold_and_percent() -> None:
    rows = parse_markdown_epoch_table(
        SAMPLE_VALUE_TABLE,
        header_marker="| epoch | MSE ↓ | MAE ↓ | Pearson ↑ | sign accuracy ↑ | R² ↑ |",
        columns=["mse", "mae", "pearson", "sign_accuracy", "r2"],
    )
    assert len(rows) == 5
    assert rows[0] == pytest.approx(
        {"epoch": 1.0, "mse": 0.0654, "mae": 0.1671, "pearson": 0.5065, "sign_accuracy": 0.6508, "r2": 0.2552}
    )
    # Bold-marked cells (epoch 4's MSE/Pearson/R2) must parse the same as plain cells.
    assert rows[3]["mse"] == pytest.approx(0.0637)
    assert rows[3]["sign_accuracy"] == pytest.approx(0.6628)


def test_parse_markdown_epoch_table_missing_marker_returns_empty_list() -> None:
    rows = parse_markdown_epoch_table(
        "no tables here", header_marker="| epoch | MSE |", columns=["mse"]
    )
    assert rows == []


def test_collect_value_stage_epochs_matches_decisions_md_d25() -> None:
    rows = collect_value_stage_epochs(SAMPLE_VALUE_TABLE)
    assert [row["epoch"] for row in rows] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert rows[4]["sign_accuracy"] == pytest.approx(0.6658)


SAMPLE_JOINT_TABLE = """
| epoch | policy CE ↓ | teacher top-1 ↑ | teacher coverage ↑ | value MSE ↓ | value sign ↑ | value R² ↑ |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.6014 | 30.71% | 73.66% | **0.0717** | **66.74%** | **0.2872** |
| 2 | 2.5829 | **30.75%** | **73.77%** | 0.0721 | 66.12% | 0.2833 |
"""


def test_collect_joint_stage_epochs_parses_policy_and_value_columns() -> None:
    rows = collect_joint_stage_epochs(SAMPLE_JOINT_TABLE)
    assert len(rows) == 2
    assert rows[0]["policy_cross_entropy"] == pytest.approx(2.6014)
    assert rows[0]["policy_top1_accuracy"] == pytest.approx(0.3071)
    assert rows[1]["value_sign_accuracy"] == pytest.approx(0.6612)


def _write_checkpoint(path: Path, *, eval_metrics: dict | None, extra: dict | None = None) -> None:
    payload: dict = {"model": {"weight": torch.zeros(2)}, "config": {"dim": 2}}
    if eval_metrics is not None:
        payload["eval_metrics"] = eval_metrics
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def test_load_checkpoint_eval_metrics_reads_real_metrics(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.pt"
    _write_checkpoint(path, eval_metrics={"mse": 0.07, "decisive_sign_accuracy": 0.66})

    metrics = load_checkpoint_eval_metrics(path)

    assert metrics == {"mse": 0.07, "decisive_sign_accuracy": 0.66}


def test_load_checkpoint_eval_metrics_missing_file_returns_none(tmp_path: Path) -> None:
    assert load_checkpoint_eval_metrics(tmp_path / "missing.pt") is None


def test_load_checkpoint_config_excludes_model_and_config_tensors(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.pt"
    _write_checkpoint(path, eval_metrics={"mse": 0.1}, extra={"training_objective": "value_only", "best_epoch": 1})

    config = load_checkpoint_config(path)

    assert config == {
        "eval_metrics": {"mse": 0.1},
        "training_objective": "value_only",
        "best_epoch": 1,
    }


def test_collect_value_repair_epochs_reads_real_per_epoch_checkpoints(tmp_path: Path) -> None:
    run_dir = tmp_path / "value_repair"
    for epoch, sign in ((1, 0.663), (2, 0.660), (3, 0.655)):
        _write_checkpoint(
            run_dir / f"value_repair_best_epoch_{epoch}.pt",
            eval_metrics={
                "mse": 0.073,
                "mae": 0.175,
                "pearson": 0.53,
                "sign_accuracy": sign,
                "r2": 0.27,
                "decisive_mae": 0.28,
                "decisive_sign_accuracy": sign,
                "won_mae": 0.5,
                "won_sign_accuracy": sign,
            },
        )

    rows = collect_value_repair_epochs(run_dir)

    assert [row["epoch"] for row in rows] == [1.0, 2.0, 3.0]
    assert rows[0]["decisive_sign_accuracy"] == pytest.approx(0.663)


def test_collect_value_repair_epochs_skips_missing_epochs(tmp_path: Path) -> None:
    run_dir = tmp_path / "value_repair"
    _write_checkpoint(run_dir / "value_repair_best_epoch_2.pt", eval_metrics={"mse": 0.07})

    rows = collect_value_repair_epochs(run_dir)

    assert [row["epoch"] for row in rows] == [2.0]


SAMPLE_DIAGNOSTICS = {
    "split": "validation",
    "checkpoints": {
        "phase2": {
            "value_by_bin": {
                "quiet": {"count": 200, "mae": 0.037, "sign_accuracy": 0.68},
                "won": {"count": 200, "mae": 0.66, "sign_accuracy": 0.74},
            }
        }
    },
    "strategies": {
        "phase2:raw": {
            "count": 800,
            "mean_cp": 235.9,
            "p90_cp": 667.0,
            "p95_cp": 1642.4,
            "near_best_accuracy": 0.56,
            "best_move_accuracy": 0.3225,
        },
        "phase2:s64:v0.5": {
            "count": 800,
            "mean_cp": 213.8,
            "p90_cp": 605.6,
            "p95_cp": 1233.6,
            "near_best_accuracy": 0.576,
            "best_move_accuracy": 0.336,
        },
    },
}


def test_assert_validation_split_accepts_validation() -> None:
    assert_validation_split(SAMPLE_DIAGNOSTICS, source="sample.json")


def test_assert_validation_split_rejects_test_split() -> None:
    payload = {**SAMPLE_DIAGNOSTICS, "split": "test"}
    with pytest.raises(ValueError, match="locked test"):
        assert_validation_split(payload, source="sample.json")


def test_collect_oracle_bin_metrics_returns_none_for_missing_checkpoint() -> None:
    assert collect_oracle_bin_metrics(SAMPLE_DIAGNOSTICS, checkpoint="joint") is None


def test_collect_oracle_bin_metrics_reads_bins() -> None:
    bins = collect_oracle_bin_metrics(SAMPLE_DIAGNOSTICS, checkpoint="phase2")
    assert bins["won"]["sign_accuracy"] == pytest.approx(0.74)
    assert bins["quiet"]["count"] == 200


def test_collect_search_strategies_reads_regret_and_accuracy() -> None:
    strategies = collect_search_strategies(SAMPLE_DIAGNOSTICS)
    assert set(strategies) == {"phase2:raw", "phase2:s64:v0.5"}
    assert strategies["phase2:s64:v0.5"]["mean_cp"] == pytest.approx(213.8)


@pytest.mark.parametrize(
    "key,expected",
    [
        ("phase2:raw", ("phase2", "raw", "n/a")),
        ("phase2:s64:v0.5", ("phase2", "s64", "v0.5")),
        ("value_repair:s64:v1", ("value_repair", "s64", "v1")),
    ],
)
def test_parse_strategy_key(key: str, expected: tuple[str, str, str]) -> None:
    assert parse_strategy_key(key) == expected


def test_collect_match_result_reads_json(tmp_path: Path) -> None:
    path = tmp_path / "match.json"
    path.write_text(
        json.dumps(
            {
                "checkpoint": "runs/joint_distill/joint_best.pt",
                "games": 10,
                "simulations": 64,
                "stockfish_elo": 1320,
                "wins": 2,
                "draws": 0,
                "losses": 8,
                "score": 0.2,
            }
        ),
        encoding="utf-8",
    )

    result = collect_match_result(path)

    assert result["score"] == pytest.approx(0.2)
    assert result["games"] == 10


def test_collect_match_result_missing_file_returns_none(tmp_path: Path) -> None:
    assert collect_match_result(tmp_path / "missing.json") is None


def test_load_json_missing_file_returns_none(tmp_path: Path) -> None:
    assert load_json(tmp_path / "missing.json") is None


def test_evidence_log_records_availability_and_notes() -> None:
    log = EvidenceLog()
    log.record("metric a", available=True, source="file.json")
    log.record("metric b", available=False, source="script.py", note="not logged")

    assert [entry.label for entry in log.entries] == ["metric a", "metric b"]
    assert log.entries[1].available is False
    assert log.entries[1].note == "not logged"


class TestPlotRunAnalysisEndToEnd:
    """End-to-end smoke test against the real repo evidence (fast, no GPU/training)."""

    def test_generator_runs_against_real_repo_and_produces_expected_outputs(self, tmp_path: Path) -> None:
        import scripts.plot_run_analysis as plot_run_analysis

        repo_root = Path(__file__).resolve().parents[1]
        output_dir = tmp_path / "run_analysis"

        argv = sys.argv
        sys.argv = [
            "plot_run_analysis.py",
            "--repo-root",
            str(repo_root),
            "--output-dir",
            str(output_dir),
        ]
        try:
            plot_run_analysis.main()
        finally:
            sys.argv = argv

        readme = output_dir / "README.md"
        assert readme.is_file()
        readme_text = readme.read_text(encoding="utf-8")
        assert "Locked test split" in readme_text

        # value + value-repair epoch curves and the common-oracle/search/match figures
        # all have real local evidence in this repo, so they must be generated.
        for name in (
            "fig1_value_metrics_by_epoch.png",
            "fig2_policy_metrics_by_epoch.png",
            "fig3_common_oracle_value_by_bin.png",
            "fig4_search_regret_by_checkpoint_sims_scale.png",
            "fig5_match_wdl_score_noisy.png",
        ):
            path = output_dir / name
            assert path.is_file(), f"expected {name} to be generated"
            assert path.stat().st_size > 0

    def test_generator_is_deterministic_across_runs(self, tmp_path: Path) -> None:
        import scripts.plot_run_analysis as plot_run_analysis

        repo_root = Path(__file__).resolve().parents[1]
        first_dir = tmp_path / "first"
        second_dir = tmp_path / "second"

        for output_dir in (first_dir, second_dir):
            plot_run_analysis.main.__wrapped__ = None
            import sys

            argv = sys.argv
            sys.argv = [
                "plot_run_analysis.py",
                "--repo-root",
                str(repo_root),
                "--output-dir",
                str(output_dir),
            ]
            try:
                plot_run_analysis.main()
            finally:
                sys.argv = argv

        first_readme = (first_dir / "README.md").read_text(encoding="utf-8")
        second_readme = (second_dir / "README.md").read_text(encoding="utf-8")
        assert first_readme == second_readme
