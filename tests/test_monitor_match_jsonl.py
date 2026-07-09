from __future__ import annotations

from pathlib import Path

from scripts.monitor_match_jsonl import format_summary, summarize_match


def test_summarize_match_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "match.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"result": "1-0", "score": 1.0}',
                '{"result": "1/2-1/2", "score": 0.5}',
                '{"result": "0-1", "score": 0.0}',
            ]
        )
    )

    summary = summarize_match(path)

    assert summary.games == 3
    assert summary.wins == 1
    assert summary.draws == 1
    assert summary.losses == 1
    assert summary.score == 1.5
    assert (
        format_summary(summary, 20)
        == "3/20 games: 1W/1D/1L score=1.5 rate=0.500 last=0-1 elo_delta=0 elo=2700"
    )


def test_missing_match_file_is_empty(tmp_path: Path) -> None:
    summary = summarize_match(tmp_path / "missing.jsonl")

    assert summary.games == 0
    assert format_summary(summary, 20) == "0/20 games: 0W/0D/0L score=0.0 rate=0.000 last=none elo_delta=-inf elo=-inf"
