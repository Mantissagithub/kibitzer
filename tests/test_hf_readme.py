from scripts.hf_readme import render_hf_readme


def test_render_hf_readme_includes_pending_checkpoint_fields() -> None:
    text = render_hf_readme(
        repo_id="Pradheep1647/kibitzer-sft-elo-pending-step-002000",
        checkpoint_name="step_002000.pt",
        step=2000,
        config={"d_model": 384, "max_seq_len": 256},
        metrics={"ema_loss": 1.25},
        elo=None,
    )

    assert "library_name: pytorch" in text
    assert "`checkpoint`: `step_002000.pt`" in text
    assert "`step`: `002000`" in text
    assert "`elo_rating`: pending/unrated" in text
    assert "https://github.com/Mantissagithub/kibitzer" in text


def test_render_hf_readme_includes_rated_eval_fields() -> None:
    text = render_hf_readme(
        repo_id="Pradheep1647/kibitzer-sft-elo-1352-step-002000",
        checkpoint_name="step_002000.pt",
        step=2000,
        elo=32.4,
        opponent="stockfish-elo-1320",
        post_eval={"n_games": 20, "score": 11.5, "elo_err": 42.1},
    )

    assert "`elo_rating`: 1352 estimated vs stockfish-elo-1320 (+32.4 elo diff)" in text
    assert "`eval_opponent`: `stockfish-elo-1320`" in text
    assert "`elo_diff`: 32.4" in text
    assert "`eval_games`: 20" in text
    assert "`eval_score`: 11.50" in text
    assert "`elo_error`: 42.1" in text