from __future__ import annotations

import pytest

from kibitzer.hf_utils import parse_bool, validate_hf_push


@pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "on"])
def test_parse_bool_true(value: str) -> None:
    assert parse_bool(value)


@pytest.mark.parametrize("value", ["false", "FALSE", "0", "no", "off"])
def test_parse_bool_false(value: str) -> None:
    assert not parse_bool(value)


def test_hf_push_requires_repo_and_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="requires --hf-repo"):
        validate_hf_push(enabled=True, repo_id=None)
    with pytest.raises(SystemExit, match="requires HF_TOKEN"):
        validate_hf_push(enabled=True, repo_id="user/repo")
