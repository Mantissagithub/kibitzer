from __future__ import annotations

import pytest

from scripts.download_lichess_elite import parse_months


def test_parse_months() -> None:
    assert parse_months("11,06,11,08") == [6, 8, 11]


@pytest.mark.parametrize("value", ["", "0", "13", "01,99"])
def test_parse_months_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        parse_months(value)
