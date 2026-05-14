from dataclasses import dataclass

from scripts.eval_and_rename_hf import _pending_repo_ids


@dataclass
class _ModelInfo:
    id: str


class _FakeApi:
    def list_models(self, *, author: str, search: str, token: str):
        assert author == "Pradheep1647"
        assert search == "kibitzer-sft-elo-pending-step-"
        assert token == "token"
        return [
            _ModelInfo("Pradheep1647/kibitzer-sft-elo-minus-0241-step-004000"),
            _ModelInfo("Pradheep1647/kibitzer-sft-elo-plus-0035-step-024000"),
            _ModelInfo("Pradheep1647/kibitzer-sft-elo-1244-step-026000"),
            _ModelInfo("Pradheep1647/kibitzer-sft-elo-pending-step-030000"),
            _ModelInfo("Pradheep1647/kibitzer-sft-elo-pending-step-028000"),
            _ModelInfo("OtherUser/kibitzer-sft-elo-pending-step-026000"),
            _ModelInfo("Pradheep1647/kibitzer-sft-elo-pending-step-notnum"),
        ]


def test_pending_repo_ids_only_returns_pending_repos() -> None:
    assert _pending_repo_ids(
        _FakeApi(),
        username="Pradheep1647",
        prefix="kibitzer-sft",
        token="token",
    ) == [
        "Pradheep1647/kibitzer-sft-elo-pending-step-028000",
        "Pradheep1647/kibitzer-sft-elo-pending-step-030000",
    ]
