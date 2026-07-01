from __future__ import annotations

from kibitzer.model import Kibitzer, KibitzerConfig
from scripts.train_bc import configure_trainable_parameters


def test_policy_only_freezes_value_head() -> None:
    model = Kibitzer(
        KibitzerConfig(
            d_model=32,
            n_heads=4,
            max_seq_len=2,
            encoder_layers=1,
            encoder_heads=4,
            trunk_layers=1,
            attention_every=1,
            ssm_state_dim=4,
        )
    )

    trainable = configure_trainable_parameters(model, policy_only=True)
    trainable_ids = {id(parameter) for parameter in trainable}
    assert trainable_ids
    for name, parameter in model.named_parameters():
        if name.startswith("value_head."):
            assert not parameter.requires_grad
            assert id(parameter) not in trainable_ids
        else:
            assert parameter.requires_grad
            assert id(parameter) in trainable_ids
