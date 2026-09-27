# The value head can take any value, the chosen action must not change; both heads start at zero.
from __future__ import annotations

import numpy as np
import torch

from agent_code.hybrid_agent.hybrid_h10.agent import conservative_action
from agent_code.hybrid_agent.hybrid_h10.agent import ACTIONS
from agent_code.hybrid_agent.hybrid_h10.agent import residual_total
from agent_code.hybrid_agent.hybrid_h10.agent import ResidualQNetwork


def test_both_heads_start_at_zero():
    assert ResidualQNetwork(6).final_layer_is_zero()


def test_untrained_total_is_exactly_the_q_agent_advantage():
    residual = ResidualQNetwork(6)
    inputs = torch.randn(32, 12)
    assert torch.allclose(residual_total(residual, inputs), inputs[..., 6:])


def test_value_head_never_changes_the_chosen_action():

    rng = np.random.default_rng(0)
    residual = ResidualQNetwork(6)
    torch.nn.init.normal_(residual.output.weight, std=0.5)
    torch.nn.init.normal_(residual.value.weight, std=2.0)
    torch.nn.init.normal_(residual.value.bias, std=2.0)
    for _ in range(200):
        inputs = torch.randn(1, 12)
        q_agent = ACTIONS[int(rng.integers(len(ACTIONS)))]
        with torch.no_grad():
            observation = inputs[:, :6]
            correction, value = residual.heads(observation)
            without = (inputs[0, 6:] + 0.25 * torch.tanh(correction[0]))
        assert conservative_action(without, q_agent) ==\
               conservative_action(without + float(value), q_agent)


def test_value_head_is_unbounded_where_the_correction_is_not():
    residual = ResidualQNetwork(6)
    torch.nn.init.constant_(residual.value.bias, 37.0)
    torch.nn.init.constant_(residual.output.bias, 50.0)
    with torch.no_grad():
        correction, value = residual.heads(torch.zeros(1, 6))
    assert float(value) == 37.0
    assert float(0.25 * torch.tanh(correction).abs().max()) <= 0.25
