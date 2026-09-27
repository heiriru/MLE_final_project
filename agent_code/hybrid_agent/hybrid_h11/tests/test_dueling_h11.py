# Input layout [features | extras | advantages] and zero heads at the start.
from __future__ import annotations

import numpy as np
import torch

from agent_code.hybrid_agent.hybrid_h11.agent import conservative_action
from agent_code.hybrid_agent.hybrid_h11.agent import ACTIONS
from agent_code.hybrid_agent.hybrid_h11.agent import (ADVANTAGE_OFFSET, INPUT_DIMENSION,
                                                      OBSERVATION_DIMENSION, residual_total)
from agent_code.hybrid_agent.hybrid_h11.agent import ResidualQNetwork

DIM, WIDE = OBSERVATION_DIMENSION, INPUT_DIMENSION


def test_input_layout_is_the_documented_one():
    assert (OBSERVATION_DIMENSION, ADVANTAGE_OFFSET, INPUT_DIMENSION) == (14, 14, 20)


def test_both_heads_start_at_zero():
    assert ResidualQNetwork(DIM).final_layer_is_zero()


def test_untrained_total_is_exactly_the_q_agent_advantage():
    residual = ResidualQNetwork(DIM)
    inputs = torch.randn(32, WIDE)
    assert torch.allclose(residual_total(residual, inputs), inputs[..., ADVANTAGE_OFFSET:])


def test_value_head_never_changes_the_chosen_action():

    rng = np.random.default_rng(0)
    residual = ResidualQNetwork(DIM)
    torch.nn.init.normal_(residual.output.weight, std=0.5)
    torch.nn.init.normal_(residual.value.weight, std=2.0)
    torch.nn.init.normal_(residual.value.bias, std=2.0)
    for _ in range(200):
        inputs = torch.randn(1, WIDE)
        q_agent = ACTIONS[int(rng.integers(len(ACTIONS)))]
        with torch.no_grad():
            correction, value = residual.heads(inputs[:, :DIM])
            without = (inputs[0, ADVANTAGE_OFFSET:] + 0.25 * torch.tanh(correction[0]))
        assert conservative_action(without, q_agent) ==\
               conservative_action(without + float(value), q_agent)


def test_value_head_is_unbounded_where_the_correction_is_not():
    residual = ResidualQNetwork(DIM)
    torch.nn.init.constant_(residual.value.bias, 37.0)
    torch.nn.init.constant_(residual.output.bias, 50.0)
    with torch.no_grad():
        correction, value = residual.heads(torch.zeros(1, DIM))
    assert float(value) == 37.0
    assert float(0.25 * torch.tanh(correction).abs().max()) <= 0.25


def test_extras_reach_the_network_and_can_change_the_correction():

    residual = ResidualQNetwork(DIM)
    torch.nn.init.normal_(residual.output.weight, std=0.5)
    same_q_agent_key = torch.zeros(2, WIDE)
    same_q_agent_key[1, 6:ADVANTAGE_OFFSET] = 1.0
    with torch.no_grad():
        correction, _ = residual.heads(same_q_agent_key[:, :DIM])
    assert not torch.allclose(correction[0], correction[1])
