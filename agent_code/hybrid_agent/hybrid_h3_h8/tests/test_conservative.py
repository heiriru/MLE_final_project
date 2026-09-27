# Small wins keep the q-agent's action, only clear wins may override; 
# the correction never exceeds its limit.
import numpy as np
import torch

from agent_code.hybrid_agent.hybrid_h3_h8.agent import RESIDUAL_LIMIT, bounded_residual, conservative_action


def test_bounded_residual_never_exceeds_predeclared_limit():
    correction = bounded_residual(torch.tensor([-1000., 0., 1000.]))
    assert torch.all(correction.abs() <= RESIDUAL_LIMIT)
    assert correction[1] == 0


def test_small_hybrid_win_does_not_override_q_agent():

    action = conservative_action(torch.tensor([0., 0.10, -1., -1., -1., -1.]),
                                 "UP")
    assert action == "UP"


def test_large_candidate_advantage_overrides_q_agent():
    action = conservative_action(torch.tensor([0., 99., -1., -1., -1., -1.]), "UP")
    assert action == "RIGHT"
