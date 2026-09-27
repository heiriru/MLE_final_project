# H2 parity: a zero residual reproduces the q-agent, also for unseen states.
import numpy as np
import torch

from agent_code.hybrid_agent.adapter.callbacks import ACTIONS, QAgentAdapter
from agent_code.hybrid_agent.prior.callbacks import QAgentPrior
from agent_code.hybrid_agent.hybrid_h3_h8.agent import QAgentHybrid, hybrid_inputs
from agent_code.hybrid_agent.hybrid_h3_h8.agent import ResidualQNetwork


def game(field, pos=(2, 2)):
    return {"field": np.asarray(field), "self": ("me", 0, True, pos), "bombs": [], "others": [],
            "coins": [], "explosion_map": np.zeros((5, 5), dtype=int)}


def test_residual_head_is_exactly_zero_and_has_h2_shape():
    model = ResidualQNetwork()
    assert model.final_layer_is_zero()
    assert tuple(model(torch.zeros(3, 12)).shape) == (3, 6)
    assert torch.count_nonzero(model(torch.randn(4, 12))) == 0


def test_zero_residual_preserves_q_agent_on_known_and_unseen_states():
    field = np.pad(np.zeros((3, 3), dtype=int), 1, constant_values=-1)
    for adapter in (QAgentAdapter(), QAgentAdapter({})):
        baseline = adapter.decide(game(field))
        hybrid = QAgentHybrid(prior=QAgentPrior(adapter=adapter)).decide(game(field))
        assert hybrid.action == baseline.action
        assert np.allclose(hybrid.residual_q, 0)
        assert hybrid_inputs(baseline.features, hybrid.prior.advantages).shape == (12,)
