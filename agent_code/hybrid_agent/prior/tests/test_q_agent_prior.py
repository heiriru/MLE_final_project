# The normalisation may never change which action the q-agent picks.
import numpy as np

from agent_code.hybrid_agent.adapter.callbacks import ACTIONS, QAgentAdapter
from agent_code.hybrid_agent.prior.callbacks import QAgentPrior, normalized_q_advantage


def game(field, pos=(2, 2)):
    return {"field": np.asarray(field), "self": ("me", 0, True, pos), "bombs": [], "others": [],
            "coins": [], "explosion_map": np.zeros((5, 5), dtype=int)}


def test_normalized_advantage_centers_values_and_preserves_order():
    advantages = normalized_q_advantage(np.array([10., 1., 2., 3., 4., 5.]), 1.0)
    assert np.isclose(advantages.mean(), 0.0)
    assert int(np.argmax(advantages)) == 0
    assert np.allclose(np.diff(advantages[1:]), 1.0)


def test_prior_preserves_q_agent_fallback_for_unseen_state():
    field = np.pad(np.zeros((3, 3), dtype=int), 1, constant_values=-1)
    prior = QAgentPrior(adapter=QAgentAdapter({})).decide(game(field))
    assert not prior.q_agent.known_state
    assert prior.action in ACTIONS
    assert prior.advantages[ACTIONS.index(prior.action)] > 0


def test_prior_action_equals_adapter_action_on_real_table_state():
    field = np.pad(np.zeros((3, 3), dtype=int), 1, constant_values=-1)
    adapter = QAgentAdapter()
    assert QAgentPrior(adapter=adapter).decide(game(field)).action == adapter.decide(game(field)).action
