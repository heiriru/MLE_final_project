# The Q-table must stay read-only, unseen states must not be added, and known states use the plain
# argmax.
import numpy as np

from agent_code.hybrid_agent.adapter.callbacks import ACTIONS, QAgentAdapter, load_frozen_q_table


def state(field, pos=(2, 2), bombs_left=True, bombs=(), others=()):
    return {"field": np.asarray(field), "self": ("me", 0, bombs_left, pos), "bombs": list(bombs),
            "others": list(others), "coins": [], "explosion_map": np.zeros((5, 5), dtype=int)}


def test_table_is_read_only_and_unseen_state_is_not_inserted():
    table = load_frozen_q_table()
    before = len(table)
    game = state(np.pad(np.zeros((3, 3), dtype=int), 1, constant_values=-1))
    adapter = QAgentAdapter({})
    decision = adapter.decide(game)
    assert not decision.known_state
    assert len(table) == before
    assert decision.action in ACTIONS
    try:
        table[decision.features] = np.zeros(6)
    except TypeError:
        pass
    else:
        raise AssertionError("frozen table unexpectedly accepted a write")


def test_known_state_uses_raw_q_table_argmax():
    field = -np.ones((5, 5), dtype=int)
    field[2, 2] = field[2, 3] = 0
    game = state(field)
    features = __import__("agent_code.hybrid_agent.adapter.callbacks", fromlist=["Q_AGENT"]).Q_AGENT.state_to_features(game)
    table = {features: np.array([999., 0., 0., 0., 1., 0.])}
    decision = QAgentAdapter(table).decide(game)
    assert decision.action == "UP"
