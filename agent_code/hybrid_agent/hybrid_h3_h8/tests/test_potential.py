# The potential is 0 at the end of a round and grows towards coins and crates.
import numpy as np

from agent_code.hybrid_agent.hybrid_h3_h8.train import potential


def state(position=(2, 2), coins=((3, 2),), crate=False):
    field = np.pad(np.zeros((3, 3), dtype=int), 1, constant_values=-1)
    if crate:
        field[3, 2] = 1
    return {"field": field, "self": ("me", 0, True, position), "coins": list(coins)}


def test_terminal_potential_is_exactly_zero():
    assert potential(None) == 0.0


def test_potential_rewards_progress_toward_coin_and_crate_access():
    assert potential(state(position=(2, 2))) > potential(state(position=(1, 2)))
    assert potential(state(crate=True)) > potential(state(crate=False))
