# The eight inputs stay in [0, 1] and behave sensibly (no bomb -> no urgency, closer -> larger).
from __future__ import annotations

import numpy as np
import pytest

import settings
from agent_code.hybrid_agent.hybrid_h11.agent import Q_AGENT
from agent_code.hybrid_agent.hybrid_h11.agent import EXTRA_DIMENSION, extra_features, walk_distances


def empty_board() -> np.ndarray:

    field = np.zeros((settings.COLS, settings.ROWS), dtype=int)
    field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = -1
    for x in range(2, settings.COLS - 1, 2):
        for y in range(2, settings.ROWS - 1, 2):
            field[x, y] = -1
    return field


def state(bombs=(), coins=(), others=(), position=(1, 1), step=0, field=None) -> dict:
    return {"field": empty_board() if field is None else field,
            "self": ("me", 0, True, position), "others": list(others),
            "bombs": list(bombs), "coins": list(coins), "step": step,
            "explosion_map": np.zeros((settings.COLS, settings.ROWS)), "round": 1}


def test_shape_and_bounds():
    vector = extra_features(state(bombs=[((1, 1), 3)], coins=[(5, 1)], others=[("o", 0, True, (7, 1))], step=200))
    assert vector.shape == (EXTRA_DIMENSION,)
    assert vector.dtype == np.float32
    assert np.all(vector >= 0.0) and np.all(vector <= 1.0)


def test_no_bombs_means_no_urgency():
    assert np.all(extra_features(state())[:5] == 0.0)


@pytest.mark.parametrize("timer", [0, 1, 2, 3])
def test_urgency_is_monotone_in_the_countdown(timer):

    soon = extra_features(state(bombs=[((1, 1), timer)]))[0]
    later = extra_features(state(bombs=[((1, 1), timer + 1)]))[0]
    assert soon > later > 0.0


def test_q_agent_merges_every_timer_past_the_first():

    keys = {Q_AGENT.state_to_features(state(bombs=[((1, 1), t)])) for t in (1, 2, 3)}
    assert len(keys) == 1
    urgencies = [float(extra_features(state(bombs=[((1, 1), t)]))[0]) for t in (1, 2, 3)]
    assert urgencies[0] > urgencies[1] > urgencies[2] > 0.0


def test_proximity_decreases_with_distance_and_is_zero_when_absent():
    close = extra_features(state(coins=[(2, 1)]))[5]
    distant = extra_features(state(coins=[(7, 1)]))[5]
    assert 0.0 < distant < close <= 0.5
    assert extra_features(state())[5] == 0.0


def test_q_agent_merges_every_coin_distance():

    keys = {Q_AGENT.state_to_features(state(coins=[(1 + d, 1)])) for d in (1, 2, 3, 5, 7)}
    assert len(keys) == 1
    proximities = [float(extra_features(state(coins=[(1 + d, 1)]))[5]) for d in (1, 2, 3, 5, 7)]
    assert proximities == sorted(proximities, reverse=True)


def test_unreachable_target_reads_as_absent():
    field = empty_board()
    field[2, 1] = field[1, 2] = -1
    assert extra_features(state(coins=[(7, 1)], field=field))[5] == 0.0


def test_step_fraction_tracks_the_round_clock():
    assert extra_features(state(step=0))[7] == 0.0
    assert extra_features(state(step=settings.MAX_STEPS))[7] == 1.0


def test_walk_distances_are_true_step_counts():
    distances = walk_distances(empty_board(), (1, 1))
    assert distances[(1, 1)] == 0 and distances[(3, 1)] == 2
    assert (2, 2) not in distances
