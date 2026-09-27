# H1 prior: turns the frozen q-agent's Q-values into normalised advantages A_D; as an agent it
# plays exactly the q-agent's action.
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from agent_code.hybrid_agent.adapter.callbacks import ACTIONS, QAgentAdapter, QAgentDecision
Q_ADVANTAGE_SCALE = 0.7439527672871241  ###calibration calculated from the q-table

# The adapter's decision plus the six normalised advantages A_D that enter the hybrid score.
@dataclass(frozen=True)
class PriorDecision:
    q_agent: QAgentDecision
    advantages: np.ndarray

    # The prior's action is the best advantage; by construction it equals the q-agent's action.
    @property
    def action(self):
        return ACTIONS[int(np.argmax(self.advantages))]

# A_D(a) = (Q(a) - mean_a Q) / c. Subtracting the mean removes the part that is the same for all
# actions; dividing by the fixed scale c puts all states on the same numerical scale. The order of
# the actions does not change.
def normalized_q_advantage(values, scale=Q_ADVANTAGE_SCALE):
    if scale <= 0:
        raise ValueError('scale must be positive')
    values = np.asarray(values, dtype=float)
    if values.shape != (len(ACTIONS),):
        raise ValueError('one value per action is required')
    return (values - values.mean()) / scale

# Turns the adapter's decision into the prior advantages A_D used by the hybrid.
class QAgentPrior:

    # Uses the frozen adapter and the calibrated scale c = 0.744.
    def __init__(self, adapter: QAgentAdapter | None=None, scale: float=Q_ADVANTAGE_SCALE):
        self.adapter = adapter or QAgentAdapter()
        self.scale = float(scale)


    # Table states: normalise the six Q-values. Fallback states: build a one-hot vector on the
    # fallback action and centre it. The assertion checks for every state that the prior still
    # prefers the q-agent's action.
    def decide(self, game_state):
        decision = self.adapter.decide(game_state)
        if decision.fallback_used:
            values = np.zeros(len(ACTIONS), dtype=float)
            values[ACTIONS.index(decision.action)] = 1.0
            advantages = normalized_q_advantage(values, scale=1.0)
        else:
            advantages = normalized_q_advantage(decision.q_values, self.scale)
        prior = PriorDecision(decision, advantages)
        if prior.action != decision.action:
            raise AssertionError('QAgent prior action diverged from frozen adapter')
        return prior
import json
import os
from pathlib import Path

# Creates the prior (H1 check agent).
def setup(self):
    self.q_agent_prior = QAgentPrior()
    self.q_agent_trace_path = os.environ.get('Q_AGENT_TRACE_PATH')
    self.q_agent_trace_agent = os.environ.get('Q_AGENT_TRACE_AGENT')


# Plays the prior's action, which must always equal the q-agent's; optionally writes a trace line.
def act(self, game_state: dict) -> str:
    prior = self.q_agent_prior.decide(game_state)
    decision = prior.q_agent
    if self.q_agent_trace_path and game_state['self'][0] == self.q_agent_trace_agent:
        record = {'round': game_state['round'], 'step': game_state['step'], 'features': decision.features, 'action': prior.action, 'known_state': decision.known_state, 'fallback_used': bool(decision.fallback_used)}
        path = Path(self.q_agent_trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, separators=(',', ':')) + '\n')
    return prior.action
