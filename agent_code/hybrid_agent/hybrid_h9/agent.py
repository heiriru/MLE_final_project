from __future__ import annotations
import importlib.util
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, ModuleType
import numpy as np
import torch
from torch import nn
SNAPSHOT = Path(__file__).resolve().parent / 'q_agent'
Q_TABLE_PATH = Path(os.environ.get('Q_AGENT_Q_TABLE') or SNAPSHOT / 'q_table.pkl')
ACTIONS = ('UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB')

# Loads the q-agent's own code
def _q_agent_callbacks() -> ModuleType:
    spec = importlib.util.spec_from_file_location('frozen_q_agent_callbacks', SNAPSHOT / 'callbacks.py')
    if spec is None or spec.loader is None:
        raise RuntimeError('could not load frozen QAgent callbacks')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
Q_AGENT = _q_agent_callbacks()

# Loads the q-agent's Q-table and freezes it
def load_frozen_q_table(path: Path=Q_TABLE_PATH) -> MappingProxyType:
    with path.open('rb') as handle:
        raw = pickle.load(handle)
    frozen = {}
    for state, values in raw.items():
        array = np.asarray(values, dtype=np.float64).copy()
        array.setflags(write=False)
        frozen[tuple(state)] = array
    return MappingProxyType(frozen)

# adapter returns the q-agent's features, its chosen action, state in q-table, q-values, if fallback used
@dataclass(frozen=True)
class QAgentDecision:
    features: tuple
    action: str
    known_state: bool
    q_values: np.ndarray | None
    fallback_used: bool

# turn q-agent into function:game state in, q-agent decision out
class QAgentAdapter:

    # modes: 'full' (normal), 'no_q_table' (always use the fallback), 'unseen_wait' (WAIT instead of the fallback).
    def __init__(self, q_table=None, mode='full'):
        if mode not in {'full', 'no_q_table', 'unseen_wait'}:
            raise ValueError(f'unknown QAgent adapter mode: {mode}')
        self.q_table = q_table if q_table is not None else load_frozen_q_table()
        self.mode = mode

    # state into features, look up q-values and take argmax. For unknown states or six equal q-values, use rule-based suggestted_action
    def decide(self, game_state):
        features = Q_AGENT.state_to_features(game_state)
        known = features in self.q_table
        values = self.q_table.get(features)
        use_table = known and self.mode != 'no_q_table'
        fallback = not use_table or np.ptp(values) < 1e-06
        if fallback:
            action = 'WAIT' if not known and self.mode == 'unseen_wait' else Q_AGENT.suggested_action(game_state)
        else:
            action = ACTIONS[int(np.argmax(values))]
        return QAgentDecision(features, action, known, values, fallback)
Q_ADVANTAGE_SCALE = 0.7439527672871241

# The adapter's decision plus the six normalised advantages A_D for hybrid score
@dataclass(frozen=True)
class PriorDecision:
    q_agent: QAgentDecision
    advantages: np.ndarray

    # priors actions being the same as q-agent
    @property
    def action(self):
        return ACTIONS[int(np.argmax(self.advantages))]

# A_D(a) = (Q(a) - mean_a Q) / c. Normalization for same numerical scales
def normalized_q_advantage(values, scale=Q_ADVANTAGE_SCALE):
    if scale <= 0:
        raise ValueError('scale must be positive')
    values = np.asarray(values, dtype=float)
    if values.shape != (len(ACTIONS),):
        raise ValueError('one value per action is required')
    return (values - values.mean()) / scale


class QAgentPrior:

    # use frozen adapter
    def __init__(self, adapter: QAgentAdapter | None=None, scale: float=Q_ADVANTAGE_SCALE):
        self.adapter = adapter or QAgentAdapter()
        self.scale = float(scale)

    # normalize q-values
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
RESIDUAL_LIMIT = float(os.environ.get('Q_AGENT_RESIDUAL_LIMIT', '0.25'))
OVERRIDE_MARGIN = float(os.environ.get('Q_AGENT_OVERRIDE_MARGIN', '0.25'))
ANCHOR_WEIGHT = float(os.environ.get('Q_AGENT_RESIDUAL_ANCHOR', '0.10'))

# L * tanh(raw): correction projection into [-L, L]
def bounded_residual(raw: torch.Tensor, residual_limit: float | None=None) -> torch.Tensor:
    return (RESIDUAL_LIMIT if residual_limit is None else residual_limit) * torch.tanh(raw)

# takes hybrid score if its better than q-agents by margin m, otherwise q-agents action 
def conservative_action(total, q_agent_action, override_margin=None):
    margin = OVERRIDE_MARGIN if override_margin is None else override_margin
    baseline = ACTIONS.index(q_agent_action)
    best = int(torch.argmax(total).item())
    if best != baseline and total[best] - total[baseline] < margin:
        return q_agent_action
    return ACTIONS[best]
INPUT_DIMENSION = 12
ACTION_DIMENSION = 6
HIDDEN_DIMENSION = 128

# two hidden layers of 128 ReLU units and a correction head
class ResidualQNetwork(nn.Module):

    # The correction head starts at zero
    def __init__(self, input_dimension: int=INPUT_DIMENSION) -> None:
        super().__init__()
        if input_dimension not in {6, 12}:
            raise ValueError('residual input dimension must be 6 or 12')
        self.input_dimension = input_dimension
        self.hidden = nn.Sequential(nn.Linear(input_dimension, HIDDEN_DIMENSION), nn.ReLU(), nn.Linear(HIDDEN_DIMENSION, HIDDEN_DIMENSION), nn.ReLU())
        self.output = nn.Linear(HIDDEN_DIMENSION, ACTION_DIMENSION)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    # Only the correction head returned
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.hidden(inputs))

    # Check if hybrid performs at zero head as q-agent
    def final_layer_is_zero(self) -> bool:
        return bool(torch.count_nonzero(self.output.weight) == 0 and torch.count_nonzero(self.output.bias) == 0)

# Builds the input vector [6 q-agent features, 6 advantages] 
def hybrid_inputs(features: tuple, advantages: np.ndarray) -> torch.Tensor:
    vector = np.concatenate((np.asarray(features, dtype=np.float32), np.asarray(advantages, dtype=np.float32)))
    if vector.shape != (12,):
        raise ValueError(f'expected exactly 12 hybrid inputs, got {vector.shape}')
    return torch.from_numpy(vector)

# hybrid computation for a state: prior, bounded correction, total scores, chosen action
@dataclass(frozen=True)
class HybridDecision:
    prior: PriorDecision
    residual_q: np.ndarray
    total_q: np.ndarray
    action: str

# q-agent prior + residual network + conservative selection.
class QAgentHybrid:

    # frozen prior+ residual network
    def __init__(self, prior: QAgentPrior | None=None, residual: ResidualQNetwork | None=None, residual_limit: float | None=None, override_margin: float | None=None) -> None:
        self.prior = prior or QAgentPrior()
        self.residual = residual or ResidualQNetwork()
        self.residual_limit, self.override_margin = (residual_limit, override_margin)
        self.residual.eval()

    # get the q-agent's advantages, compute the network's corrections and V(s), add them up to the hybrid score and pick the action with the conservative rule
    def decide(self, game_state: dict) -> HybridDecision:
        prior = self.prior.decide(game_state)
        inputs = hybrid_inputs(prior.q_agent.features, prior.advantages).unsqueeze(0)
        with torch.no_grad():
            residual_q = bounded_residual(self.residual(inputs[:, :self.residual.input_dimension]), self.residual_limit).squeeze(0)
            total_q = torch.as_tensor(prior.advantages, dtype=torch.float32) + residual_q
        action = conservative_action(total_q, prior.action, self.override_margin)
        return HybridDecision(prior, residual_q.cpu().numpy(), total_q.cpu().numpy(), action)
