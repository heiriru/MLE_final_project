from __future__ import annotations
import importlib.util
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, ModuleType
import numpy as np
import torch
from collections import deque
import settings
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

#load agents q table
def load_frozen_q_table(path: Path=Q_TABLE_PATH) -> MappingProxyType:
    with path.open('rb') as handle:
        raw = pickle.load(handle)
    frozen = {}
    for state, values in raw.items():
        array = np.asarray(values, dtype=np.float64).copy()
        array.setflags(write=False)
        frozen[tuple(state)] = array
    return MappingProxyType(frozen)

# returns the q-agent's features, its chosen action, whether the state was in the table, the raw Q-values, and whether the fallback was used.
@dataclass(frozen=True)
class QAgentDecision:
    features: tuple
    action: str
    known_state: bool
    q_values: np.ndarray | None
    fallback_used: bool

# warpping for q-agent so we can call it like a function that takes the state and returns its decision
class QAgentAdapter:

    # mode meaning: 'full' (normal), 'no_q_table' (use fallback), or unseen_wait - use wait when state is unseen instead of fallback
    def __init__(self, q_table=None, mode='full'):
        if mode not in {'full', 'no_q_table', 'unseen_wait'}:
            raise ValueError(f'unknown QAgent adapter mode: {mode}')
        self.q_table = q_table if q_table is not None else load_frozen_q_table()
        self.mode = mode

    # Calculate state in six features, find q-table value and decision via argmax of the values. For new state or six equal values, take rule-based agetns suggested_action
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

# output the decision of the adapter plus normalised advantages for hybrid score
@dataclass(frozen=True)
class PriorDecision:
    q_agent: QAgentDecision
    advantages: np.ndarray

    # priors actions is from q-agent
    @property
    def action(self):
        return ACTIONS[int(np.argmax(self.advantages))]

# A_D(a) = (Q(a) - mean_a Q) / c. Normalize
def normalized_q_advantage(values, scale=Q_ADVANTAGE_SCALE):
    if scale <= 0:
        raise ValueError('scale must be positive')
    values = np.asarray(values, dtype=float)
    if values.shape != (len(ACTIONS),):
        raise ValueError('one value per action is required')
    return (values - values.mean()) / scale

class QAgentPrior:

    def __init__(self, adapter: QAgentAdapter | None=None, scale: float=Q_ADVANTAGE_SCALE):
        self.adapter = adapter or QAgentAdapter()
        self.scale = float(scale)

    # Table states: normalise the six Q-values. Fallback states: build vector on the fallback action and centre it. For every state, prior prefers q-agents action via assertion
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

# L * tanh(raw): project network correction into [-L, L], network can now only correct so much
def bounded_residual(raw: torch.Tensor, residual_limit: float | None=None) -> torch.Tensor:
    return (RESIDUAL_LIMIT if residual_limit is None else residual_limit) * torch.tanh(raw)

# Calculate best action: correction from residual network if it beats q-agents action by margin m, otherwise q-agent
def conservative_action(total, q_agent_action, override_margin=None):
    margin = OVERRIDE_MARGIN if override_margin is None else override_margin
    baseline = ACTIONS.index(q_agent_action)
    best = int(torch.argmax(total).item())
    if best != baseline and total[best] - total[baseline] < margin:
        return q_agent_action
    return ACTIONS[best]
EXTRA_DIMENSION = 8
DANGER_SCALE = float(settings.BOMB_TIMER + 2)
FEATURE_DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))

# Calculation when tiles become deadly: 1 for an current explosion, timer + 1 for tiles in a bomb's blast, 0 for safe tiles. If several bombs reach a tile, the most urgent one counts.
def danger_countdown(game_state: dict) -> np.ndarray:
    field = np.asarray(game_state['field'])
    countdown = np.zeros_like(field, dtype=np.int16)
    countdown[np.asarray(game_state.get('explosion_map', 0)) > 0] = 1
    for (bx, by), timer in game_state.get('bombs', ()):
        steps = int(timer) + 1
        for cx, cy in _blast_coords(field, bx, by):
            if countdown[cx, cy] == 0 or steps < countdown[cx, cy]:
                countdown[cx, cy] = steps
    return countdown

# All tiles a bomb at (x, y) would hit: its own tile plus up to BOMB_POWER tiles in each
# direction, stopped by walls.
def _blast_coords(field: np.ndarray, x: int, y: int) -> list[tuple[int, int]]:
    coords = [(x, y)]
    for dx, dy in FEATURE_DELTAS:
        for step in range(1, settings.BOMB_POWER + 1):
            nx, ny = (x + dx * step, y + dy * step)
            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]) or field[nx, ny] == -1:
                break
            coords.append((nx, ny))
    return coords

# Breadth-first search from the agent's tile over free tiles (walls and crates block, bombs and
# agents do not). Returns the walking distance to every reachable tile.
def walk_distances(field: np.ndarray, start: tuple[int, int]) -> dict[tuple[int, int], int]:
    distances = {tuple(start): 0}
    queue = deque([tuple(start)])
    while queue:
        x, y = queue.popleft()
        for dx, dy in FEATURE_DELTAS:
            nxt = (x + dx, y + dy)
            if nxt in distances:
                continue
            if not (0 <= nxt[0] < field.shape[0] and 0 <= nxt[1] < field.shape[1]):
                continue
            if field[nxt] != 0:
                continue
            distances[nxt] = distances[x, y] + 1
            queue.append(nxt)
    return distances

# 1 / (1 + distance to the nearest reachable target), or 0 if no target is reachable
def _proximity(distances: dict[tuple[int, int], int], targets) -> float:
    reachable = [distances[tuple(t)] for t in targets if tuple(t) in distances]
    return 1.0 / (1.0 + min(reachable)) if reachable else 0.0

# eight other inputs: danger urgency on the own tile and the four neighbours, proximity to the nearest coin and the nearest opponent, and the round progress step/400
def extra_features(game_state: dict) -> np.ndarray:
    field = np.asarray(game_state['field'])
    position = tuple(game_state['self'][3])
    countdown = danger_countdown(game_state)

    # (6 - steps) / 6 for a tile that becomes lethal inn steps, 0 for a safe tile
    def urgency(tile: tuple[int, int]) -> float:
        if not (0 <= tile[0] < field.shape[0] and 0 <= tile[1] < field.shape[1]):
            return 0.0
        steps = int(countdown[tile])
        return 0.0 if steps == 0 else float(np.clip((DANGER_SCALE - steps) / DANGER_SCALE, 0.0, 1.0))
    values = [urgency(position)]
    values += [urgency((position[0] + dx, position[1] + dy)) for dx, dy in FEATURE_DELTAS]
    distances = walk_distances(field, position)
    values.append(_proximity(distances, game_state.get('coins', ())))
    values.append(_proximity(distances, [other[3] for other in game_state.get('others', ())]))
    values.append(min(float(game_state.get('step', 0)) / settings.MAX_STEPS, 1.0))
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (EXTRA_DIMENSION,):
        raise ValueError(f'expected {EXTRA_DIMENSION} extra features, got {vector.shape}')
    return vector
INPUT_DIMENSION = 14
ALLOWED_DIMENSIONS = (6, 12, 14)
ACTION_DIMENSION = 6
HIDDEN_DIMENSION = 128

# two hidden layers of 128 ReLU units and two heads, a correction and one value V(s) head for the state
class ResidualQNetwork(nn.Module):

    # heads start with weights and bias at 0
    def __init__(self, input_dimension: int=INPUT_DIMENSION) -> None:
        super().__init__()
        if input_dimension not in ALLOWED_DIMENSIONS:
            raise ValueError(f'residual input dimension must be one of {ALLOWED_DIMENSIONS}')
        self.input_dimension = input_dimension
        self.hidden = nn.Sequential(nn.Linear(input_dimension, HIDDEN_DIMENSION), nn.ReLU(), nn.Linear(HIDDEN_DIMENSION, HIDDEN_DIMENSION), nn.ReLU())
        self.output = nn.Linear(HIDDEN_DIMENSION, ACTION_DIMENSION)
        self.value = nn.Linear(HIDDEN_DIMENSION, 1)
        for head in (self.output, self.value):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    # return only corrections
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.hidden(inputs))

    # return corrections and V(s)
    def heads(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.hidden(inputs)
        return (self.output(features), self.value(features))

    def state_value(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.value(self.hidden(inputs))

    # test whether hybrid is same as q-agent
    def final_layer_is_zero(self) -> bool:
        return all((bool(torch.count_nonzero(head.weight) == 0 and torch.count_nonzero(head.bias) == 0) for head in (self.output, self.value)))
Q_AGENT_DIMENSION = 6
OBSERVATION_DIMENSION = Q_AGENT_DIMENSION + EXTRA_DIMENSION
ADVANTAGE_OFFSET = OBSERVATION_DIMENSION
INPUT_DIMENSION = OBSERVATION_DIMENSION + len(ACTIONS)

# build input vector here
def hybrid_inputs(features: tuple, extras: np.ndarray, advantages: np.ndarray) -> torch.Tensor:
    vector = np.concatenate((np.asarray(features, dtype=np.float32), np.asarray(extras, dtype=np.float32), np.asarray(advantages, dtype=np.float32)))
    if vector.shape != (INPUT_DIMENSION,):
        raise ValueError(f'expected exactly {INPUT_DIMENSION} hybrid inputs, got {vector.shape}')
    return torch.from_numpy(vector)

# full input vector for one state
def state_inputs(game_state: dict, prior: PriorDecision) -> torch.Tensor:
    return hybrid_inputs(prior.q_agent.features, extra_features(game_state), prior.advantages)

# Q = V(s) + A_D(a) + L*tanh(delta(a)) for every action (also for whole batches in training)
def residual_total(residual: ResidualQNetwork, inputs: torch.Tensor, residual_limit: float | None=None) -> torch.Tensor:
    observation = inputs[..., :residual.input_dimension]
    correction, value = residual.heads(observation)
    return value + inputs[..., ADVANTAGE_OFFSET:] + bounded_residual(correction, residual_limit)

# all hybrids compuitations for one state
@dataclass(frozen=True)
class HybridDecision:
    prior: PriorDecision
    residual_q: np.ndarray
    total_q: np.ndarray
    action: str
    state_value: float = 0.0
    inputs: np.ndarray | None = None

# hybrid policy: q-agent prior + residual network + conservative selection
class QAgentHybrid:

      # frozen prior + residual network
    def __init__(self, prior: QAgentPrior | None=None, residual: ResidualQNetwork | None=None, residual_limit: float | None=None, override_margin: float | None=None) -> None:
        self.prior = prior or QAgentPrior()
        self.residual = residual or ResidualQNetwork()
        self.residual_limit, self.override_margin = (residual_limit, override_margin)
        self.residual.eval()

    # calcullate hybrid score and make a decision at last with our conservative rule
    def decide(self, game_state: dict) -> HybridDecision:
        prior = self.prior.decide(game_state)
        inputs = state_inputs(game_state, prior).unsqueeze(0)
        with torch.no_grad():
            observation = inputs[:, :self.residual.input_dimension]
            correction, value = self.residual.heads(observation)
            residual_q = bounded_residual(correction, self.residual_limit).squeeze(0)
            state_value = float(value.squeeze())
            total_q = torch.as_tensor(prior.advantages, dtype=torch.float32) + residual_q + state_value
        action = conservative_action(total_q, prior.action, self.override_margin)
        return HybridDecision(prior, residual_q.cpu().numpy(), total_q.cpu().numpy(), action, state_value, inputs.squeeze(0).numpy())
