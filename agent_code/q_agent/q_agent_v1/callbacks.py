# q-agent v1: the earlier version of our q-agent (Sept 8 table) that the hybrid was built on; the
# reference opponent in all hybrid experiments.
from __future__ import annotations
import importlib.util
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, ModuleType
import numpy as np
SNAPSHOT = Path(__file__).resolve().parent / 'q_agent'

# Where the Q-table comes from: an explicit path in Q_AGENT_Q_TABLE, otherwise the table in the
# local snapshot folder ./q_agent.
def _q_table_path() -> Path:
    override = os.environ.get('Q_AGENT_Q_TABLE')
    candidates = [Path(override)] if override else []
    candidates += [SNAPSHOT / 'q_table.pkl']
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'no QAgent Q-table found; looked in {[str(c) for c in candidates]}')
Q_TABLE_PATH = _q_table_path()
ACTIONS = ('UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB')

# Loads the q-agent's own code (state_to_features, suggested_action) from the snapshot folder
# ./q_agent as a separate module, so we use exactly the q-agent version the hybrid was built on.
def _q_agent_callbacks() -> ModuleType:
    spec = importlib.util.spec_from_file_location('frozen_q_agent_callbacks', SNAPSHOT / 'callbacks.py')
    if spec is None or spec.loader is None:
        raise RuntimeError('could not load frozen QAgent callbacks')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
Q_AGENT = _q_agent_callbacks()

# Loads the q-agent's Q-table and freezes it: every value array is made non-writable and the dict
# is wrapped in a MappingProxyType, so no code can change the table or add states by accident.
def load_frozen_q_table(path: Path=Q_TABLE_PATH) -> MappingProxyType:
    with path.open('rb') as handle:
        raw = pickle.load(handle)
    frozen = {}
    for state, values in raw.items():
        array = np.asarray(values, dtype=np.float64).copy()
        array.setflags(write=False)
        frozen[tuple(state)] = array
    return MappingProxyType(frozen)

# What the adapter returns for one game state: the q-agent's features, its chosen action, whether
# the state was in the table, the raw Q-values, and whether the fallback was used.
@dataclass(frozen=True)
class QAgentDecision:
    features: tuple
    action: str
    known_state: bool
    q_values: np.ndarray | None
    fallback_used: bool

# Wraps the frozen q-agent so it can be queried like a function: game state in, q-agent decision
# out. It never learns.
class QAgentAdapter:

    # mode is only for the H1 ablation: 'full' (normal), 'no_q_table' (always use the fallback),
    # 'unseen_wait' (WAIT instead of the fallback for unseen states).
    def __init__(self, q_table=None, mode='full'):
        if mode not in {'full', 'no_q_table', 'unseen_wait'}:
            raise ValueError(f'unknown QAgent adapter mode: {mode}')
        self.q_table = q_table if q_table is not None else load_frozen_q_table()
        self.mode = mode

    # Encodes the state with the q-agent's own state_to_features, looks it up in the table and
    # takes the argmax. If the state is unknown or all six values are equal, the q-agent's rule-
    # based suggested_action is used instead, exactly as the q-agent itself would do.
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
import json

# Creates the frozen q-agent v1: the reference opponent of all hybrid experiments.
def setup(self):
    self.unibomber_adapter = QAgentAdapter(mode=os.environ.get('Q_AGENT_ADAPTER_MODE', 'full'))
    self.q_agent_trace_path = os.environ.get('Q_AGENT_TRACE_PATH')
    self.q_agent_trace_agent = os.environ.get('Q_AGENT_TRACE_AGENT')
    self.logger.info('Loaded read-only QAgent adapter (%s states, mode=%s).', len(self.unibomber_adapter.q_table), self.unibomber_adapter.mode)

# Plays exactly the q-agent's action; optionally writes a trace line (used for the H1/H2 parity
# check).
def act(self, game_state: dict) -> str:
    decision = self.unibomber_adapter.decide(game_state)
    if self.q_agent_trace_path and game_state['self'][0] == self.q_agent_trace_agent:
        record = {'round': game_state['round'], 'step': game_state['step'], 'features': decision.features, 'action': decision.action, 'known_state': decision.known_state, 'fallback_used': bool(decision.fallback_used)}
        path = Path(self.q_agent_trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, separators=(',', ':')) + '\n')
    return decision.action
