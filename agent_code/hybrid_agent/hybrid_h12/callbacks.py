import json
import os
from pathlib import Path
import numpy as np
import torch
from agent_code.hybrid_agent.hybrid_h12.agent import ACTIONS, OBSERVATION_DIMENSION, OVERRIDE_MARGIN, Q_AGENT, QAgentHybrid, RESIDUAL_LIMIT, ResidualQNetwork

# Loads trained weights into the network
def load_residual(residual, checkpoint):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    missing, unexpected = residual.load_state_dict(payload['residual'], strict=False)
    if unexpected:
        raise RuntimeError(f'unexpected keys in {checkpoint}: {sorted(unexpected)}')
    if not missing:
        return 'dueling checkpoint'
    if sorted(missing) != ['value.bias', 'value.weight']:
        raise RuntimeError(f'incompatible checkpoint {checkpoint}: missing {sorted(missing)}')
    return 'pre-dueling checkpoint, value head starts at zero'

# builds the hybrid (input size, limit and margin),load the trained checkpoint. Without a checkpoint starts in training mode starting as q-agent
def setup(self):
    dimension = int(os.environ.get('Q_AGENT_H12_INPUT_DIM', str(OBSERVATION_DIMENSION)))
    self.unibomber_hybrid = QAgentHybrid(residual=ResidualQNetwork(dimension), residual_limit=float(os.environ.get('H12_RESIDUAL_LIMIT', RESIDUAL_LIMIT)), override_margin=float(os.environ.get('H12_OVERRIDE_MARGIN', OVERRIDE_MARGIN)))
    checkpoint = Path(os.environ.get('BOMBERMAN_Q_AGENT_H12_MODEL') or Path(__file__).resolve().parent / 'residual.pt')
    if checkpoint.exists():
        note = load_residual(self.unibomber_hybrid.residual, checkpoint)
        self.logger.info(f'H12 loaded {checkpoint} ({note})')
    elif not getattr(self, 'train', False):
        raise RuntimeError(f'H12 residual weights not found at {checkpoint}')
    else:
        self.logger.info('H12 training from a zero residual: the agent starts as QAgent')
    self.unibomber_hybrid.residual.eval()
    self.q_agent_trace_path = os.environ.get('Q_AGENT_TRACE_PATH')
    self.q_agent_trace_agent = os.environ.get('Q_AGENT_TRACE_AGENT')

# lets the hybrid decide. at probability epsilon do random move for explorationt
def act(self, game_state):
    hybrid = self.unibomber_hybrid.decide(game_state)
    if getattr(self, 'train', False) and self.h3_rng.random() < self.h3_epsilon():
        action = str(self.h3_rng.choice(np.asarray(ACTIONS)))
    else:
        action = hybrid.action
    if self.q_agent_trace_path and game_state['self'][0] == self.q_agent_trace_agent:
        write_trace(self, game_state, hybrid, action)
    return action

# to a trace file. Used to measure how often the hybrid deviates from the q-agent.
def write_trace(self, game_state, hybrid, action):
    q_agent = hybrid.prior.q_agent
    record = {'round': game_state['round'], 'step': game_state['step'], 'features': q_agent.features, 'action': action, 'q_agent_action': hybrid.prior.action, 'residual_override': bool(hybrid.action != hybrid.prior.action), 'deployed_override': bool(action != hybrid.prior.action), 'mode': int(Q_AGENT.game_mode(game_state)), 'residual_max_abs': float(np.max(np.abs(hybrid.residual_q))), 'state_value': hybrid.state_value, 'known_state': q_agent.known_state, 'fallback_used': bool(q_agent.fallback_used)}
    path = Path(self.q_agent_trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, separators=(',', ':')) + '\n')
