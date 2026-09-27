from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import torch
from agent_code.hybrid_agent.hybrid_h11.agent import ACTIONS, Q_AGENT, QAgentHybrid, OBSERVATION_DIMENSION, ResidualQNetwork

# Loads trained weights into the network
def load_residual(residual: ResidualQNetwork, checkpoint: Path) -> str:
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = payload['residual']
    missing, unexpected = residual.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f'unexpected keys in {checkpoint}: {sorted(unexpected)}')
    if not missing:
        return 'dueling checkpoint'
    if sorted(missing) != ['value.bias', 'value.weight']:
        raise RuntimeError(f'incompatible checkpoint {checkpoint}: missing {sorted(missing)}')
    return 'pre-dueling checkpoint, value head starts at zero'

# builds the hybrid (input size, limit and margin),load the trained checkpoint. Without a checkpoint starts in training mode starting as q-agent
def setup(self) -> None:
    dimension = int(os.environ.get('Q_AGENT_H11_INPUT_DIM', str(OBSERVATION_DIMENSION)))
    self.unibomber_hybrid = QAgentHybrid(residual=ResidualQNetwork(dimension), residual_limit=float(os.environ.get('H11_RESIDUAL_LIMIT', '0.25')), override_margin=float(os.environ.get('H11_OVERRIDE_MARGIN', '0.25')))
    checkpoint = os.environ.get('BOMBERMAN_Q_AGENT_H11_MODEL') or Path(__file__).resolve().parent / 'residual.pt'
    if Path(checkpoint).exists():
        note = load_residual(self.unibomber_hybrid.residual, Path(checkpoint))
        self.logger.info(f'H11 loaded {checkpoint} ({note})')
    elif not getattr(self, 'train', False):
        raise RuntimeError(f'H11 residual weights not found at {checkpoint}')
    else:
        self.logger.info('H11 training from a zero residual: the agent starts as QAgent')
    self.unibomber_hybrid.residual.eval()
    self.q_agent_trace_path = os.environ.get('Q_AGENT_TRACE_PATH')
    self.q_agent_trace_agent = os.environ.get('Q_AGENT_TRACE_AGENT')

## lets the hybrid decide. at probability epsilon do random move for explorationt
def act(self, game_state: dict) -> str:
    hybrid = self.unibomber_hybrid.decide(game_state)
    decision = hybrid.prior.q_agent
    if getattr(self, 'train', False) and self.h3_rng.random() < self.h3_epsilon():
        action = str(self.h3_rng.choice(np.asarray(ACTIONS)))
    else:
        action = hybrid.action
    if self.q_agent_trace_path and game_state['self'][0] == self.q_agent_trace_agent:
        record = {'round': game_state['round'], 'step': game_state['step'], 'features': hybrid.prior.q_agent.features, 'action': action, 'q_agent_action': hybrid.prior.action, 'residual_override': bool(hybrid.action != hybrid.prior.action), 'deployed_override': bool(action != hybrid.prior.action), 'mode': int(Q_AGENT.game_mode(game_state)), 'residual_max_abs': float(np.max(np.abs(hybrid.residual_q))), 'state_value': hybrid.state_value, 'known_state': hybrid.prior.q_agent.known_state, 'fallback_used': bool(hybrid.prior.q_agent.fallback_used)}
        path = Path(self.q_agent_trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, separators=(',', ':')) + '\n')
    return action
