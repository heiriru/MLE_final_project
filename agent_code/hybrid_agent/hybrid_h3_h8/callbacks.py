from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import torch
from agent_code.hybrid_agent.adapter.callbacks import ACTIONS, Q_AGENT
from agent_code.hybrid_agent.hybrid_h3_h8.agent import QAgentHybrid, ResidualQNetwork

# Builds the hybrid and loaded checkpoint
def setup(self):
    self.unibomber_hybrid = QAgentHybrid(residual=ResidualQNetwork(int(os.environ.get('Q_AGENT_HYBRID_INPUT_DIM', '12'))))
    checkpoint = os.environ.get('BOMBERMAN_Q_AGENT_HYBRID_MODEL')
    if checkpoint and Path(checkpoint).exists():
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.unibomber_hybrid.residual.load_state_dict(payload['residual'])
    elif not getattr(self, 'train', False) and (not self.unibomber_hybrid.residual.final_layer_is_zero()):
        raise AssertionError('H2 residual output layer must start exactly zero')
    self.q_agent_trace_path = os.environ.get('Q_AGENT_TRACE_PATH')
    self.q_agent_trace_agent = os.environ.get('Q_AGENT_TRACE_AGENT')

# Hybrid decision, epsilon-greedy exploration during training, and an optional trace line.
def act(self, game_state: dict) -> str:
    hybrid = self.unibomber_hybrid.decide(game_state)
    prior = hybrid.prior
    if getattr(self, 'train', False) and self.h3_rng.random() < self.h3_epsilon():
        action = str(self.h3_rng.choice(np.asarray(ACTIONS)))
    else:
        action = hybrid.action
    if self.q_agent_trace_path and game_state['self'][0] == self.q_agent_trace_agent:
        record = {'round': game_state['round'], 'step': game_state['step'], 'features': prior.q_agent.features, 'action': action, 'q_agent_action': prior.action, 'residual_override': bool(action != prior.action), 'mode': int(Q_AGENT.game_mode(game_state)), 'residual_max_abs': float(np.max(np.abs(hybrid.residual_q))), 'known_state': prior.q_agent.known_state, 'fallback_used': bool(prior.q_agent.fallback_used)}
        path = Path(self.q_agent_trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, separators=(',', ':')) + '\n')
    return action
