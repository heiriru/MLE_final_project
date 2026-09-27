from __future__ import annotations
from agent_code.hybrid_agent.hybrid_h9.agent import QAgentHybrid, ResidualQNetwork
import os
from pathlib import Path
import torch

# load network and apply it with margin limit etc
def setup(self) -> None:
    self.unibomber_hybrid = QAgentHybrid(residual=ResidualQNetwork(6), residual_limit=float(os.environ.get('H9_RESIDUAL_LIMIT', '0.25')), override_margin=float(os.environ.get('H9_OVERRIDE_MARGIN', '0.25')))
    checkpoint = os.environ.get('BOMBERMAN_Q_AGENT_H9_MODEL') or Path(__file__).resolve().parent / 'residual.pt'
    if not Path(checkpoint).exists():
        raise RuntimeError(f'H9 residual weights not found at {checkpoint}')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    self.unibomber_hybrid.residual.load_state_dict(payload['residual'])
    self.unibomber_hybrid.residual.eval()

# Returns the hybrid's decision
def act(self, game_state: dict) -> str:
    decision = self.unibomber_hybrid.decide(game_state)
    return decision.action
