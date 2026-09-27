from __future__ import annotations
import os
import numpy as np
import torch
from agent_code.hybrid_agent.adapter.callbacks import ACTIONS
from torch import nn
from dataclasses import dataclass
from agent_code.hybrid_agent.prior.callbacks import QAgentPrior, PriorDecision
RESIDUAL_LIMIT = float(os.environ.get('Q_AGENT_RESIDUAL_LIMIT', '0.25'))
OVERRIDE_MARGIN = float(os.environ.get('Q_AGENT_OVERRIDE_MARGIN', '0.25'))
ANCHOR_WEIGHT = float(os.environ.get('Q_AGENT_RESIDUAL_ANCHOR', '0.10'))

# L * tanh(raw): project residual output onto [-L,L]
def bounded_residual(raw: torch.Tensor, residual_limit: float | None=None) -> torch.Tensor:
    return (RESIDUAL_LIMIT if residual_limit is None else residual_limit) * torch.tanh(raw)

# residual network output if it outperforms q-agents decision by margin m, otherwise q-agents decision
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

    # 12 inputs (6 q-agent features + 6 advantages)
    def __init__(self, input_dimension: int=INPUT_DIMENSION) -> None:
        super().__init__()
        if input_dimension not in {6, 12}:
            raise ValueError('residual input dimension must be 6 or 12')
        self.input_dimension = input_dimension
        self.hidden = nn.Sequential(nn.Linear(input_dimension, HIDDEN_DIMENSION), nn.ReLU(), nn.Linear(HIDDEN_DIMENSION, HIDDEN_DIMENSION), nn.ReLU())
        self.output = nn.Linear(HIDDEN_DIMENSION, ACTION_DIMENSION)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    #six raw corrections.
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.hidden(inputs))

    # hybrid identical to the q-agent?
    def final_layer_is_zero(self) -> bool:
        return bool(torch.count_nonzero(self.output.weight) == 0 and torch.count_nonzero(self.output.bias) == 0)

# Input vector [6 q-agent features, 6 advantages].
def hybrid_inputs(features: tuple, advantages: np.ndarray) -> torch.Tensor:
    vector = np.concatenate((np.asarray(features, dtype=np.float32), np.asarray(advantages, dtype=np.float32)))
    if vector.shape != (12,):
        raise ValueError(f'expected exactly 12 hybrid inputs, got {vector.shape}')
    return torch.from_numpy(vector)

# Prior, bounded correction, total scores and chosen action for one state.
@dataclass(frozen=True)
class HybridDecision:
    prior: PriorDecision
    residual_q: np.ndarray
    total_q: np.ndarray
    action: str

# q-agent prior + bounded correction + conservative rule.
class QAgentHybrid:

    # q-agent prior with residual network combined
    def __init__(self, prior: QAgentPrior | None=None, residual: ResidualQNetwork | None=None, residual_limit: float | None=None, override_margin: float | None=None) -> None:
        self.prior = prior or QAgentPrior()
        self.residual = residual or ResidualQNetwork()
        self.residual_limit, self.override_margin = (residual_limit, override_margin)
        self.residual.eval()

    # Score = A_D + L*tanh(correction) plus conservative decision pick
    def decide(self, game_state: dict) -> HybridDecision:
        prior = self.prior.decide(game_state)
        inputs = hybrid_inputs(prior.q_agent.features, prior.advantages).unsqueeze(0)
        with torch.no_grad():
            residual_q = bounded_residual(self.residual(inputs[:, :self.residual.input_dimension]), self.residual_limit).squeeze(0)
            total_q = torch.as_tensor(prior.advantages, dtype=torch.float32) + residual_q
        action = conservative_action(total_q, prior.action, self.override_margin)
        return HybridDecision(prior, residual_q.cpu().numpy(), total_q.cpu().numpy(), action)
