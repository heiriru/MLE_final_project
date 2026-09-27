from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from collections import deque
import os
import json
from pathlib import Path
from time import perf_counter
import torch
from torch.nn import functional as F
import events as e
from agent_code.hybrid_agent.hybrid_h10.agent import ACTIONS, hybrid_inputs, residual_total, ANCHOR_WEIGHT, RESIDUAL_LIMIT, bounded_residual, ResidualQNetwork

# One saved transition, inputs of the state, action, (n-step) reward, inputs of the next state, if round ended, and how many steps the reward covers.
@dataclass(frozen=True)
class Transition:
    inputs: np.ndarray
    action: int
    reward: float
    next_inputs: np.ndarray
    done: bool
    n_steps: int = 1

# Experience replay buffer: stores state transitions and creates mini batches (for training later)
class UniformReplay:

    # Empty buffer with its random generator, sampling is reproducible for a seed.
    def __init__(self, capacity: int, seed: int=0) -> None:
        if capacity < 1:
            raise ValueError('capacity must be positive')
        self.capacity, self.rng = (capacity, np.random.default_rng(seed))
        self.items: list[Transition] = []
        self.cursor = 0

     # Appends a transition; if buffer is full, rewrite oldest one
    def add(self, transition: Transition) -> None:
        if len(self.items) < self.capacity:
            self.items.append(transition)
        else:
            self.items[self.cursor] = transition
        self.cursor = (self.cursor + 1) % self.capacity

    # Draws batch_size transitions uniformly at random
    def sample(self, batch_size: int) -> list[Transition]:
        if not self.items:
            raise ValueError('cannot sample empty replay')
        indices = self.rng.integers(0, len(self.items), size=min(batch_size, len(self.items)))
        return [self.items[int(index)] for index in indices]

    # Number of stored transitions (to decide when training starts)
    def __len__(self) -> int:
        return len(self.items)

# From single steps to n-step transitions, for the n-step rewards
class NStepAccumulator:

    # n = number of steps per transition (3 in the final agent), gamma = discount.
    def __init__(self, n: int, gamma: float) -> None:
        if n < 1 or not 0 < gamma <= 1:
            raise ValueError('n >= 1 and 0 < gamma <= 1 required')
        self.n, self.gamma, self.pending = (n, gamma, deque())

    # Adds one step and returns completed n-step transitions
    def append(self, transition: Transition) -> list[Transition]:
        self.pending.append(transition)
        return self._drain(terminal=transition.done)

    # Sums up rewards one n-step transition from the first state to the state n steps later
    def _drain(self, terminal: bool) -> list[Transition]:
        emitted = []
        while self.pending and (terminal or len(self.pending) >= self.n):
            horizon = min(self.n, len(self.pending))
            window = list(self.pending)[:horizon]
            first, last = (window[0], window[-1])
            reward = sum((self.gamma ** k * item.reward for k, item in enumerate(window)))
            emitted.append(Transition(first.inputs, first.action, reward, last.next_inputs, last.done, horizon))
            self.pending.popleft()
            if not terminal:
                break
        return emitted
COIN_WEIGHT = float(os.environ.get('Q_AGENT_H5_COIN_PHI', '0.20'))
CRATE_ACCESS_WEIGHT = float(os.environ.get('Q_AGENT_H5_CRATE_PHI', '0.04'))

# reward shaping potential: increase the closer it is to a coin (-0.20 * distance / 34) or next to a crate(+0.04 per crate). 0 after the round has ended
def potential(game_state: dict | None) -> float:
    if game_state is None:
        return 0.0
    field = np.asarray(game_state['field'])
    x, y = game_state['self'][3]
    scale = max(field.shape[0] + field.shape[1], 1)
    coins = game_state.get('coins', ())
    coin_term = 0.0 if not coins else -COIN_WEIGHT * min((abs(x - cx) + abs(y - cy) for cx, cy in coins)) / scale
    crate_neighbors = sum((field[x + dx, y + dy] == 1 for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0))))
    return float(coin_term + CRATE_ACCESS_WEIGHT * crate_neighbors)
ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = os.environ.get('BOMBERMAN_Q_AGENT_H10_MODEL', 'results/q_agent_hybrid/hybrid/checkpoints/H10_residual.pkl')
METRICS_PATH = os.environ.get('BOMBERMAN_Q_AGENT_H10_METRICS', 'results/q_agent_hybrid/hybrid/runs/H10_metrics.jsonl')
GAMMA = float(os.environ.get('Q_AGENT_H3_GAMMA', '0.99'))
LEARNING_RATE = float(os.environ.get('Q_AGENT_H3_LR', '0.0003'))
BATCH_SIZE = int(os.environ.get('Q_AGENT_H3_BATCH', '64'))
CAPACITY = int(os.environ.get('Q_AGENT_H3_CAPACITY', '100000'))
LEARNING_STARTS = int(os.environ.get('Q_AGENT_H3_LEARNING_STARTS', '1000'))
TRAIN_EVERY = int(os.environ.get('Q_AGENT_H3_TRAIN_EVERY', '2'))
TARGET_EVERY = int(os.environ.get('Q_AGENT_H3_TARGET_EVERY', '500'))
EPS_START, EPS_FINAL = (float(os.environ.get('Q_AGENT_H3_EPS_START', '0.20')), float(os.environ.get('Q_AGENT_H3_EPS_FINAL', '0.05')))
EPS_DECAY = int(os.environ.get('Q_AGENT_H3_EPS_DECAY', '20000'))
N_STEP = int(os.environ.get('Q_AGENT_H4_N_STEP', '1'))
REWARDS = {e.COIN_COLLECTED: 1.0, e.KILLED_OPPONENT: 5.0, e.KILLED_SELF: -5.0, e.GOT_KILLED: -5.0}

# Double-DQN target: the online network chooses the best next action, the target network evaluates it. Target = n-step reward + gamma^n * that value 
def double_dqn_targets(online, target, rewards, dones, next_inputs, gamma, n_steps=None):
    with torch.no_grad():
        selected = residual_total(online, next_inputs).argmax(dim=1, keepdim=True)
        target_next = residual_total(target, next_inputs).gather(1, selected).squeeze(1)
        discount = gamma if n_steps is None else torch.as_tensor(gamma) ** n_steps
        return rewards + discount * (1.0 - dones) * target_next


# gradient step on a mini-batch: predicted hybrid score of the taken action vs. the Double-DQN target, Smooth-L1 loss plus a small anchor term that pulls the correction towards zero, gradient clipping at 10, using Adam update.
def train_step(online: ResidualQNetwork, target: ResidualQNetwork, optimizer, batch: list[Transition], gamma: float=GAMMA) -> float:
    inputs = torch.as_tensor(np.stack([item.inputs for item in batch]), dtype=torch.float32)
    actions = torch.as_tensor([item.action for item in batch], dtype=torch.long)
    rewards = torch.as_tensor([item.reward for item in batch], dtype=torch.float32)
    dones = torch.as_tensor([item.done for item in batch], dtype=torch.float32)
    next_inputs = torch.as_tensor(np.stack([item.next_inputs for item in batch]), dtype=torch.float32)
    n_steps = torch.as_tensor([item.n_steps for item in batch], dtype=torch.float32)
    targets = double_dqn_targets(online, target, rewards, dones, next_inputs, gamma, n_steps)
    predicted = residual_total(online, inputs).gather(1, actions[:, None]).squeeze(1)
    correction = bounded_residual(online(inputs[..., :online.input_dimension]))
    loss = F.smooth_l1_loss(predicted, targets) + ANCHOR_WEIGHT * correction.square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
    optimizer.step()
    return float(loss.detach())

# target network copied from online network, replay buffer, Adam optimiser, the n-step accumulator and the linear epsilon schedule
def setup_training(self) -> None:
    seed = int(os.environ.get('BOMBERMAN_SEED', '101'))
    torch.set_num_threads(int(os.environ.get('Q_AGENT_H3_CPU_THREADS', '1')))
    self.unibomber_hybrid.residual.to('cpu')
    self.h3_target = ResidualQNetwork(self.unibomber_hybrid.residual.input_dimension).to('cpu')
    self.h3_target.load_state_dict(self.unibomber_hybrid.residual.state_dict())
    self.h3_replay, self.h3_optimizer = (UniformReplay(CAPACITY, seed), torch.optim.Adam(self.unibomber_hybrid.residual.parameters(), lr=LEARNING_RATE))
    self.h4_queue = NStepAccumulator(N_STEP, GAMMA)
    self.h3_rng, self.h3_interactions, self.h3_updates = (np.random.default_rng(seed), 0, 0)
    self.h3_losses, self.h3_rewards = (deque(maxlen=200), deque(maxlen=100))
    self.h3_round_reward, self.h3_started = (0.0, perf_counter())
    self.h3_epsilon = lambda: EPS_START + min(self.h3_interactions / EPS_DECAY, 1.0) * (EPS_FINAL - EPS_START)
    metrics = ROOT / METRICS_PATH
    metrics.parent.mkdir(parents=True, exist_ok=True)
    self.h3_metrics_handle = metrics.open('w', encoding='utf-8')
    history_path = os.environ.get('BOMBERMAN_Q_AGENT_TRAIN_HISTORY')
    self.h3_history_handle = None
    if history_path:
        history = ROOT / history_path
        history.parent.mkdir(parents=True, exist_ok=True)
        self.h3_history_handle = history.open('w', encoding='utf-8')

# compute everything for 1 transtition
def _transition(self, old_state, action: str, events, new_state, done: bool) -> None:
    if old_state is None or action not in ACTIONS:
        return
    current = self.unibomber_hybrid.decide(old_state)
    if new_state is None:
        next_inputs = np.zeros(12, dtype=np.float32)
    else:
        following = self.unibomber_hybrid.decide(new_state)
        next_inputs = hybrid_inputs(following.prior.q_agent.features, following.prior.advantages).numpy()
    reward = float(sum((REWARDS.get(event, 0.0) for event in events)))
    reward += GAMMA * potential(new_state) - potential(old_state)
    self.h3_round_reward += reward
    raw = Transition(hybrid_inputs(current.prior.q_agent.features, current.prior.advantages).numpy(), ACTIONS.index(action), reward, next_inputs, done)
    for transition in self.h4_queue.append(raw):
        self.h3_replay.add(transition)
        self.h3_interactions += 1
    if len(self.h3_replay) >= LEARNING_STARTS and self.h3_interactions % TRAIN_EVERY == 0:
        self.h3_losses.append(train_step(self.unibomber_hybrid.residual, self.h3_target, self.h3_optimizer, self.h3_replay.sample(BATCH_SIZE)))
        self.h3_updates += 1
        if self.h3_updates % TARGET_EVERY == 0:
            self.h3_target.load_state_dict(self.unibomber_hybrid.residual.state_dict())

# sample buffer diagnostics: correction at its limit?, its average size, spread of V(s), the override rate
def _split_diagnostics(self, sample_size: int=256) -> dict:
    if len(self.h3_replay) < sample_size:
        return {}
    batch = self.h3_replay.sample(sample_size)
    inputs = torch.as_tensor(np.stack([item.inputs for item in batch]), dtype=torch.float32)
    residual = self.unibomber_hybrid.residual
    with torch.no_grad():
        raw, value = residual.heads(inputs[..., :residual.input_dimension])
        correction = bounded_residual(raw)
    saturated = (correction.abs() >= 0.99 * RESIDUAL_LIMIT).float().mean()
    return {'saturation': float(saturated), 'correction_abs_mean': float(correction.abs().mean()), 'value_mean': float(value.mean()), 'value_std': float(value.std())}

# Framework callback after every step: passes the step to _transition.
def game_events_occurred(self, old_game_state, self_action, new_game_state, events) -> None:
    _transition(self, old_game_state, self_action, events, new_game_state, False)

# Framework callback at the end of a round: handles the final (terminal) step, logs the round reward, and every 100 rounds writes metrics and saves the checkpoint.
def end_of_round(self, last_game_state, last_action, events) -> None:
    _transition(self, last_game_state, last_action, events, None, True)
    self.h3_rewards.append(self.h3_round_reward)
    completed_reward = self.h3_round_reward
    self.h3_round_reward = 0.0
    round_number = last_game_state.get('round', 0)
    if self.h3_history_handle is not None:
        self.h3_history_handle.write(json.dumps({'round': round_number, 'reward': completed_reward, 'epsilon': self.h3_epsilon(), 'interactions': self.h3_interactions, 'updates': self.h3_updates}) + '\n')
        self.h3_history_handle.flush()
    if round_number % 100:
        return
    elapsed = perf_counter() - self.h3_started
    row = {'round': round_number, 'interactions': self.h3_interactions, 'updates': self.h3_updates, 'epsilon': self.h3_epsilon(), 'mean_round_reward': float(np.mean(self.h3_rewards)), 'mean_loss': float(np.mean(self.h3_losses)) if self.h3_losses else None, 'cpu_seconds': elapsed}
    row.update(_split_diagnostics(self))
    self.h3_metrics_handle.write(json.dumps(row) + '\n')
    self.h3_metrics_handle.flush()
    checkpoint = ROOT / MODEL_PATH
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'residual': self.unibomber_hybrid.residual.state_dict(), 'round': round_number, 'interactions': self.h3_interactions, 'config': {'gamma': GAMMA, 'device': 'cpu'}}, checkpoint)
