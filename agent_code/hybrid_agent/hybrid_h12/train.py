import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from torch.nn import functional as F
import events as e
from agent_code.hybrid_agent.hybrid_h12.agent import ACTIONS, ANCHOR_WEIGHT, INPUT_DIMENSION, RESIDUAL_LIMIT, ResidualQNetwork, bounded_residual, residual_total

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
    def __init__(self, capacity, seed=0):
        if capacity < 1:
            raise ValueError('capacity must be positive')
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.items = []
        self.cursor = 0

    # Appends a transition; if buffer is full, rewrite oldest one
    def add(self, transition):
        if len(self.items) < self.capacity:
            self.items.append(transition)
        else:
            self.items[self.cursor] = transition
        self.cursor = (self.cursor + 1) % self.capacity

    # Draws batch_size transitions uniformly at random
    def sample(self, batch_size):
        if not self.items:
            raise ValueError('cannot sample empty replay')
        size = min(batch_size, len(self.items))
        indices = self.rng.integers(0, len(self.items), size=size)
        return [self.items[int(index)] for index in indices]

    # Number of stored transitions (to decide when training starts)
    def __len__(self):
        return len(self.items)

# From single steps to n-step transitions, for the n-step rewards
class NStepAccumulator:

    # n = number of steps per transition (3 in the final agent), gamma = discount
    def __init__(self, n, gamma):
        if n < 1 or not 0 < gamma <= 1:
            raise ValueError('n >= 1 and 0 < gamma <= 1 required')
        self.n = n
        self.gamma = gamma
        self.pending = deque()

    # Adds one step and returns completed n-step transitions
    def append(self, transition):
        self.pending.append(transition)
        return self._drain(terminal=transition.done)

    # Sums up rewards one n-step transition from the first state to the state n steps later
    def _drain(self, terminal):
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
NEIGHBOUR_DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))

# reward shaping potential: increase the closer it is to a coin (-0.20 * distance / 34) or next to a crate(+0.04 per crate). 0 after the round has ended
def potential(game_state):
    if game_state is None:
        return 0.0
    field = np.asarray(game_state['field'])
    x, y = game_state['self'][3]
    scale = max(field.shape[0] + field.shape[1], 1)
    coins = game_state.get('coins', ())
    if coins:
        nearest = min((abs(x - cx) + abs(y - cy) for cx, cy in coins))
        coin_term = -COIN_WEIGHT * nearest / scale
    else:
        coin_term = 0.0
    crate_neighbours = sum((field[x + dx, y + dy] == 1 for dx, dy in NEIGHBOUR_DELTAS))
    return float(coin_term + CRATE_ACCESS_WEIGHT * crate_neighbours)
ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = os.environ.get('BOMBERMAN_Q_AGENT_H12_MODEL', 'results/q_agent_hybrid/hybrid/checkpoints/H12_residual.pkl')
METRICS_PATH = os.environ.get('BOMBERMAN_Q_AGENT_H12_METRICS', 'results/q_agent_hybrid/hybrid/runs/H12_metrics.jsonl')
GAMMA = float(os.environ.get('Q_AGENT_H3_GAMMA', '0.99'))
LEARNING_RATE = float(os.environ.get('Q_AGENT_H3_LR', '0.0003'))
BATCH_SIZE = int(os.environ.get('Q_AGENT_H3_BATCH', '64'))
CAPACITY = int(os.environ.get('Q_AGENT_H3_CAPACITY', '100000'))
LEARNING_STARTS = int(os.environ.get('Q_AGENT_H3_LEARNING_STARTS', '1000'))
TRAIN_EVERY = int(os.environ.get('Q_AGENT_H3_TRAIN_EVERY', '2'))
TARGET_EVERY = int(os.environ.get('Q_AGENT_H3_TARGET_EVERY', '500'))
EPS_START = float(os.environ.get('Q_AGENT_H3_EPS_START', '0.20'))
EPS_FINAL = float(os.environ.get('Q_AGENT_H3_EPS_FINAL', '0.05'))
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
def train_step(online, target, optimizer, batch, gamma=GAMMA):
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
def setup_training(self):
    seed = int(os.environ.get('BOMBERMAN_SEED', '101'))
    torch.set_num_threads(int(os.environ.get('Q_AGENT_H3_CPU_THREADS', '1')))
    residual = self.unibomber_hybrid.residual
    residual.to('cpu')
    self.h3_target = ResidualQNetwork(residual.input_dimension).to('cpu')
    self.h3_target.load_state_dict(residual.state_dict())
    self.h3_replay = UniformReplay(CAPACITY, seed)
    self.h3_optimizer = torch.optim.Adam(residual.parameters(), lr=LEARNING_RATE)
    self.h4_queue = NStepAccumulator(N_STEP, GAMMA)
    self.h3_rng = np.random.default_rng(seed)
    self.h3_interactions = 0
    self.h3_updates = 0
    self.h12_decisions = 0
    self.h12_overrides = 0
    self.h3_losses = deque(maxlen=200)
    self.h3_rewards = deque(maxlen=100)
    self.h3_round_reward = 0.0
    self.h3_started = perf_counter()
    self.h3_epsilon = lambda: EPS_START + min(self.h3_interactions / EPS_DECAY, 1.0) * (EPS_FINAL - EPS_START)
    metrics = ROOT / METRICS_PATH
    metrics.parent.mkdir(parents=True, exist_ok=True)
    self.h3_metrics_handle = metrics.open('w', encoding='utf-8')
    self.h3_history_handle = None
    history_path = os.environ.get('BOMBERMAN_Q_AGENT_TRAIN_HISTORY')
    if history_path:
        history = ROOT / history_path
        history.parent.mkdir(parents=True, exist_ok=True)
        self.h3_history_handle = history.open('w', encoding='utf-8')

# compute the shaped reward, store n-step transitions in the buffer, update network parameter every n steps, copy online network to target netwrok every t steps
def _transition(self, old_state, action, events, new_state, done):
    if old_state is None or action not in ACTIONS:
        return
    current = self.unibomber_hybrid.decide(old_state)
    if new_state is None:
        next_inputs = np.zeros(INPUT_DIMENSION, dtype=np.float32)
    else:
        following = self.unibomber_hybrid.decide(new_state)
        next_inputs = np.asarray(following.inputs, dtype=np.float32)
    reward = float(sum((REWARDS.get(event, 0.0) for event in events)))
    reward += GAMMA * potential(new_state) - potential(old_state)
    self.h3_round_reward += reward
    self.h12_decisions += 1
    self.h12_overrides += int(current.action != current.prior.action)
    raw = Transition(np.asarray(current.inputs, dtype=np.float32), ACTIONS.index(action), reward, next_inputs, done)
    for transition in self.h4_queue.append(raw):
        self.h3_replay.add(transition)
        self.h3_interactions += 1
    if len(self.h3_replay) >= LEARNING_STARTS and self.h3_interactions % TRAIN_EVERY == 0:
        loss = train_step(self.unibomber_hybrid.residual, self.h3_target, self.h3_optimizer, self.h3_replay.sample(BATCH_SIZE))
        self.h3_losses.append(loss)
        self.h3_updates += 1
        if self.h3_updates % TARGET_EVERY == 0:
            self.h3_target.load_state_dict(self.unibomber_hybrid.residual.state_dict())

# sample buffer diagnostics: correction at its limit?, its average size, spread of V(s), the override rate
def _split_diagnostics(self, sample_size=256):
    if len(self.h3_replay) < sample_size:
        return {}
    batch = self.h3_replay.sample(sample_size)
    inputs = torch.as_tensor(np.stack([item.inputs for item in batch]), dtype=torch.float32)
    residual = self.unibomber_hybrid.residual
    with torch.no_grad():
        raw, value = residual.heads(inputs[..., :residual.input_dimension])
        correction = bounded_residual(raw)
    saturated = (correction.abs() >= 0.99 * RESIDUAL_LIMIT).float().mean()
    return {'saturation': float(saturated), 'correction_abs_mean': float(correction.abs().mean()), 'value_mean': float(value.mean()), 'value_std': float(value.std()), 'override_rate': self.h12_overrides / max(self.h12_decisions, 1)}

# Framework callback after every step: passes the step to _transition.
def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    _transition(self, old_game_state, self_action, events, new_game_state, False)

# Framework callback at the end of a round: handles the final (terminal) step, logs the round reward, and every 100 rounds writes metrics and saves the checkpoint.
def end_of_round(self, last_game_state, last_action, events):
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
    row = {'round': round_number, 'interactions': self.h3_interactions, 'updates': self.h3_updates, 'epsilon': self.h3_epsilon(), 'mean_round_reward': float(np.mean(self.h3_rewards)), 'mean_loss': float(np.mean(self.h3_losses)) if self.h3_losses else None, 'cpu_seconds': perf_counter() - self.h3_started}
    row.update(_split_diagnostics(self))
    self.h3_metrics_handle.write(json.dumps(row) + '\n')
    self.h3_metrics_handle.flush()
    checkpoint = ROOT / MODEL_PATH
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'residual': self.unibomber_hybrid.residual.state_dict(), 'round': round_number, 'interactions': self.h3_interactions, 'config': {'gamma': GAMMA, 'device': 'cpu'}}, checkpoint)
