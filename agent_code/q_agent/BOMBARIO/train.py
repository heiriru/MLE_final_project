
import os
import json
import pickle
from typing import List
from collections import deque      # fuer den n-step-Puffer

import numpy as np

import events as e
### import functions from callback.py
from .callbacks import (
    MODEL_FILE, state_to_features, suggested_action, ACTIONS, MOVE_ACTIONS,
)



ALPHA = float(os.environ.get("BM_ALPHA", "0.05"))               # learning rate
GAMMA = 0.9               # discount rate
EPS_START = float(os.environ.get("BM_EPS_START", "0.3"))        # epsilon at the start of each phase
EPS_END = float(os.environ.get("BM_EPS_END", "0.05"))
EPS_DECAY_ROUNDS = int(os.environ.get("BM_EPS_DECAY", "1000"))  # epsilon decay rate
USE_SYMMETRY = True
N_STEP = 5                #  n-step Returns 





BOMB_WARMUP_ROUNDS = 300  # bomb rewards start only after this many rounds

STATS_FILE = "training_stats.json"
META_FILE = "train_meta.json"   # saves the cumulative round counter across phases to let epsilon decay over multiple phases

#### symmetries, use permutation of the directions
SYMMETRIES = [
    ([0, 1, 2, 3], [0, 1, 2, 3, 4, 5]),   # identity
    ([0, 3, 2, 1], [0, 3, 2, 1, 4, 5]),   # right <-> left
    ([2, 1, 0, 3], [2, 1, 0, 3, 4, 5]),   # up <-> down
    ([2, 3, 0, 1], [2, 3, 0, 1, 4, 5]),   # 180° turn
]


### transform only the first 4 features, depending on the permutation
def _transform_state(state, perm):
    
    return (state[perm[0]], state[perm[1]], state[perm[2]], state[perm[3]],
            state[4], state[5])

### epsilon from the cumulitive round count (so it keeps decaying across phases)
def _epsilon_for(n):
    return max(EPS_END, EPS_START - (EPS_START - EPS_END) * n / EPS_DECAY_ROUNDS)


### setup all list
def setup_training(self):

    here = os.path.dirname(__file__)
    meta_path = os.path.join(here, META_FILE)
    if len(self.q_table) == 0:
        self._total_rounds = 0                       # fresh start (q-table was deleted)
    elif os.path.isfile(meta_path):
        with open(meta_path) as f:
            self._total_rounds = json.load(f).get("total_rounds", 0)
    else:
        self._total_rounds = 0

    self.epsilon = EPS_START          # each phase starts with a moderate exploration rate
    self.nstep_buffer = deque()       
    self.round_reward = 0.0
    self._coins = self._invalid = self._kills = 0
    self.stats = {"round": [], "reward": [], "coins": [], "steps": [],
                  "invalid": [], "kills": [], "epsilon": []}
    self.logger.info(f"Training is ready (total rounds so far: {self._total_rounds})")

### Returns a list with all custom events that occured between game steps 
def _custom_events(s_old, s_new, action, old_game_state, round_no):

    ev = []
    if s_old is None or s_new is None:
        return ev

### if bfs action was followed give the rewards for this
    suggestion = suggested_action(old_game_state)
    if action in MOVE_ACTIONS:
        if action == suggestion:
            ev.append(FOLLOWED_SUGGESTION)
        elif suggestion in MOVE_ACTIONS:
            ev.append(IGNORED_SUGGESTION)

### if moved into or out of danger (use feature 5)
    was_danger = s_old[4] == 4
    now_danger = s_new[4] == 4
    if not was_danger and now_danger:
        ev.append(MOVED_INTO_DANGER)
    elif was_danger and not now_danger:
        ev.append(MOVED_OUT_OF_DANGER)
    elif was_danger and now_danger:
        ev.append(STAYED_IN_DANGER)

### Did Bomb dropping make sense? (use feature 5)
### bomb incentives (and the "wasted spot" penalty) only after warmup
    if round_no >= BOMB_WARMUP_ROUNDS:
        if action == "BOMB":
            ev.append(GOOD_BOMB if 1 <= s_old[4] <= 3 else USELESS_BOMB)
        elif s_old[4] in (1, 2, 3):
            ev.append(WASTED_GOOD_SPOT)   # stood on a good bomb spot but did not bomb

### useless waiting if not in danger 
    if action == "WAIT" and not was_danger:
        ev.append(WAITED_NEEDLESSLY)

    return ev


### look up if state is in q table, otherwise return empty Actions array
def _q_get(self, state):
    
    if state not in self.q_table:
        self.q_table[state] = np.zeros(len(ACTIONS))
    return self.q_table[state]



### we choose to include symmetrie simply by learning from each step the 4 symmetry transformed steps, 
### so we dont reduce the size of the q table, but we learn 4 times faster during training
### implement the bellman equation here
def _q_update(self, s_old, action, reward, s_new):
    
    if s_old is None:
        return
    a = ACTIONS.index(action)
    symmetries = SYMMETRIES if USE_SYMMETRY else SYMMETRIES[:1]

### transform the state and the action and update q for every transformed state
    for perm, act_perm in symmetries:
        ss = _transform_state(s_old, perm)
        aa = act_perm[a]
        
        if s_new is None:
            target = reward
        else:
            ssn = _transform_state(s_new, perm)
            target = reward + GAMMA * np.max(_q_get(self, ssn))
        q = _q_get(self, ss)
        q[aa] += ALPHA * (target - q[aa])


## the n-step learning step
def _apply_nstep_target(self, s, action, target):
    if s is None:
        return
    a = ACTIONS.index(action)
    symmetries = SYMMETRIES if USE_SYMMETRY else SYMMETRIES[:1]
    for perm, act_perm in symmetries:
        ss = _transform_state(s, perm)
        aa = act_perm[a]
        q = _q_get(self, ss)
        q[aa] += ALPHA * (target - q[aa])


## the n_step update rule, see documentation for more information
def _nstep_update(self, bootstrap_state):
    G = 0.0
    for k, (_, _, r) in enumerate(self.nstep_buffer):
        G += (GAMMA ** k) * r
    if bootstrap_state is not None:
        m = len(self.nstep_buffer)
        G += (GAMMA ** m) * np.max(_q_get(self, bootstrap_state))
    s0, a0, _ = self.nstep_buffer[0]
    _apply_nstep_target(self, s0, a0, G)
    self.nstep_buffer.popleft()


### calculate the reward after the next step and update q
def game_events_occurred(self, old_game_state, self_action,
                         new_game_state, events: List[str]):
    
    s_old = state_to_features(old_game_state)
    s_new = state_to_features(new_game_state)

    events = list(events) + _custom_events(s_old, s_new, self_action, old_game_state,
                                           self._total_rounds)
    ## crate Combo for more crates
    n_crates = events.count(e.CRATE_DESTROYED)
    if n_crates >= 2:
        events += [CRATE_COMBO] * (n_crates - 1)
    ### calculate all the rewards, with custom waits, see below
    reward = reward_from_events(self.logger, events)

    self.round_reward += reward

    ### update stats
    if e.COIN_COLLECTED in events:
        self._coins += 1
    if e.INVALID_ACTION in events:
        self._invalid += 1
    if e.KILLED_OPPONENT in events:
        self._kills += 1

    # the nstep buffer
    self.nstep_buffer.append((s_old, self_action, reward))
    if len(self.nstep_buffer) >= N_STEP:
        _nstep_update(self, s_new)

### end of round: update stats, reduce epsilon,
def end_of_round(self, last_game_state, last_action, events: List[str]):
    
    events = list(events)
    reward = reward_from_events(self.logger, events)
    self.round_reward += reward
    if e.KILLED_OPPONENT in events:
        self._kills += 1

    s_last = state_to_features(last_game_state)
   
    self.nstep_buffer.append((s_last, last_action, reward))
    while self.nstep_buffer:
        _nstep_update(self, None)

    ### reduce epsilon for next episode - decays within the phase (per-phase round),
    ### while _total_rounds (cumulative) is only used for the one-time bomb warmup
    self._total_rounds += 1
    rnd = last_game_state["round"]
    self.epsilon = _epsilon_for(rnd)

    # Statistik dieser Runde festhalten
    self.stats["round"].append(rnd)
    self.stats["reward"].append(self.round_reward)
    self.stats["coins"].append(self._coins)
    self.stats["steps"].append(last_game_state["step"])
    self.stats["invalid"].append(self._invalid)
    self.stats["kills"].append(self._kills)
    self.stats["epsilon"].append(self.epsilon)
    if rnd % 50 == 0:
        avg = np.mean(self.stats["reward"][-50:])
        self.logger.info(f"Runde {rnd}: Ø-Belohnung (50) = {avg:.2f}, "
                         f"epsilon = {self.epsilon:.3f}, Zustaende = {len(self.q_table)}")

    ### reset for next round
    self.round_reward = 0.0
    self._coins = self._invalid = self._kills = 0

    ### save everything
    here = os.path.dirname(__file__)
    with open(os.path.join(here, MODEL_FILE), "wb") as f:
        pickle.dump(self.q_table, f)
    with open(os.path.join(here, STATS_FILE), "w") as f:
        json.dump(self.stats, f)
    with open(os.path.join(here, META_FILE), "w") as f:
        json.dump({"total_rounds": self._total_rounds}, f)


### Reward shaping

FOLLOWED_SUGGESTION = "FOLLOWED_SUGGESTION"   
IGNORED_SUGGESTION = "IGNORED_SUGGESTION"     
MOVED_INTO_DANGER = "MOVED_INTO_DANGER"       
MOVED_OUT_OF_DANGER = "MOVED_OUT_OF_DANGER"   
STAYED_IN_DANGER = "STAYED_IN_DANGER"         
GOOD_BOMB = "GOOD_BOMB"
USELESS_BOMB = "USELESS_BOMB"
WAITED_NEEDLESSLY = "WAITED_NEEDLESSLY"
WASTED_GOOD_SPOT = "WASTED_GOOD_SPOT"         # to avoid loops, the agent needs to be punished for not dropping bombs!
CRATE_COMBO = "CRATE_COMBO"                   # extra bonos to attract multiple crate destruction

## Rewards
REWARDS = {
    ### real rewards
    e.COIN_COLLECTED: 9.0,   
    e.KILLED_OPPONENT: 7.0,   
    e.CRATE_DESTROYED: 0.4,   
    CRATE_COMBO: 0.2,   
    e.COIN_FOUND: 0.5,

    
    e.KILLED_SELF: -6.0,
    e.GOT_KILLED: -6.0,
    e.INVALID_ACTION: -0.5,
    e.SURVIVED_ROUND: 0,   

### custom rewards
    FOLLOWED_SUGGESTION: 0.1,
    IGNORED_SUGGESTION: -0.3,
    MOVED_INTO_DANGER: -1.0,
    MOVED_OUT_OF_DANGER: 1.0,   
    STAYED_IN_DANGER: -0.5,   
    GOOD_BOMB: 2,
    USELESS_BOMB: -0.5,
    WAITED_NEEDLESSLY: -0.4,
    WASTED_GOOD_SPOT: -0.7,   
}

### sum all rewards of round
def reward_from_events(logger, events) -> float:
    
    total = sum(REWARDS.get(ev, 0.0) for ev in events)
    if logger is not None:
        logger.info(f"Belohnung {total:+.2f} fuer: {', '.join(events)}")
    return total
