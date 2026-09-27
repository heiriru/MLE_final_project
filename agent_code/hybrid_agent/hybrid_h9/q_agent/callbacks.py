import os
import pickle
from collections import deque
import numpy as np
from settings import BOMB_POWER, BOMB_TIMER
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MOVE_ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
EPSILON = 0.1
MODEL_FILE = 'q_table.pkl'
PRINT = 0

def setup(self):
    path = os.path.join(os.path.dirname(__file__), MODEL_FILE)
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            self.q_table = pickle.load(f)
        self.logger.info(f'Q-Table geladen ({len(self.q_table)} Zustaende).')
    else:
        self.q_table = {}
        self.logger.info('Neue, leere Q-Table angelegt.')

def act(self, game_state: dict) -> str:
    features = state_to_features(game_state)
    epsilon = getattr(self, 'epsilon', EPSILON)
    if self.train and np.random.rand() < epsilon:
        return np.random.choice(ACTIONS)
    if features in self.q_table:
        q = self.q_table[features]
        if np.ptp(q) < 1e-06:
            action = suggested_action(game_state)
        else:
            action = ACTIONS[int(np.argmax(q))]
    else:
        self.q_table[features] = np.zeros(len(ACTIONS))
        action = suggested_action(game_state)
    if PRINT:
        print('FEATURES', features, 'ACTION', action)
    return action

def state_to_features(game_state: dict) -> tuple:
    if game_state is None:
        return None
    field = game_state['field']
    pos = game_state['self'][3]
    others = [o[3] for o in game_state['others']]
    bombs = [b[0] for b in game_state['bombs']]
    danger = bomb_danger(game_state)
    mode = game_mode(game_state)
    good_dir = good_direction(game_state, danger, mode)
    x1 = neighbor_value(field, danger, others, bombs, pos, 0, good_dir)
    x2 = neighbor_value(field, danger, others, bombs, pos, 1, good_dir)
    x3 = neighbor_value(field, danger, others, bombs, pos, 2, good_dir)
    x4 = neighbor_value(field, danger, others, bombs, pos, 3, good_dir)
    x5 = current_field_value(game_state, danger)
    x6 = mode
    return (x1, x2, x3, x4, x5, x6)

def blast_coords(field, x, y):
    coords = [(x, y)]
    for dx, dy in DIRECTIONS:
        for step in range(1, BOMB_POWER + 1):
            nx, ny = (x + dx * step, y + dy * step)
            if field[nx, ny] == -1:
                break
            coords.append((nx, ny))
    return coords

def bomb_danger(game_state):
    field = game_state['field']
    danger = np.zeros_like(field, dtype=np.int16)
    danger[game_state['explosion_map'] > 0] = 1
    for (bx, by), t in game_state['bombs']:
        steps = t + 1
        for cx, cy in blast_coords(field, bx, by):
            if danger[cx, cy] == 0 or steps < danger[cx, cy]:
                danger[cx, cy] = steps
    return danger

def _deadly_now(danger):
    return {(i, j) for i in range(danger.shape[0]) for j in range(danger.shape[1]) if danger[i, j] == 1}

def walkable(field, others, bombs, x, y):
    return field[x, y] == 0 and (x, y) not in others and ((x, y) not in bombs)

def game_mode(game_state):
    if game_state['coins']:
        return 0
    if np.any(game_state['field'] == 1):
        return 1
    return 2

def _bfs_next_step(field, start, targets, blocked):
    if not targets:
        return -1
    targets = set(targets)
    if start in targets:
        return -1
    queue = deque([(start[0], start[1], None)])
    visited = {start}
    while queue:
        x, y, first = queue.popleft()
        if (x, y) in targets and first is not None:
            return first
        for i, (dx, dy) in enumerate(DIRECTIONS):
            nxt = (x + dx, y + dy)
            if nxt in visited or nxt in blocked or field[nxt[0], nxt[1]] != 0:
                continue
            visited.add(nxt)
            queue.append((nxt[0], nxt[1], i if first is None else first))
    return -1

def get_path_bfs_safe_tile(game_state, danger):
    field = game_state['field']
    pos = game_state['self'][3]
    others = [o[3] for o in game_state['others']]
    bombs = [b[0] for b in game_state['bombs']]
    deadly_now = _deadly_now(danger)
    safe = {(i, j) for i in range(field.shape[0]) for j in range(field.shape[1]) if field[i, j] == 0 and danger[i, j] == 0}
    return _bfs_next_step(field, pos, safe, set(others) | set(bombs) | deadly_now)

def _adjacent_free(field, cells):
    result = set()
    for cx, cy in cells:
        for dx, dy in DIRECTIONS:
            nx, ny = (int(cx) + dx, int(cy) + dy)
            if field[nx, ny] == 0:
                result.add((nx, ny))
    return result

def can_place_bomb(game_state):
    return game_state['self'][2]

def can_escape_after_bomb(field, pos, blocked, danger=None):
    blast = set(blast_coords(field, *pos))
    queue = deque([(pos, 0)])
    seen = {pos}
    while queue:
        (cx, cy), dist = queue.popleft()
        if (cx, cy) not in blast and (danger is None or danger[cx, cy] == 0):
            return True
        if dist >= BOMB_TIMER:
            continue
        for dx, dy in DIRECTIONS:
            nxt = (cx + dx, cy + dy)
            if nxt in seen or nxt in blocked or field[nxt[0], nxt[1]] != 0:
                continue
            seen.add(nxt)
            queue.append((nxt, dist + 1))
    return False

def _bomb_hits(field, others, x, y):
    hits = 0
    for cx, cy in blast_coords(field, x, y):
        if field[cx, cy] == 1 or (cx, cy) in others:
            hits += 1
    return hits

def _dist_to_crate(field, start, blocked):
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        (x, y), d = queue.popleft()
        for dx, dy in DIRECTIONS:
            nx, ny = (x + dx, y + dy)
            if (nx, ny) in seen or field[nx, ny] == -1 or (nx, ny) in blocked:
                continue
            if field[nx, ny] == 1:
                return d + 1
            seen.add((nx, ny))
            queue.append(((nx, ny), d + 1))
    return None

def good_direction(game_state, danger, mode):
    field = game_state['field']
    pos = game_state['self'][3]
    others = [o[3] for o in game_state['others']]
    bombs = [b[0] for b in game_state['bombs']]
    blocked = set(others) | set(bombs) | _deadly_now(danger)
    if danger[pos] > 0:
        d = get_path_bfs_safe_tile(game_state, danger)
        if d != -1:
            return d
    if mode == 0:
        d = _bfs_next_step(field, pos, game_state['coins'], blocked)
        if d != -1:
            return d
    if mode == 2:
        enemy_blast = set()
        for ex, ey in others:
            enemy_blast |= set(blast_coords(field, ex, ey))
        d = _bfs_next_step(field, pos, _adjacent_free(field, others), blocked)
        if d != -1:
            ddx, ddy = DIRECTIONS[d]
            if (pos[0] + ddx, pos[1] + ddy) not in enemy_blast:
                return d
    cur_hits = _bomb_hits(field, others, pos[0], pos[1])
    best_dir, best_hits = (-1, cur_hits)
    for j, (dx, dy) in enumerate(DIRECTIONS):
        nx, ny = (pos[0] + dx, pos[1] + dy)
        if not walkable(field, others, bombs, nx, ny) or danger[nx, ny] != 0:
            continue
        h = _bomb_hits(field, others, nx, ny)
        if h > best_hits:
            best_hits, best_dir = (h, j)
    if best_dir != -1:
        return best_dir
    here = _dist_to_crate(field, pos, blocked)
    best_dir, best_dist = (-1, here)
    for j, (dx, dy) in enumerate(DIRECTIONS):
        nx, ny = (pos[0] + dx, pos[1] + dy)
        if not walkable(field, others, bombs, nx, ny) or danger[nx, ny] != 0:
            continue
        dc = _dist_to_crate(field, (nx, ny), blocked)
        if dc is None:
            continue
        if best_dist is None or dc < best_dist:
            best_dist, best_dir = (dc, j)
    return best_dir

def neighbor_value(field, danger, others, bombs, pos, direction, good_dir):
    dx, dy = DIRECTIONS[direction]
    nx, ny = (pos[0] + dx, pos[1] + dy)
    if not walkable(field, others, bombs, nx, ny):
        return 2
    if danger[nx, ny] == 1:
        return 2
    if direction == good_dir:
        return 1
    if danger[nx, ny] > 1:
        return 3
    return 0

def current_field_value(game_state, danger):
    field = game_state['field']
    pos = game_state['self'][3]
    others = [o[3] for o in game_state['others']]
    bombs = [b[0] for b in game_state['bombs']]
    if danger[pos] > 0:
        return 4
    if not can_place_bomb(game_state):
        return 0
    hits = sum((1 for cx, cy in blast_coords(field, *pos) if field[cx, cy] == 1 or (cx, cy) in others))
    if hits == 0:
        return 0
    if not can_escape_after_bomb(field, pos, set(others) | set(bombs), danger):
        return 0
    if hits >= 6:
        return 3
    if hits >= 3:
        return 2
    return 1

def suggested_action(game_state) -> str:
    if game_state is None:
        return 'WAIT'
    danger = bomb_danger(game_state)
    good_dir = good_direction(game_state, danger, game_mode(game_state))
    if good_dir in (0, 1, 2, 3):
        return MOVE_ACTIONS[good_dir]
    if 1 <= current_field_value(game_state, danger) <= 3:
        return 'BOMB'
    return 'WAIT'
