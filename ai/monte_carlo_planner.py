"""
Monte Carlo rollouts for agent B: sample many futures after each candidate first move,
average a heuristic score, pick the best first move.

Enemy in rollouts uses a simple random walk (stochastic environment).
Reward matches the active mode: flee (distance + power access), hunt (chase weaker), equal (orb proximity).
"""

from __future__ import annotations

import random
from typing import List, Literal, Tuple

Pos = Tuple[int, int]
Mode = Literal["flee", "hunt", "equal"]


def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def legal_moves(pos: Pos, blocked: List[List[bool]], w: int, h: int) -> List[Pos]:
    x, y = pos
    out: List[Pos] = []
    for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
        if 0 <= nx < w and 0 <= ny < h and not blocked[ny][nx]:
            out.append((nx, ny))
    return out if out else [pos]


def _mode_from_mp(my_mp: int, enemy_mp: int) -> Mode:
    if my_mp < enemy_mp:
        return "flee"
    if my_mp > enemy_mp:
        return "hunt"
    return "equal"


def _best_orb_score(pos: Pos, orbs: list, step_penalty: float = 0.35) -> float:
    if not orbs:
        return 0.0
    return max(float(o.value) - step_penalty * manhattan(pos, o.pos) for o in orbs)


def _rollout_score(
    my_start: Pos,
    enemy_start: Pos,
    *,
    mode: Mode,
    my_mp: int,
    enemy_mp: int,
    orbs: list,
    blocked: List[List[bool]],
    w: int,
    h: int,
    rng: random.Random,
    depth: int,
) -> float:
    myp, enp = my_start, enemy_start
    for _ in range(depth):
        en_legal = legal_moves(enp, blocked, w, h)
        enp = rng.choice(en_legal)
        my_legal = legal_moves(myp, blocked, w, h)
        myp = rng.choice(my_legal)

    if mode == "flee":
        distance = manhattan(myp, enp)
        capture_penalty = 50.0 if myp == enp and my_mp < enemy_mp else 0.0
        return distance * 2.0 + _best_orb_score(myp, orbs) - capture_penalty
    if mode == "hunt":
        d = manhattan(myp, enp)
        if myp == enp and my_mp > enemy_mp:
            return 200.0 - d
        return float(-d)
    # equal: get closer to best orb
    if not orbs:
        return 0.0
    best = min(manhattan(myp, o.pos) for o in orbs)
    return float(-best)


def monte_carlo_plan_move(
    my_pos: Pos,
    enemy_pos: Pos,
    my_mp: int,
    enemy_mp: int,
    orbs: list,
    blocked: List[List[bool]],
    *,
    grid_w: int,
    grid_h: int,
    rng: random.Random,
    rollouts: int = 56,
    depth: int = 14,
) -> Pos:
    mode = _mode_from_mp(my_mp, enemy_mp)
    moves = legal_moves(my_pos, blocked, grid_w, grid_h)
    if len(moves) == 1:
        return moves[0]

    best_m = moves[0]
    best_avg = -1e18
    for m in moves:
        total = 0.0
        for _ in range(rollouts):
            total += _rollout_score(
                m,
                enemy_pos,
                mode=mode,
                my_mp=my_mp,
                enemy_mp=enemy_mp,
                orbs=orbs,
                blocked=blocked,
                w=grid_w,
                h=grid_h,
                rng=rng,
                depth=depth,
            )
        avg = total / rollouts
        if avg > best_avg:
            best_avg = avg
            best_m = m
    return best_m
