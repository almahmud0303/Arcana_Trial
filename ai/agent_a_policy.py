"""
Agent A: A* movement plus recursive min-max artifact search.
MP rules: lower MP flees while looking for power, higher MP chases enemy, equal MP uses minimax.
"""

from __future__ import annotations

import random
from typing import List, Tuple

from ai.minimax import next_step_toward_goal, recursive_minimax_next_move, shortest_path_len
from core.artifact import PowerOrb

Pos = Tuple[int, int]


def _best_orb_score(
    pos: Pos,
    orbs: List[PowerOrb],
    *,
    grid_w: int,
    grid_h: int,
    blocked: List[List[bool]],
    step_penalty: float = 0.35,
) -> float:
    if not orbs:
        return 0.0

    best = -1e18
    for orb in orbs:
        dist = shortest_path_len(pos, orb.pos, grid_w=grid_w, grid_h=grid_h, blocked=blocked)
        if dist is None:
            continue
        best = max(best, orb.value - step_penalty * dist)
    return 0.0 if best == -1e18 else best


def plan_move_agent_a(
    my_pos: Pos,
    enemy_pos: Pos,
    my_mp: int,
    enemy_mp: int,
    orbs: List[PowerOrb],
    blocked: List[List[bool]],
    *,
    grid_w: int,
    grid_h: int,
    rng: random.Random,
    prev_pos: Pos | None,
) -> Pos:
    x, y = my_pos
    legal = [
        nb
        for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
        if 0 <= nb[0] < grid_w and 0 <= nb[1] < grid_h and not blocked[nb[1]][nb[0]]
    ]
    if not legal:
        return my_pos

    def dist_enemy(p: Pos) -> int:
        return abs(p[0] - enemy_pos[0]) + abs(p[1] - enemy_pos[1])

    cur_d = dist_enemy(my_pos)

    def pick_no_uturn(cands: List[Pos]) -> Pos:
        if prev_pos is None:
            return rng.choice(cands)
        alt = [c for c in cands if c != prev_pos]
        return rng.choice(alt if alt else cands)

    if my_mp < enemy_mp:
        def escape_power_score(p: Pos) -> float:
            enemy_distance = dist_enemy(p)
            orb_score = _best_orb_score(p, orbs, grid_w=grid_w, grid_h=grid_h, blocked=blocked)
            closer_penalty = 4.0 if enemy_distance < cur_d else 0.0
            return enemy_distance * 2.0 + orb_score - closer_penalty

        farther = [p for p in legal if dist_enemy(p) > cur_d]
        if farther:
            best = max(escape_power_score(p) for p in farther)
            group = [p for p in farther if escape_power_score(p) == best]
            return pick_no_uturn(group)
        same = [p for p in legal if dist_enemy(p) == cur_d]
        if same:
            best = max(escape_power_score(p) for p in same)
            group = [p for p in same if escape_power_score(p) == best]
            return pick_no_uturn(group)
        best = max(escape_power_score(p) for p in legal)
        group = [p for p in legal if escape_power_score(p) == best]
        return pick_no_uturn(group)

    if my_mp > enemy_mp:
        nxt = next_step_toward_goal(my_pos, enemy_pos, grid_w=grid_w, grid_h=grid_h, blocked=blocked)
        if nxt != prev_pos or nxt == my_pos:
            return nxt
        alt = [p for p in legal if p != prev_pos]
        return rng.choice(alt if alt else legal)

    nxt = recursive_minimax_next_move(
        my_pos,
        enemy_pos,
        my_mp,
        enemy_mp,
        orbs,
        grid_w=grid_w,
        grid_h=grid_h,
        blocked=blocked,
        depth=4,
        prev_pos=prev_pos,
    )
    if nxt != prev_pos or nxt == my_pos:
        return nxt
    alt = [p for p in legal if p != prev_pos]
    return rng.choice(alt if alt else legal)
