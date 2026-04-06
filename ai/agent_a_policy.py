"""
Agent A: A* steps + minimax-style orb scoring.
MP rules: lower MP flees (greedy max distance); higher MP chases enemy; equal MP seeks orbs.
"""

from __future__ import annotations

import random
from typing import List, Tuple

from ai.minimax import minimax_style_pick_best_orb, next_step_toward_goal
from core.artifact import PowerOrb

Pos = Tuple[int, int]


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
        farther = [p for p in legal if dist_enemy(p) > cur_d]
        if farther:
            best = max(dist_enemy(p) for p in farther)
            group = [p for p in farther if dist_enemy(p) == best]
            return pick_no_uturn(group)
        same = [p for p in legal if dist_enemy(p) == cur_d]
        if same:
            return pick_no_uturn(same)
        best = max(dist_enemy(p) for p in legal)
        group = [p for p in legal if dist_enemy(p) == best]
        return pick_no_uturn(group)

    if my_mp > enemy_mp:
        nxt = next_step_toward_goal(my_pos, enemy_pos, grid_w=grid_w, grid_h=grid_h, blocked=blocked)
        if nxt != prev_pos or nxt == my_pos:
            return nxt
        alt = [p for p in legal if p != prev_pos]
        return rng.choice(alt if alt else legal)

    best_o = minimax_style_pick_best_orb(my_pos, orbs, grid_w=grid_w, grid_h=grid_h, blocked=blocked)
    if best_o is not None:
        nxt = next_step_toward_goal(my_pos, best_o.pos, grid_w=grid_w, grid_h=grid_h, blocked=blocked)
        if nxt != prev_pos or nxt == my_pos:
            return nxt
        alt = [p for p in legal if p != prev_pos]
        return rng.choice(alt if alt else legal)

    return pick_no_uturn(legal)
