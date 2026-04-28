"""
Target choice for agent A: maximize magic gained while minimizing travel.

Pure shortest path is computed with A* (minimum number of steps on this grid).

The *selection* among several power-ups uses a min–max style objective:
  score(pickup) = value − λ × (shortest_path_length)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ai.astar import find_path
from core.artifact import PowerOrb

Pos = Tuple[int, int]


def shortest_path_len(
    start: Pos,
    goal: Pos,
    *,
    grid_w: int,
    grid_h: int,
    blocked: List[List[bool]],
) -> Optional[int]:
    def is_blocked(p: Pos) -> bool:
        x, y = p
        return blocked[y][x]

    path = find_path(start, goal, grid_w=grid_w, grid_h=grid_h, is_blocked=is_blocked)
    if not path:
        return None
    return len(path) - 1


def minimax_style_pick_best_orb(
    agent_pos: Pos,
    orbs: List[PowerOrb],
    *,
    grid_w: int,
    grid_h: int,
    blocked: List[List[bool]],
    step_penalty: float = 0.35,
) -> Optional[PowerOrb]:
    """MAX over pickups of (value − step_penalty × shortest_steps)."""
    if not orbs:
        return None
    best: Optional[PowerOrb] = None
    best_score = -1e18
    for o in orbs:
        dist = shortest_path_len(agent_pos, o.pos, grid_w=grid_w, grid_h=grid_h, blocked=blocked)
        if dist is None:
            continue
        score = o.value - step_penalty * dist
        if score > best_score:
            best_score = score
            best = o
    return best


def legal_moves(
    pos: Pos,
    *,
    grid_w: int,
    grid_h: int,
    blocked: List[List[bool]],
) -> List[Pos]:
    x, y = pos
    out: List[Pos] = []
    for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
        if 0 <= nx < grid_w and 0 <= ny < grid_h and not blocked[ny][nx]:
            out.append((nx, ny))
    return out if out else [pos]


def _collect_at(pos: Pos, mp: int, orbs: List[PowerOrb]) -> tuple[int, List[PowerOrb]]:
    new_mp = mp
    kept: List[PowerOrb] = []
    for orb in orbs:
        if orb.pos == pos:
            new_mp += orb.value
        else:
            kept.append(orb)
    return new_mp, kept


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


def _manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _evaluate_for_agent_a(
    a_pos: Pos,
    b_pos: Pos,
    a_mp: int,
    b_mp: int,
    orbs: List[PowerOrb],
    *,
    grid_w: int,
    grid_h: int,
    blocked: List[List[bool]],
) -> float:
    if a_pos == b_pos:
        if a_mp > b_mp:
            return 1000.0
        if b_mp > a_mp:
            return -1000.0
        return -4.0

    distance = _manhattan(a_pos, b_pos)
    mp_score = (a_mp - b_mp) * 12.0
    orb_score = _best_orb_score(a_pos, orbs, grid_w=grid_w, grid_h=grid_h, blocked=blocked)
    enemy_orb_score = _best_orb_score(b_pos, orbs, grid_w=grid_w, grid_h=grid_h, blocked=blocked)

    if a_mp > b_mp:
        position_score = 10.0 - distance
    elif a_mp < b_mp:
        position_score = float(distance)
    else:
        position_score = 0.25 * distance

    return mp_score + orb_score - enemy_orb_score + position_score


def minimax_decide_move(
    a_pos: Pos,
    b_pos: Pos,
    a_mp: int,
    b_mp: int,
    orbs: List[PowerOrb],
    *,
    grid_w: int,
    grid_h: int,
    blocked: List[List[bool]],
    depth: int = 4,
) -> Pos:
    """Classical depth-limited minimax move for agent A."""

    def minimax(
        cur_a_pos: Pos,
        cur_b_pos: Pos,
        cur_a_mp: int,
        cur_b_mp: int,
        cur_orbs: List[PowerOrb],
        remaining_depth: int,
        maximizing: bool,
    ) -> float:
        if remaining_depth == 0 or cur_a_pos == cur_b_pos:
            return _evaluate_for_agent_a(
                cur_a_pos,
                cur_b_pos,
                cur_a_mp,
                cur_b_mp,
                cur_orbs,
                grid_w=grid_w,
                grid_h=grid_h,
                blocked=blocked,
            )

        if maximizing:
            best = -1e18
            for move in legal_moves(cur_a_pos, grid_w=grid_w, grid_h=grid_h, blocked=blocked):
                next_a_mp, next_orbs = _collect_at(move, cur_a_mp, cur_orbs)
                score = minimax(move, cur_b_pos, next_a_mp, cur_b_mp, next_orbs, remaining_depth - 1, False)
                best = max(best, score)
            return best

        best = 1e18
        for move in legal_moves(cur_b_pos, grid_w=grid_w, grid_h=grid_h, blocked=blocked):
            next_b_mp, next_orbs = _collect_at(move, cur_b_mp, cur_orbs)
            score = minimax(cur_a_pos, move, cur_a_mp, next_b_mp, next_orbs, remaining_depth - 1, True)
            best = min(best, score)
        return best

    best_move = a_pos
    best_score = -1e18
    for move in legal_moves(a_pos, grid_w=grid_w, grid_h=grid_h, blocked=blocked):
        next_a_mp, next_orbs = _collect_at(move, a_mp, orbs)
        score = minimax(move, b_pos, next_a_mp, b_mp, next_orbs, depth - 1, False)
        if score > best_score:
            best_score = score
            best_move = move
    return best_move


def next_step_toward_goal(
    agent_pos: Pos,
    goal: Pos,
    *,
    grid_w: int,
    grid_h: int,
    blocked: List[List[bool]],
) -> Pos:
    """One step along an A* shortest path (minimum grid moves)."""

    def is_blocked(p: Pos) -> bool:
        x, y = p
        return blocked[y][x]

    path = find_path(agent_pos, goal, grid_w=grid_w, grid_h=grid_h, is_blocked=is_blocked)
    if len(path) >= 2:
        return path[1]
    return agent_pos
