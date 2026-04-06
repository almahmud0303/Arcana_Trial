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
