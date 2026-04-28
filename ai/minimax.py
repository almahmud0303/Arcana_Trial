"""
Recursive minimax search for Agent A's equal-MP artifact-seeking mode.

Agent A maximizes its score while the opponent minimizes it.
The tree alternates between dedicated max_value and min_value functions,
simulating movement, orb collection, and immediate capture checks.
"""

from __future__ import annotations

from math import inf
from typing import List, Optional, Sequence, Tuple

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


def _legal_moves(pos: Pos, *, grid_w: int, grid_h: int, blocked: List[List[bool]]) -> List[Pos]:
    x, y = pos
    moves = [
        nb
        for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
        if 0 <= nb[0] < grid_w and 0 <= nb[1] < grid_h and not blocked[nb[1]][nb[0]]
    ]
    return moves if moves else [pos]


def _collect_orbs_at(pos: Pos, mp: int, orbs: Sequence[PowerOrb]) -> tuple[int, tuple[PowerOrb, ...]]:
    new_mp = mp
    kept: list[PowerOrb] = []
    for orb in orbs:
        if orb.pos == pos:
            new_mp += orb.value
        else:
            kept.append(orb)
    return new_mp, tuple(kept)


def _orb_potential(
    pos: Pos,
    orbs: Sequence[PowerOrb],
    *,
    grid_w: int,
    grid_h: int,
    blocked: List[List[bool]],
    step_penalty: float,
) -> float:
    if not orbs:
        return 0.0

    best = -inf
    for orb in orbs:
        dist = shortest_path_len(pos, orb.pos, grid_w=grid_w, grid_h=grid_h, blocked=blocked)
        if dist is None:
            continue
        best = max(best, orb.value - step_penalty * dist)
    return 0.0 if best == -inf else best


def _terminal_score(a_pos: Pos, b_pos: Pos, a_mp: int, b_mp: int) -> Optional[float]:
    if a_pos != b_pos:
        return None
    if a_mp > b_mp:
        return 10_000.0 + 100.0 * (a_mp - b_mp)
    if b_mp > a_mp:
        return -10_000.0 - 100.0 * (b_mp - a_mp)
    return 0.0


def _evaluate_state(
    a_pos: Pos,
    b_pos: Pos,
    a_mp: int,
    b_mp: int,
    orbs: Sequence[PowerOrb],
    *,
    grid_w: int,
    grid_h: int,
    blocked: List[List[bool]],
    step_penalty: float,
) -> float:
    terminal = _terminal_score(a_pos, b_pos, a_mp, b_mp)
    if terminal is not None:
        return terminal

    enemy_distance = abs(a_pos[0] - b_pos[0]) + abs(a_pos[1] - b_pos[1])
    orb_gain = _orb_potential(a_pos, orbs, grid_w=grid_w, grid_h=grid_h, blocked=blocked, step_penalty=step_penalty)
    orb_threat = _orb_potential(b_pos, orbs, grid_w=grid_w, grid_h=grid_h, blocked=blocked, step_penalty=step_penalty)

    mp_term = 12.0 * (a_mp - b_mp)
    spacing_term = -0.35 * enemy_distance
    orb_term = 2.0 * (orb_gain - orb_threat)
    return mp_term + spacing_term + orb_term


def max_value(
    a_pos: Pos,
    b_pos: Pos,
    a_mp: int,
    b_mp: int,
    orbs: Sequence[PowerOrb],
    *,
    grid_w: int,
    grid_h: int,
    blocked: List[List[bool]],
    depth: int,
    alpha: float,
    beta: float,
    step_penalty: float,
) -> float:
    terminal = _terminal_score(a_pos, b_pos, a_mp, b_mp)
    if terminal is not None:
        return terminal
    if depth <= 0:
        return _evaluate_state(
            a_pos,
            b_pos,
            a_mp,
            b_mp,
            orbs,
            grid_w=grid_w,
            grid_h=grid_h,
            blocked=blocked,
            step_penalty=step_penalty,
        )

    best = -inf
    for move in _legal_moves(a_pos, grid_w=grid_w, grid_h=grid_h, blocked=blocked):
        next_a_mp, next_orbs = _collect_orbs_at(move, a_mp, orbs)
        if move == b_pos:
            score = _terminal_score(move, b_pos, next_a_mp, b_mp)
            if score is None:
                score = _evaluate_state(
                    move,
                    b_pos,
                    next_a_mp,
                    b_mp,
                    next_orbs,
                    grid_w=grid_w,
                    grid_h=grid_h,
                    blocked=blocked,
                    step_penalty=step_penalty,
                )
        else:
            score = min_value(
                move,
                b_pos,
                next_a_mp,
                b_mp,
                next_orbs,
                grid_w=grid_w,
                grid_h=grid_h,
                blocked=blocked,
                depth=depth - 1,
                alpha=alpha,
                beta=beta,
                step_penalty=step_penalty,
            )

        best = max(best, score)
        alpha = max(alpha, best)
        if alpha >= beta:
            break

    return best


def min_value(
    a_pos: Pos,
    b_pos: Pos,
    a_mp: int,
    b_mp: int,
    orbs: Sequence[PowerOrb],
    *,
    grid_w: int,
    grid_h: int,
    blocked: List[List[bool]],
    depth: int,
    alpha: float,
    beta: float,
    step_penalty: float,
) -> float:
    terminal = _terminal_score(a_pos, b_pos, a_mp, b_mp)
    if terminal is not None:
        return terminal
    if depth <= 0:
        return _evaluate_state(
            a_pos,
            b_pos,
            a_mp,
            b_mp,
            orbs,
            grid_w=grid_w,
            grid_h=grid_h,
            blocked=blocked,
            step_penalty=step_penalty,
        )

    worst = inf
    for move in _legal_moves(b_pos, grid_w=grid_w, grid_h=grid_h, blocked=blocked):
        next_b_mp, next_orbs = _collect_orbs_at(move, b_mp, orbs)
        if move == a_pos:
            score = _terminal_score(a_pos, move, a_mp, next_b_mp)
            if score is None:
                score = _evaluate_state(
                    a_pos,
                    move,
                    a_mp,
                    next_b_mp,
                    next_orbs,
                    grid_w=grid_w,
                    grid_h=grid_h,
                    blocked=blocked,
                    step_penalty=step_penalty,
                )
        else:
            score = max_value(
                a_pos,
                move,
                a_mp,
                next_b_mp,
                next_orbs,
                grid_w=grid_w,
                grid_h=grid_h,
                blocked=blocked,
                depth=depth - 1,
                alpha=alpha,
                beta=beta,
                step_penalty=step_penalty,
            )

        worst = min(worst, score)
        beta = min(beta, worst)
        if alpha >= beta:
            break

    return worst


def recursive_minimax_next_move(
    my_pos: Pos,
    enemy_pos: Pos,
    my_mp: int,
    enemy_mp: int,
    orbs: List[PowerOrb],
    *,
    grid_w: int,
    grid_h: int,
    blocked: List[List[bool]],
    depth: int = 4,
    step_penalty: float = 0.35,
    prev_pos: Pos | None = None,
) -> Pos:
    """Choose Agent A's next move using a recursive min-max game tree."""
    legal = _legal_moves(my_pos, grid_w=grid_w, grid_h=grid_h, blocked=blocked)
    if len(legal) == 1:
        return legal[0]

    best_score = -inf
    best_moves: list[Pos] = []
    for move in legal:
        next_a_mp, next_orbs = _collect_orbs_at(move, my_mp, orbs)
        if move == enemy_pos:
            score = _terminal_score(move, enemy_pos, next_a_mp, enemy_mp)
            if score is None:
                score = _evaluate_state(
                    move,
                    enemy_pos,
                    next_a_mp,
                    enemy_mp,
                    next_orbs,
                    grid_w=grid_w,
                    grid_h=grid_h,
                    blocked=blocked,
                    step_penalty=step_penalty,
                )
        else:
            score = min_value(
                move,
                enemy_pos,
                next_a_mp,
                enemy_mp,
                next_orbs,
                grid_w=grid_w,
                grid_h=grid_h,
                blocked=blocked,
                depth=max(0, depth - 1),
                alpha=-inf,
                beta=inf,
                step_penalty=step_penalty,
            )

        if score > best_score + 1e-9:
            best_score = score
            best_moves = [move]
        elif abs(score - best_score) <= 1e-9:
            best_moves.append(move)

    if prev_pos is not None:
        non_uturn = [move for move in best_moves if move != prev_pos]
        if non_uturn:
            best_moves = non_uturn

    return sorted(best_moves)[0]