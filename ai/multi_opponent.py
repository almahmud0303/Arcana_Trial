"""Pick a single focus rival when 3+ mages: who to flee, who to hunt, or all tied."""

from __future__ import annotations

from typing import List, Tuple

Pos = Tuple[int, int]


def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def focus_for_flee(my_pos: Pos, my_mp: int, others: List[Tuple[Pos, int]]) -> Tuple[Pos, int] | None:
    """Closest mage who strictly outranks you in MP (run away from them)."""
    stronger = [(p, m) for p, m in others if m > my_mp]
    if not stronger:
        return None
    return min(stronger, key=lambda om: manhattan(my_pos, om[0]))


def focus_for_hunt(my_pos: Pos, my_mp: int, others: List[Tuple[Pos, int]]) -> Tuple[Pos, int] | None:
    """Among strictly weaker mages, nearest one with the lowest MP (prey)."""
    weaker = [(p, m) for p, m in others if m < my_mp]
    if not weaker:
        return None
    min_mp = min(m for _, m in weaker)
    pool = [(p, m) for p, m in weaker if m == min_mp]
    return min(pool, key=lambda om: manhattan(my_pos, om[0]))


def synthetic_pair_for_planner(
    my_pos: Pos,
    my_mp: int,
    others: List[Tuple[Pos, int]],
) -> Tuple[Pos, int]:
    """
    Reduce multi-agent state to one (enemy_pos, enemy_mp) for the 2-player-style planners.
    Flee focus → threat; hunt focus → prey; otherwise any other (equal MP everyone).
    """
    if not others:
        return my_pos, my_mp

    ff = focus_for_flee(my_pos, my_mp, others)
    if ff is not None:
        return ff

    fh = focus_for_hunt(my_pos, my_mp, others)
    if fh is not None:
        return fh

    return others[0][0], others[0][1]
