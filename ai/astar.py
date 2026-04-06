"""A* on a 4-neighbor grid with uniform step cost → shortest path (minimum steps)."""

from __future__ import annotations

import heapq
from typing import Callable, Dict, List, Tuple

Pos = Tuple[int, int]


def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _neighbors4(p: Pos) -> List[Pos]:
    x, y = p
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def find_path(
    start: Pos,
    goal: Pos,
    *,
    grid_w: int,
    grid_h: int,
    is_blocked: Callable[[Pos], bool],
) -> List[Pos]:
    if start == goal:
        return [start]

    def in_bounds(p: Pos) -> bool:
        x, y = p
        return 0 <= x < grid_w and 0 <= y < grid_h

    open_heap: list[tuple[int, int, Pos]] = []
    heapq.heappush(open_heap, (manhattan(start, goal), 0, start))
    came_from: Dict[Pos, Pos] = {}
    g_score: Dict[Pos, int] = {start: 0}
    seen: set[Pos] = set()

    while open_heap:
        _, g, current = heapq.heappop(open_heap)
        if current in seen:
            continue
        seen.add(current)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for nb in _neighbors4(current):
            if not in_bounds(nb) or is_blocked(nb):
                continue
            tentative = g + 1
            if tentative < g_score.get(nb, 10**9):
                came_from[nb] = current
                g_score[nb] = tentative
                f = tentative + manhattan(nb, goal)
                heapq.heappush(open_heap, (f, tentative, nb))

    return []
