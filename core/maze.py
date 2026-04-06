"""Connected random maze generation (SETUP.md: core/maze.py)."""

from __future__ import annotations

import random
from typing import List, Tuple

from config import settings

Pos = Tuple[int, int]
GRID = settings.GRID


def neighbors4(x: int, y: int) -> List[Pos]:
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def reachable_count(blocked: List[List[bool]]) -> Tuple[int, int]:
    free = 0
    start = None
    for y in range(GRID):
        for x in range(GRID):
            if not blocked[y][x]:
                free += 1
                if start is None:
                    start = (x, y)
    if free == 0 or start is None:
        return 0, 0
    seen = {start}
    stack = [start]
    while stack:
        cx, cy = stack.pop()
        for nx, ny in neighbors4(cx, cy):
            if 0 <= nx < GRID and 0 <= ny < GRID and not blocked[ny][nx] and (nx, ny) not in seen:
                seen.add((nx, ny))
                stack.append((nx, ny))
    return len(seen), free


def is_fully_connected(blocked: List[List[bool]]) -> bool:
    reach, free = reachable_count(blocked)
    return free > 0 and reach == free


def generate_grid(rng: random.Random) -> List[List[bool]]:
    blocked = [[False] * GRID for _ in range(GRID)]
    target_blocks = int(GRID * GRID * settings.TARGET_BLOCK_RATIO)
    placed = 0

    for _ in range(settings.MAX_WAVES):
        if placed >= target_blocks:
            break
        candidates = [(x, y) for y in range(GRID) for x in range(GRID) if not blocked[y][x]]
        if not candidates:
            break
        rng.shuffle(candidates)
        wave_added = 0
        for x, y in candidates:
            if placed >= target_blocks:
                break
            blocked[y][x] = True
            if is_fully_connected(blocked):
                placed += 1
                wave_added += 1
            else:
                blocked[y][x] = False
        if wave_added == 0:
            break

    assert is_fully_connected(blocked), "generator bug: disconnected path"
    return blocked


def path_cells(blocked: List[List[bool]]) -> List[Pos]:
    return [(x, y) for y in range(GRID) for x in range(GRID) if not blocked[y][x]]
