"""Mage agents (SETUP.md: core/agent.py)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Tuple

from config import settings
from core.maze import path_cells

Pos = Tuple[int, int]


@dataclass
class Mage:
    label: str
    pos: Pos
    mp: int
    prev_pos: Pos | None = field(default=None, repr=False)


def commit_move(m: Mage, new_pos: Pos) -> None:
    if new_pos != m.pos:
        m.prev_pos = m.pos
        m.pos = new_pos


def spawn_two_mages(rng: random.Random, blocked: list[list[bool]]) -> tuple[Mage, Mage]:
    cells = path_cells(blocked)
    a = rng.choice(cells)
    g = settings.GRID
    min_sep = (g + g) // 4

    def dist(p: Pos, q: Pos) -> int:
        return abs(p[0] - q[0]) + abs(p[1] - q[1])

    far = [c for c in cells if dist(c, a) >= min_sep]
    if far:
        b = rng.choice(far)
    else:
        b = max(cells, key=lambda c: dist(c, a))
        if b == a and len(cells) > 1:
            b = max((c for c in cells if c != a), key=lambda c: dist(c, a))
    return Mage("A", a, settings.START_MP), Mage("B", b, settings.START_MP)
