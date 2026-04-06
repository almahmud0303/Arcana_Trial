"""Power pickups on the grid (SETUP.md: core/artifact.py)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

Pos = Tuple[int, int]


@dataclass(frozen=True)
class PowerOrb:
    pos: Pos
    value: int
