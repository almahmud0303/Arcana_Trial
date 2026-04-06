"""Spawns, pickups, duel rules, new match (SETUP.md: core/game_loop.py)."""

from __future__ import annotations

import random
from typing import List, Set, Tuple

from config import settings
from core.agent import Mage, commit_move, spawn_two_mages
from core.artifact import PowerOrb
from core.maze import generate_grid, neighbors4, path_cells

Pos = Tuple[int, int]


def spawn_orbs(
    rng: random.Random,
    blocked: List[List[bool]],
    forbidden: Set[Pos],
    count: int,
) -> List[PowerOrb]:
    cells = [c for c in path_cells(blocked) if c not in forbidden]
    rng.shuffle(cells)
    orbs: List[PowerOrb] = []
    for i in range(min(count, len(cells))):
        x, y = cells[i]
        orbs.append(PowerOrb(pos=(x, y), value=rng.choice(settings.POWER_VALUES)))
    return orbs


def respawn_orbs_fill(
    rng: random.Random,
    blocked: List[List[bool]],
    m_a: Mage,
    m_b: Mage,
    orbs: List[PowerOrb],
) -> List[PowerOrb]:
    while len(orbs) < settings.MAX_ORBS_ON_FIELD:
        occ = {o.pos for o in orbs} | {m_a.pos, m_b.pos}
        cells = [c for c in path_cells(blocked) if c not in occ]
        if not cells:
            break
        x, y = rng.choice(cells)
        orbs.append(PowerOrb(pos=(x, y), value=rng.choice(settings.POWER_VALUES)))
    return orbs


def collect_for_mage(m: Mage, orbs: List[PowerOrb]) -> List[PowerOrb]:
    kept: List[PowerOrb] = []
    for o in orbs:
        if o.pos == m.pos:
            m.mp += o.value
        else:
            kept.append(o)
    return kept


def legal_nbs(blocked: List[List[bool]], p: Pos) -> List[Pos]:
    x, y = p
    g = settings.GRID
    out = []
    for nx, ny in neighbors4(x, y):
        if 0 <= nx < g and 0 <= ny < g and not blocked[ny][nx]:
            out.append((nx, ny))
    return out


def separate_equal_duel(blocked: List[List[bool]], m_a: Mage, m_b: Mage, rng: random.Random) -> None:
    cell = m_a.pos
    nbs = legal_nbs(blocked, cell)
    rng.shuffle(nbs)
    if len(nbs) >= 2:
        commit_move(m_a, nbs[0])
        commit_move(m_b, nbs[1])
    elif len(nbs) == 1:
        if rng.random() < 0.5:
            commit_move(m_a, nbs[0])
        else:
            commit_move(m_b, nbs[0])


def check_capture(m_a: Mage, m_b: Mage) -> str | None:
    if m_a.pos != m_b.pos:
        return None
    if m_a.mp > m_b.mp:
        return "A"
    if m_b.mp > m_a.mp:
        return "B"
    return "tie"


def mp_mode_label(my_mp: int, enemy_mp: int) -> str:
    if my_mp < enemy_mp:
        return "flee"
    if my_mp > enemy_mp:
        return "hunt"
    return "seek MP"


def new_game(rng: random.Random) -> tuple[List[List[bool]], Mage, Mage, List[PowerOrb]]:
    blocked = generate_grid(rng)
    m_a, m_b = spawn_two_mages(rng, blocked)
    orbs = spawn_orbs(rng, blocked, {m_a.pos, m_b.pos}, settings.MAX_ORBS_ON_FIELD)
    return blocked, m_a, m_b, orbs
