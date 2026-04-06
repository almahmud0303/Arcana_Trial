"""Game, maze, simulation, and 3D UI constants (SETUP.md: config/settings.py)."""

from __future__ import annotations

from typing import Tuple

# Maze & simulation
GRID = 10
TARGET_BLOCK_RATIO = 0.45
MAX_WAVES = max(200, GRID * GRID // 2)
START_MP = 10
POWER_VALUES: Tuple[int, ...] = (2, 4, 6)
MAX_ORBS_ON_FIELD = 3
MOVE_INTERVAL_MS = 260
MONTE_CARLO_ROLLOUTS = 56
MONTE_CARLO_DEPTH = 14

# 3D view / window
TILE = 1.0
WALL_H = 0.82
HUD_HEIGHT = 130
WIN_W = 880
WIN_H = 720
SUBSTEP_PAUSE_MS = 100.0
PAUSE_BETWEEN_ROUNDS_MS = 100.0

# OpenGL colors (0–1)
PATH = (0.92, 0.92, 0.94)
WALL = (0.78, 0.48, 0.32)
CA = (0.62, 0.82, 1.0)
CB = (0.35, 0.86, 0.55)
ORB_COL = (0.78, 0.55, 0.95)
