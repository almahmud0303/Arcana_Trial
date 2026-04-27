"""
3D maze duel (OpenGL + pygame): same rules as before — A* / MC+A* mages, orbs, MP duel.

Install: pip install pygame-ce PyOpenGL PyOpenGL_accelerate

Controls
  • Drag left mouse: orbit (yaw + pitch) around the maze centre
  • Mouse wheel: zoom in / out
  • [ ] : orbit yaw   •  ' / ;  : pitch up / down   •  - / = : zoom
  • Space: pause / resume simulation   •  R: new maze  |  Esc: quit

Simulation runs in slow steps: A moves → pause → B moves → pause → repeat.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pygame
from pygame.locals import DOUBLEBUF, OPENGL

from OpenGL.GL import (
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_AMBIENT,
    GL_AMBIENT_AND_DIFFUSE,
    GL_COLOR_MATERIAL,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_DIFFUSE,
    GL_FRONT_AND_BACK,
    GL_LIGHT0,
    GL_LIGHTING,
    GL_LINEAR,
    GL_MODELVIEW,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POSITION,
    GL_PROJECTION,
    GL_UNPACK_ALIGNMENT,
    GL_QUADS,
    GL_RGBA,
    GL_SRC_ALPHA,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TRIANGLE_FAN,
    GL_UNSIGNED_BYTE,
    glBegin,
    glBindTexture,
    glBlendFunc,
    glClear,
    glClearColor,
    glColor3f,
    glColor4f,
    glColorMaterial,
    glDeleteTextures,
    glDisable,
    glEnable,
    glEnd,
    glGenTextures,
    glLightfv,
    glPixelStorei,
    glLoadIdentity,
    glMatrixMode,
    glNormal3f,
    glOrtho,
    glPopMatrix,
    glPushMatrix,
    glRotatef,
    glTexCoord2f,
    glTexImage2D,
    glTexParameteri,
    glTranslatef,
    glVertex2f,
    glVertex3f,
    glViewport,
)
from OpenGL.GLU import GLU_SMOOTH, gluLookAt, gluNewQuadric, gluPerspective, gluQuadricNormals, gluSphere

from ai.agent_a_policy import plan_move_agent_a
from ai.monte_carlo_planner import monte_carlo_plan_move
from config import settings
from core.agent import commit_move
from core.game_loop import (
    check_capture,
    collect_for_mage,
    mp_mode_label,
    new_game,
    respawn_orbs_fill,
    separate_equal_duel,
)

TILE = settings.TILE
WALL_H = settings.WALL_H
HUD_HEIGHT = settings.HUD_HEIGHT
WIN_W, WIN_H = settings.WIN_W, settings.WIN_H
SUBSTEP_PAUSE_MS = settings.SUBSTEP_PAUSE_MS
PAUSE_BETWEEN_ROUNDS_MS = settings.PAUSE_BETWEEN_ROUNDS_MS
PATH = settings.PATH
WALL = settings.WALL
CA = settings.CA
CB = settings.CB
ORB_COL = settings.ORB_COL
GRID = settings.GRID

# Distinguishable cursed-maze palette
# Walls are very dark, tall, and neon-blue. Walkable floor is flat/muted stone.
BG_CLEAR = (0.008, 0.012, 0.03)
ABYSS_FLOOR = (0.01, 0.014, 0.026)

# Free grid / walkable floor: muted teal stone, low glow so it stays clearly flat.
FLOOR_DARK = (0.055, 0.09, 0.105)
FLOOR_STONE = (0.13, 0.20, 0.20)
FLOOR_SEAM = (0.12, 0.34, 0.38)

# Cursed walls: deep dark blue body with bright arcane blue curse glow.
WALL_CORE = (0.005, 0.018, 0.07)
WALL_STONE = (0.025, 0.075, 0.18)
WALL_GLOW = (0.18, 0.62, 1.0)
ARCANE_BLUE = (0.42, 0.82, 1.0)
SERPENT_MAGIC = (0.26, 0.95, 0.42)
PHOENIX_MAGIC = (1.0, 0.38, 0.13)
PHOENIX_GOLD = (1.0, 0.76, 0.24)
STAFF_DARK = (0.075, 0.055, 0.09)
ORB_VALUE_COLORS = {
    2: (0.68, 0.42, 1.0),
    4: (0.26, 0.86, 1.0),
    6: (1.0, 0.72, 0.22),
}
CARDINAL_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def cell_center(gx: int, gy: int) -> tuple[float, float, float]:
    return (gx + 0.5) * TILE, 0.0, (gy + 0.5) * TILE


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def mix_rgb(a: tuple[float, float, float], b: tuple[float, float, float], amount: float) -> tuple[float, float, float]:
    amount = clamp01(amount)
    return (
        a[0] + (b[0] - a[0]) * amount,
        a[1] + (b[1] - a[1]) * amount,
        a[2] + (b[2] - a[2]) * amount,
    )


def rgb_bytes(rgb: tuple[float, float, float], scale: int = 255) -> tuple[int, int, int]:
    return tuple(int(clamp01(c) * scale) for c in rgb)


def hash01(x: int, y: int, salt: int = 0) -> float:
    n = (x * 374761393 + y * 668265263 + salt * 1442695041) & 0xFFFFFFFF
    n = (n ^ (n >> 13)) * 1274126177 & 0xFFFFFFFF
    return ((n ^ (n >> 16)) & 0xFFFF) / 0xFFFF


def pulse01(t: float, speed: float, phase: float = 0.0) -> float:
    return 0.5 + 0.5 * math.sin(t * speed + phase)


def draw_floor_quad(
    x0: float,
    z0: float,
    x1: float,
    z1: float,
    y: float,
    rgb: tuple[float, float, float],
    alpha: float | None = None,
) -> None:
    if alpha is None:
        glColor3f(*rgb)
    else:
        glColor4f(rgb[0], rgb[1], rgb[2], alpha)
    glBegin(GL_QUADS)
    glNormal3f(0, 1, 0)
    glVertex3f(x0, y, z0)
    glVertex3f(x1, y, z0)
    glVertex3f(x1, y, z1)
    glVertex3f(x0, y, z1)
    glEnd()


def draw_vertical_quad(points: tuple[tuple[float, float, float], ...], rgb: tuple[float, float, float], alpha: float) -> None:
    glColor4f(rgb[0], rgb[1], rgb[2], alpha)
    glBegin(GL_QUADS)
    for x, y, z in points:
        glVertex3f(x, y, z)
    glEnd()


def draw_rotated_floor_bar(
    cx: float,
    cz: float,
    y: float,
    angle: float,
    length: float,
    width: float,
    rgb: tuple[float, float, float],
    alpha: float,
) -> None:
    c = math.cos(angle)
    s = math.sin(angle)
    dx = c * length * 0.5
    dz = s * length * 0.5
    px = -s * width * 0.5
    pz = c * width * 0.5
    glColor4f(rgb[0], rgb[1], rgb[2], alpha)
    glBegin(GL_QUADS)
    glVertex3f(cx - dx + px, y, cz - dz + pz)
    glVertex3f(cx + dx + px, y, cz + dz + pz)
    glVertex3f(cx + dx - px, y, cz + dz - pz)
    glVertex3f(cx - dx - px, y, cz - dz - pz)
    glEnd()


def draw_arcane_disc(
    cx: float,
    cz: float,
    radius: float,
    rgb: tuple[float, float, float],
    alpha: float,
    *,
    y: float = 0.018,
    segments: int = 30,
) -> None:
    glColor4f(rgb[0], rgb[1], rgb[2], alpha)
    glBegin(GL_TRIANGLE_FAN)
    glNormal3f(0, 1, 0)
    glVertex3f(cx, y, cz)
    for i in range(segments + 1):
        ang = (i / segments) * math.tau
        glVertex3f(cx + math.cos(ang) * radius, y, cz + math.sin(ang) * radius)
    glEnd()


def draw_box(cx: float, cy: float, cz: float, hx: float, hy: float, hz: float, rgb: tuple[float, float, float]) -> None:
    glColor3f(*rgb)
    x0, x1 = cx - hx, cx + hx
    y0, y1 = cy - hy, cy + hy
    z0, z1 = cz - hz, cz + hz
    glBegin(GL_QUADS)
    # top
    glNormal3f(0, 1, 0)
    glVertex3f(x0, y1, z0)
    glVertex3f(x1, y1, z0)
    glVertex3f(x1, y1, z1)
    glVertex3f(x0, y1, z1)
    # bottom
    glNormal3f(0, -1, 0)
    glVertex3f(x0, y0, z0)
    glVertex3f(x0, y0, z1)
    glVertex3f(x1, y0, z1)
    glVertex3f(x1, y0, z0)
    # +X
    glNormal3f(1, 0, 0)
    glVertex3f(x1, y0, z0)
    glVertex3f(x1, y0, z1)
    glVertex3f(x1, y1, z1)
    glVertex3f(x1, y1, z0)
    # -X
    glNormal3f(-1, 0, 0)
    glVertex3f(x0, y0, z0)
    glVertex3f(x0, y1, z0)
    glVertex3f(x0, y1, z1)
    glVertex3f(x0, y0, z1)
    # +Z
    glNormal3f(0, 0, 1)
    glVertex3f(x0, y0, z1)
    glVertex3f(x0, y1, z1)
    glVertex3f(x1, y1, z1)
    glVertex3f(x1, y0, z1)
    # -Z
    glNormal3f(0, 0, -1)
    glVertex3f(x0, y0, z0)
    glVertex3f(x1, y0, z0)
    glVertex3f(x1, y1, z0)
    glVertex3f(x0, y1, z0)
    glEnd()


def draw_labyrinth_backdrop(t: float, quad) -> None:
    size = GRID * TILE
    margin = 3.0
    glDisable(GL_LIGHTING)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    draw_floor_quad(-margin, -margin, size + margin, size + margin, -0.035, ABYSS_FLOOR, 1.0)

    glow = 0.08 + 0.035 * pulse01(t, 0.7)
    draw_floor_quad(-0.08, -0.08, size + 0.08, 0.04, -0.018, ARCANE_BLUE, glow)
    draw_floor_quad(-0.08, size - 0.04, size + 0.08, size + 0.08, -0.018, WALL_GLOW, glow)
    draw_floor_quad(-0.08, 0.0, 0.04, size, -0.018, SERPENT_MAGIC, glow)
    draw_floor_quad(size - 0.04, 0.0, size + 0.08, size, -0.018, PHOENIX_MAGIC, glow)

    for i in range(34):
        hx = hash01(i, 2, 11)
        hz = hash01(i, 7, 17)
        x = -1.2 + hx * (size + 2.4)
        z = -1.2 + hz * (size + 2.4)
        y = 0.08 + hash01(i, 3, 23) * 0.55 + 0.035 * math.sin(t * (0.9 + hx) + i)
        color = (ARCANE_BLUE, WALL_GLOW, SERPENT_MAGIC, PHOENIX_GOLD)[i % 4]
        alpha = 0.08 + 0.08 * pulse01(t, 0.8 + hash01(i, 5, 29), i * 0.8)
        glPushMatrix()
        glColor4f(color[0], color[1], color[2], alpha)
        glTranslatef(x, y, z)
        gluSphere(quad, 0.018 + 0.018 * hash01(i, 4, 31), 8, 8)
        glPopMatrix()

    glEnable(GL_LIGHTING)
    glDisable(GL_BLEND)


def draw_floor_tile(
    gx: int,
    gy: int,
    rgb: tuple[float, float, float],
    t: float = 0.0,
    blocked: list[list[bool]] | None = None,
) -> None:
    x0, x1 = gx * TILE, (gx + 1) * TILE
    z0, z1 = gy * TILE, (gy + 1) * TILE
    y = 0.005
    shade = hash01(gx, gy, 3)
    stone = mix_rgb(FLOOR_DARK, FLOOR_STONE, 0.35 + shade * 0.55)
    # Keep walkable tiles visibly different from walls: blue-green stone, not neon.
    stone = mix_rgb(stone, (0.02, 0.09, 0.10), 0.18)
    if (gx + gy) % 2:
        stone = mix_rgb(stone, (0.075, 0.082, 0.13), 0.25)
    draw_floor_quad(x0, z0, x1, z1, y, stone)
    inset = 0.075
    inset_stone = mix_rgb(stone, (0.24, 0.48, 0.54), 0.16 + 0.08 * shade)
    draw_floor_quad(x0 + inset, z0 + inset, x1 - inset, z1 - inset, y + 0.002, inset_stone)

    glDisable(GL_LIGHTING)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    seam = 0.014
    # Weak floor seams prevent paths from looking like glowing wall tops.
    seam_alpha = 0.08 + 0.04 * pulse01(t, 0.5, gx * 0.7 + gy)
    draw_floor_quad(x0, z0, x1, z0 + seam, y + 0.003, FLOOR_SEAM, seam_alpha)
    draw_floor_quad(x0, z1 - seam, x1, z1, y + 0.003, FLOOR_SEAM, seam_alpha * 0.65)
    draw_floor_quad(x0, z0, x0 + seam, z1, y + 0.003, FLOOR_SEAM, seam_alpha * 0.7)
    draw_floor_quad(x1 - seam, z0, x1, z1, y + 0.003, FLOOR_SEAM, seam_alpha * 0.55)

    bevel = 0.045
    draw_floor_quad(x0 + bevel, z0 + bevel, x1 - bevel, z0 + bevel * 1.8, y + 0.006, (0.6, 0.92, 1.0), 0.045)
    draw_floor_quad(x0 + bevel, z0 + bevel, x0 + bevel * 1.8, z1 - bevel, y + 0.006, (0.6, 0.92, 1.0), 0.035)
    draw_floor_quad(x0 + bevel, z1 - bevel * 1.8, x1 - bevel, z1 - bevel, y + 0.006, (0.0, 0.0, 0.0), 0.16)
    draw_floor_quad(x1 - bevel * 1.8, z0 + bevel, x1 - bevel, z1 - bevel, y + 0.006, (0.0, 0.0, 0.0), 0.13)

    if blocked is not None:
        shadow = (0.0, 0.0, 0.012)
        glow_edge = (0.25, 0.68, 0.78)
        if gy == 0 or blocked[gy - 1][gx]:
            draw_floor_quad(x0, z0, x1, z0 + 0.1, y + 0.008, shadow, 0.25)
            draw_floor_quad(x0, z0 + 0.1, x1, z0 + 0.125, y + 0.009, glow_edge, 0.08)
        if gy == GRID - 1 or blocked[gy + 1][gx]:
            draw_floor_quad(x0, z1 - 0.1, x1, z1, y + 0.008, shadow, 0.25)
            draw_floor_quad(x0, z1 - 0.125, x1, z1 - 0.1, y + 0.009, glow_edge, 0.08)
        if gx == 0 or blocked[gy][gx - 1]:
            draw_floor_quad(x0, z0, x0 + 0.1, z1, y + 0.008, shadow, 0.23)
            draw_floor_quad(x0 + 0.1, z0, x0 + 0.125, z1, y + 0.009, glow_edge, 0.07)
        if gx == GRID - 1 or blocked[gy][gx + 1]:
            draw_floor_quad(x1 - 0.1, z0, x1, z1, y + 0.008, shadow, 0.23)
            draw_floor_quad(x1 - 0.125, z0, x1 - 0.1, z1, y + 0.009, glow_edge, 0.07)

    cx, _, cz = cell_center(gx, gy)
    path_alpha = 0.035 + 0.025 * pulse01(t, 0.65, gx * 0.8 + gy * 0.45)
    draw_arcane_disc(cx, cz, 0.14, FLOOR_SEAM, path_alpha, y=y + 0.009, segments=18)

    rune_seed = hash01(gx, gy, 13)
    if rune_seed > 0.62:
        rune_color = mix_rgb(ARCANE_BLUE, WALL_GLOW, hash01(gx, gy, 19))
        alpha = 0.08 + 0.08 * pulse01(t, 1.2, rune_seed * math.tau)
        draw_arcane_disc(cx, cz, 0.22, rune_color, alpha * 0.28, y=y + 0.006, segments=20)
        draw_rotated_floor_bar(cx, cz, y + 0.007, rune_seed * math.tau, 0.46, 0.022, rune_color, alpha)
        draw_rotated_floor_bar(cx, cz, y + 0.008, rune_seed * math.tau + math.pi * 0.5, 0.32, 0.018, rune_color, alpha * 0.65)

    glEnable(GL_LIGHTING)
    glDisable(GL_BLEND)


def draw_cursed_wall(gx: int, gy: int, t: float) -> None:
    cx, _, cz = cell_center(gx, gy)
    shade = hash01(gx, gy, 41)
    base = mix_rgb(WALL_CORE, WALL_STONE, 0.25 + shade * 0.6)
    draw_box(cx, WALL_H * 0.5, cz, TILE * 0.48, WALL_H * 0.5, TILE * 0.48, base)

    x0, x1 = gx * TILE + 0.035, (gx + 1) * TILE - 0.035
    z0, z1 = gy * TILE + 0.035, (gy + 1) * TILE - 0.035
    # Strong blue curse glow makes walls pop clearly above the flat floor.
    rune = mix_rgb(ARCANE_BLUE, (0.6, 0.9, 1.0), 0.5)
    alpha = 0.32 + 0.25 * pulse01(t, 1.7, shade * math.tau)

    glDisable(GL_LIGHTING)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    draw_floor_quad(x0, z0, x1, z1, WALL_H + 0.006, mix_rgb(base, WALL_GLOW, 0.35), 0.5)
    draw_floor_quad(x0 + 0.08, z0 + 0.08, x1 - 0.08, z1 - 0.08, WALL_H + 0.011, rune, alpha * 0.8)
    draw_floor_quad(x0 - 0.04, z0 - 0.04, x1 + 0.04, z1 + 0.04, WALL_H + 0.01, WALL_GLOW, alpha * 0.22)
    draw_rotated_floor_bar(cx, cz, WALL_H + 0.016, shade * math.tau, 0.72, 0.032, rune, alpha)
    draw_rotated_floor_bar(cx, cz, WALL_H + 0.018, shade * math.tau + math.pi * 0.5, 0.46, 0.024, rune, alpha * 0.7)

    strip_x = gx * TILE + 0.18 + hash01(gx, gy, 47) * 0.64
    strip_z = gy * TILE + 0.18 + hash01(gx, gy, 53) * 0.64
    y0 = 0.09
    y1 = WALL_H * (0.62 + 0.25 * hash01(gx, gy, 59))
    half = 0.018
    face_alpha = alpha * 0.55
    draw_vertical_quad(
        (
            (strip_x - half, y0, gy * TILE - 0.003),
            (strip_x + half, y0, gy * TILE - 0.003),
            (strip_x + half, y1, gy * TILE - 0.003),
            (strip_x - half, y1, gy * TILE - 0.003),
        ),
        rune,
        face_alpha,
    )
    draw_vertical_quad(
        (
            ((gx + 1) * TILE + 0.003, y0, strip_z - half),
            ((gx + 1) * TILE + 0.003, y0, strip_z + half),
            ((gx + 1) * TILE + 0.003, y1, strip_z + half),
            ((gx + 1) * TILE + 0.003, y1, strip_z - half),
        ),
        rune,
        face_alpha * 0.8,
    )

    glEnable(GL_LIGHTING)
    glDisable(GL_BLEND)


def draw_valid_move_hints(
    blocked: list[list[bool]],
    active_pos: tuple[int, int] | None,
    color: tuple[float, float, float],
    t: float,
) -> None:
    if active_pos is None:
        return

    ax, ay = active_pos
    glDisable(GL_LIGHTING)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    for dx, dy in CARDINAL_STEPS:
        gx, gy = ax + dx, ay + dy
        if not (0 <= gx < GRID and 0 <= gy < GRID) or blocked[gy][gx]:
            continue
        cx, _, cz = cell_center(gx, gy)
        alpha = 0.13 + 0.06 * pulse01(t, 2.2, gx + gy)
        draw_arcane_disc(cx, cz, 0.28, color, alpha * 0.38, y=0.041, segments=24)
        draw_rotated_floor_bar(cx, cz, 0.046, math.atan2(dy, dx), 0.48, 0.026, color, alpha)
        draw_rotated_floor_bar(cx, cz, 0.047, math.atan2(dy, dx) + math.pi * 0.5, 0.18, 0.018, (0.9, 1.0, 1.0), alpha * 0.55)

    glEnable(GL_LIGHTING)
    glDisable(GL_BLEND)


def draw_danger_zone(
    blocked: list[list[bool]],
    danger_pos: tuple[int, int] | None,
    color: tuple[float, float, float],
    t: float,
) -> None:
    if danger_pos is None:
        return

    ex, ey = danger_pos
    danger_color = mix_rgb(color, (1.0, 0.05, 0.06), 0.55)
    glDisable(GL_LIGHTING)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    for gy in range(max(0, ey - 2), min(GRID, ey + 3)):
        for gx in range(max(0, ex - 2), min(GRID, ex + 3)):
            dist = abs(gx - ex) + abs(gy - ey)
            if dist > 2 or blocked[gy][gx]:
                continue
            cx, _, cz = cell_center(gx, gy)
            alpha = (0.17 if dist == 0 else 0.1 if dist == 1 else 0.055) + 0.035 * pulse01(t, 1.8, gx * 0.9 + gy)
            radius = 0.48 if dist == 0 else 0.38 if dist == 1 else 0.3
            draw_arcane_disc(cx, cz, radius, danger_color, alpha, y=0.038, segments=28)
            if dist <= 1:
                draw_rotated_floor_bar(cx, cz, 0.043, math.pi * 0.25, 0.62, 0.025, danger_color, alpha * 0.82)
                draw_rotated_floor_bar(cx, cz, 0.044, -math.pi * 0.25, 0.62, 0.025, danger_color, alpha * 0.82)

    glEnable(GL_LIGHTING)
    glDisable(GL_BLEND)


def draw_mage_magical(
    gx: int,
    gy: int,
    rgb: tuple[float, float, float],
    quad,
    t: float,
    phase: float,
    archetype: str,
) -> None:
    cx, _, cz = cell_center(gx, gy)

    pulse = 0.5 + 0.5 * math.sin(t * 3.4 + phase)

    if archetype == "serpent":
        body_rgb = (
            min(1.0, rgb[0] * 0.45 + 0.08),
            min(1.0, rgb[1] * 0.95 + 0.16),
            min(1.0, rgb[2] * 0.5 + 0.08),
        )
        core_rgb = (
            min(1.0, body_rgb[0] + 0.16),
            min(1.0, body_rgb[1] + 0.2),
            min(1.0, body_rgb[2] + 0.14),
        )
        sigil_alpha_main, sigil_alpha_sub = 0.3, 0.22
        aura = 0.17 + 0.07 * pulse
        wisp_count = 3
        wisp_radius = 0.22
        wisp_alpha = 0.78
        y_bob_scale = 0.8
    else:
        body_rgb = (
            min(1.0, rgb[0] * 0.95 + 0.24),
            min(1.0, rgb[1] * 0.6 + 0.16),
            max(0.04, rgb[2] * 0.28),
        )
        core_rgb = (
            min(1.0, body_rgb[0] + 0.18),
            min(1.0, body_rgb[1] + 0.16),
            min(1.0, body_rgb[2] + 0.08),
        )
        sigil_alpha_main, sigil_alpha_sub = 0.36, 0.26
        aura = 0.21 + 0.1 * pulse
        wisp_count = 5
        wisp_radius = 0.3
        wisp_alpha = 0.88
        y_bob_scale = 1.0
            # Phoenix fire particles
    if archetype == "phoenix":
        for i in range(12):
            ang = t * 2.8 + phase + i * math.tau / 12
            radius = 0.18 + 0.12 * math.sin(t * 3.0 + i)
            px = cx + math.cos(ang) * radius
            pz = cz + math.sin(ang) * radius
            py = 0.35 + 0.55 * ((t * 0.8 + i * 0.13) % 1.0)

            size = 0.025 + 0.025 * pulse01(t, 4.0, i)
            fire_color = (
                1.0,
                0.35 + 0.45 * pulse01(t, 5.0, i),
                0.05,
            )

            glPushMatrix()
            glColor4f(fire_color[0], fire_color[1], fire_color[2], 0.75)
            glTranslatef(px, py, pz)
            gluSphere(quad, size, 8, 8)
            glPopMatrix()


    # Ground sigil with two rotating rings and crossed runic bars.
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glDisable(GL_LIGHTING)
    ang1 = t * 1.15 + phase
    ang2 = -t * 0.87 + phase * 0.6
    for ang, alpha, scale in ((ang1, sigil_alpha_main, 1.0), (ang2, sigil_alpha_sub, 0.78)):
        c = math.cos(ang)
        s = math.sin(ang)
        r = 0.36 * scale
        glColor4f(rgb[0], rgb[1], rgb[2], alpha)
        glBegin(GL_QUADS)
        glVertex3f(cx + (-r) * c - (-0.03) * s, 0.011, cz + (-r) * s + (-0.03) * c)
        glVertex3f(cx + (r) * c - (-0.03) * s, 0.011, cz + (r) * s + (-0.03) * c)
        glVertex3f(cx + (r) * c - (0.03) * s, 0.011, cz + (r) * s + (0.03) * c)
        glVertex3f(cx + (-r) * c - (0.03) * s, 0.011, cz + (-r) * s + (0.03) * c)
        glEnd()
        glBegin(GL_QUADS)
        glVertex3f(cx + (-0.03) * c - (-r) * s, 0.011, cz + (-0.03) * s + (-r) * c)
        glVertex3f(cx + (0.03) * c - (-r) * s, 0.011, cz + (0.03) * s + (-r) * c)
        glVertex3f(cx + (0.03) * c - (r) * s, 0.011, cz + (0.03) * s + (r) * c)
        glVertex3f(cx + (-0.03) * c - (r) * s, 0.011, cz + (-0.03) * s + (r) * c)
        glEnd()

    # Volumetric aura shell.
    glPushMatrix()
    glColor4f(core_rgb[0], core_rgb[1], core_rgb[2], 0.16)
    glTranslatef(cx, 0.54, cz)
    gluSphere(quad, aura, 16, 16)
    glPopMatrix()

    if archetype == "serpent":
        glPushMatrix()
        glColor4f(max(0.04, body_rgb[0] - 0.05), min(1.0, body_rgb[1] + 0.05), max(0.04, body_rgb[2] - 0.05), 0.22)
        glTranslatef(cx, 0.45, cz)
        gluSphere(quad, aura + 0.05, 16, 16)
        glPopMatrix()

    glEnable(GL_LIGHTING)
    glDisable(GL_BLEND)

    if archetype == "serpent":
        # Coiled serpent body made of animated segments.
        seg_count = 6
        for i in range(seg_count):
            a = t * 2.1 + phase + i * 0.7
            radius = 0.18 - i * 0.018
            sx = cx + radius * math.cos(a)
            sz = cz + radius * math.sin(a)
            sy = 0.16 + i * 0.09 + 0.02 * math.sin(t * 3.0 + i)
            blend = i / max(1, seg_count - 1)
            seg_rgb = (
                min(1.0, body_rgb[0] * (1.0 - 0.22 * blend) + 0.06 * blend),
                min(1.0, body_rgb[1] * (1.0 - 0.18 * blend) + 0.12 * blend),
                min(1.0, body_rgb[2] * (1.0 - 0.2 * blend) + 0.04 * blend),
            )
            glPushMatrix()
            glColor3f(seg_rgb[0], seg_rgb[1], seg_rgb[2])
            glTranslatef(sx, sy, sz)
            gluSphere(quad, 0.12 - 0.01 * i, 14, 14)
            glPopMatrix()

        # Serpent head with bright eyes.
        head_a = t * 2.1 + phase + (seg_count - 1) * 0.7
        hx = cx + 0.08 * math.cos(head_a)
        hz = cz + 0.08 * math.sin(head_a)
        hy = 0.16 + (seg_count - 1) * 0.09 + 0.03
        glPushMatrix()
        glColor3f(core_rgb[0], core_rgb[1], core_rgb[2])
        glTranslatef(hx, hy, hz)
        gluSphere(quad, 0.1, 14, 14)
        glPopMatrix()

        eye_rgb = (0.95, 1.0, 0.5)
        for side in (-1.0, 1.0):
            glPushMatrix()
            glColor3f(*eye_rgb)
            glTranslatef(hx + 0.026 * side, hy + 0.016, hz + 0.036)
            gluSphere(quad, 0.018, 10, 10)
            glPopMatrix()
    else:
        # Phoenix core body and head flame.
        glPushMatrix()
        glColor3f(body_rgb[0], body_rgb[1], body_rgb[2])
        glTranslatef(cx, 0.42 + y_bob_scale * 0.02 * math.sin(t * 2.0 + phase), cz)
        gluSphere(quad, 0.16, 16, 16)
        glPopMatrix()

        glPushMatrix()
        glColor3f(core_rgb[0], core_rgb[1], core_rgb[2])
        glTranslatef(cx, 0.66 + y_bob_scale * 0.025 * math.sin(t * 2.4 + phase), cz)
        gluSphere(quad, 0.11, 14, 14)
        glPopMatrix()

        glPushMatrix()
        glColor3f(1.0, min(1.0, core_rgb[1] + 0.28), 0.25)
        glTranslatef(cx, 0.84 + 0.035 * math.sin(t * 3.2 + phase), cz)
        gluSphere(quad, 0.075, 12, 12)
        glPopMatrix()

        # Two animated glowing wings (thin boxes), plus tail embers.
        wing_open = 0.26 + 0.07 * math.sin(t * 5.0 + phase)
        wing_tilt = 0.6 + 0.18 * math.sin(t * 4.2 + phase)
        for side in (-1.0, 1.0):
            wx = cx + side * wing_open
            wy = 0.52 + 0.03 * math.sin(t * 4.5 + side)
            wz = cz - 0.02
            wing_glow = (
                min(1.0, core_rgb[0] + 0.14),
                min(1.0, core_rgb[1] + 0.16),
                min(1.0, core_rgb[2] + 0.08),
            )
            draw_box(wx, wy, wz, 0.22, 0.02, wing_tilt, wing_glow)
            draw_box(wx, wy, wz, 0.14, 0.015, wing_tilt * 0.7, core_rgb)

        for i in range(3):
            glPushMatrix()
            glColor3f(1.0, 0.6 + 0.1 * i, 0.18)
            glTranslatef(cx, 0.26 - i * 0.06, cz - 0.12 - i * 0.08)
            gluSphere(quad, 0.055 - i * 0.01, 12, 12)
            glPopMatrix()

    # A small wand-staff keeps the avatars tied to wizard dueling, not just creatures.
    staff_side = -1.0 if archetype == "serpent" else 1.0
    staff_x = cx + staff_side * 0.29
    staff_z = cz + 0.2
    draw_box(staff_x, 0.42, staff_z, 0.018, 0.34, 0.018, STAFF_DARK)
    glPushMatrix()
    glColor3f(core_rgb[0], core_rgb[1], core_rgb[2])
    glTranslatef(staff_x, 0.78 + 0.025 * math.sin(t * 3.0 + phase), staff_z)
    gluSphere(quad, 0.055, 12, 12)
    glPopMatrix()

    # Orbiting wisps.
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glDisable(GL_LIGHTING)
    for i in range(wisp_count):
        ang = t * (1.9 + i * 0.15) + phase + i * (2.0 * math.pi / max(1, wisp_count))
        wx = cx + wisp_radius * math.cos(ang)
        wz = cz + wisp_radius * math.sin(ang)
        wy = 0.58 + (0.04 if archetype == "serpent" else 0.06) * math.sin(t * 4.2 + i)
        glPushMatrix()
        if archetype == "serpent":
            glColor4f(min(1.0, body_rgb[0] + 0.2), min(1.0, body_rgb[1] + 0.22), min(1.0, body_rgb[2] + 0.16), wisp_alpha)
        else:
            glColor4f(1.0, min(1.0, body_rgb[1] + 0.25), min(1.0, body_rgb[2] + 0.1), wisp_alpha)
        glTranslatef(wx, wy, wz)
        gluSphere(quad, 0.04 if archetype == "serpent" else 0.05, 10, 10)
        glPopMatrix()

    if archetype == "serpent":
        # Venom motes close to the serpent body.
        for i in range(2):
            ang = -t * (1.8 + i * 0.2) + phase + i * math.pi
            sx = cx + 0.12 * math.cos(ang)
            sz = cz + 0.12 * math.sin(ang)
            sy = 0.34 + 0.06 * math.sin(t * 3.5 + i * 1.3)
            glPushMatrix()
            glColor4f(0.78, 0.95, 0.35, 0.72)
            glTranslatef(sx, sy, sz)
            gluSphere(quad, 0.05, 10, 10)
            glPopMatrix()

    glEnable(GL_LIGHTING)
    glDisable(GL_BLEND)


def build_orb_value_textures(font: pygame.font.Font) -> dict[int, tuple[int, int, int]]:
    """Textures for +2 / +4 / +6 labels (tex_id, pixel_w, pixel_h)."""
    out: dict[int, tuple[int, int, int]] = {}
    for val in settings.POWER_VALUES:
        color = ORB_VALUE_COLORS.get(val, ORB_COL)
        text_rgb = tuple(int(180 + c * 75) for c in color)
        text = font.render(f"+{val}", True, text_rgb)
        tw, th = text.get_size()
        surf = pygame.Surface((tw + 18, th + 10), pygame.SRCALPHA)
        bg_rgb = tuple(int(c * 90) for c in color)
        edge_rgb = tuple(int(155 + c * 90) for c in color)
        pygame.draw.rect(surf, (*bg_rgb, 130), surf.get_rect(), border_radius=8)
        pygame.draw.rect(surf, (*edge_rgb, 210), surf.get_rect(), width=1, border_radius=8)
        surf.blit(text, ((surf.get_width() - tw) // 2, (surf.get_height() - th) // 2 - 1))
        surf = surf.convert_alpha()
        w, h = surf.get_size()
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, int(tex))
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        try:
            raw = pygame.image.tobytes(surf, "RGBA", True)
        except AttributeError:
            raw = pygame.image.tostring(surf, "RGBA", True)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, raw)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        out[val] = (int(tex), w, h)
    return out


def draw_artifact_orb(
    orb,
    quad,
    *,
    cam_yaw: float,
    cam_pitch: float,
    time_seconds: float,
    orb_labels: dict[int, tuple[int, int, int]],
) -> None:
    cx, _, cz = cell_center(*orb.pos)
    color = ORB_VALUE_COLORS.get(orb.value, ORB_COL)
    pulse = pulse01(time_seconds, 2.4, orb.value * 0.7 + orb.pos[0])
    bob = 0.04 * math.sin(time_seconds * 2.2 + orb.pos[1] + orb.value)
    ring_angle = time_seconds * (0.9 + orb.value * 0.08)

    glDisable(GL_LIGHTING)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    draw_arcane_disc(cx, cz, 0.43 + 0.04 * pulse, color, 0.11 + 0.07 * pulse, y=0.025, segments=34)
    draw_rotated_floor_bar(cx, cz, 0.031, ring_angle, 0.76, 0.025, color, 0.22 + 0.13 * pulse)
    draw_rotated_floor_bar(cx, cz, 0.032, -ring_angle * 0.8, 0.52, 0.018, PHOENIX_GOLD, 0.14 + 0.08 * pulse)

    glPushMatrix()
    glColor4f(color[0], color[1], color[2], 0.14 + 0.08 * pulse)
    glTranslatef(cx, 0.52 + bob, cz)
    gluSphere(quad, 0.34 + 0.04 * pulse, 18, 18)
    glPopMatrix()

    for i in range(5):
        ang = ring_angle * (1.2 + i * 0.1) + i * math.tau / 5
        radius = 0.32 + 0.05 * math.sin(time_seconds * 1.8 + i)
        spark_color = color if i % 2 else PHOENIX_GOLD
        glPushMatrix()
        glColor4f(spark_color[0], spark_color[1], spark_color[2], 0.52)
        glTranslatef(cx + math.cos(ang) * radius, 0.6 + bob + 0.06 * math.sin(ang), cz + math.sin(ang) * radius)
        gluSphere(quad, 0.025, 8, 8)
        glPopMatrix()

    glEnable(GL_LIGHTING)
    glDisable(GL_BLEND)

    glColor3f(*mix_rgb(color, (1.0, 1.0, 1.0), 0.22))
    glPushMatrix()
    glTranslatef(cx, 0.52 + bob, cz)
    gluSphere(quad, 0.2, 16, 16)
    glPopMatrix()

    if orb.value not in orb_labels:
        return

    tid, tw, th = orb_labels[orb.value]
    glPushMatrix()
    glTranslatef(cx, 0.92 + bob, cz)
    glRotatef(-math.degrees(cam_yaw), 0.0, 1.0, 0.0)
    glRotatef(-math.degrees(cam_pitch), 1.0, 0.0, 0.0)
    glDisable(GL_LIGHTING)
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor3f(1.0, 1.0, 1.0)
    glBindTexture(GL_TEXTURE_2D, tid)
    half_h = 0.21
    half_w = half_h * (tw / max(th, 1))
    glNormal3f(0.0, 0.0, 1.0)
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 0.0)
    glVertex3f(-half_w, -half_h, 0.0)
    glTexCoord2f(1.0, 0.0)
    glVertex3f(half_w, -half_h, 0.0)
    glTexCoord2f(1.0, 1.0)
    glVertex3f(half_w, half_h, 0.0)
    glTexCoord2f(0.0, 1.0)
    glVertex3f(-half_w, half_h, 0.0)
    glEnd()
    glDisable(GL_TEXTURE_2D)
    glDisable(GL_BLEND)
    glEnable(GL_LIGHTING)
    glPopMatrix()


def draw_magic_particles(t: float, quad) -> None:
    size = GRID * TILE
    glDisable(GL_LIGHTING)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    for i in range(26):
        hx = hash01(i, 11, 71)
        hz = hash01(i, 17, 73)
        x = 0.15 + hx * (size - 0.3)
        z = 0.15 + hz * (size - 0.3)
        y = 0.2 + hash01(i, 19, 79) * 1.15 + 0.05 * math.sin(t * (1.1 + hx) + i)
        color = (SERPENT_MAGIC, PHOENIX_MAGIC, ARCANE_BLUE, WALL_GLOW)[i % 4]
        alpha = 0.08 + 0.12 * pulse01(t, 0.7 + hz, i)
        glPushMatrix()
        glColor4f(color[0], color[1], color[2], alpha)
        glTranslatef(x, y, z)
        gluSphere(quad, 0.012 + 0.022 * hash01(i, 23, 83), 8, 8)
        glPopMatrix()

    glEnable(GL_LIGHTING)
    glDisable(GL_BLEND)


def draw_world(
    blocked,
    m_a,
    m_b,
    orbs,
    quad,
    *,
    cam_yaw: float,
    cam_pitch: float,
    time_seconds: float,
    orb_labels: dict[int, tuple[int, int, int]],
    active_pos: tuple[int, int] | None = None,
    active_color: tuple[float, float, float] = ARCANE_BLUE,
    danger_pos: tuple[int, int] | None = None,
    danger_color: tuple[float, float, float] = PHOENIX_MAGIC,
) -> None:
    draw_labyrinth_backdrop(time_seconds, quad)

    for gy in range(GRID):
        for gx in range(GRID):
            if blocked[gy][gx]:
                draw_cursed_wall(gx, gy, time_seconds)
            else:
                draw_floor_tile(gx, gy, PATH, time_seconds, blocked)

    draw_danger_zone(blocked, danger_pos, danger_color, time_seconds)
    draw_valid_move_hints(blocked, active_pos, active_color, time_seconds)

    for o in orbs:
        draw_artifact_orb(
            o,
            quad,
            cam_yaw=cam_yaw,
            cam_pitch=cam_pitch,
            time_seconds=time_seconds,
            orb_labels=orb_labels,
        )

    draw_mage_magical(m_a.pos[0], m_a.pos[1], SERPENT_MAGIC, quad, time_seconds, phase=0.0, archetype="serpent")
    draw_mage_magical(m_b.pos[0], m_b.pos[1], PHOENIX_MAGIC, quad, time_seconds, phase=1.7, archetype="phoenix")
    draw_magic_particles(time_seconds, quad)


def setup_gl() -> None:
    glEnable(GL_DEPTH_TEST)
    glClearColor(BG_CLEAR[0], BG_CLEAR[1], BG_CLEAR[2], 1.0)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.16, 0.14, 0.24, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.72, 0.78, 1.0, 1.0))
    pos = (float(GRID * 0.8), float(GRID * 1.1), float(GRID * 0.6), 1.0)
    glLightfv(GL_LIGHT0, GL_POSITION, pos)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)


def render_fit_text(
    surf: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    color: tuple[int, int, int],
    pos: tuple[int, int],
    max_width: int,
) -> None:
    if font.size(text)[0] <= max_width:
        surf.blit(font.render(text, True, color), pos)
        return

    fitted = text
    while fitted and font.size(fitted + "...")[0] > max_width:
        fitted = fitted[:-1]
    surf.blit(font.render(f"{fitted}...", True, color), pos)


def strategy_label(my_mp: int, enemy_mp: int) -> str:
    mode = mp_mode_label(my_mp, enemy_mp)
    if mode == "hunt":
        return "Aggressive"
    if mode == "flee":
        return "Defensive"
    return "Tactical"


def artifact_indicator(orbs) -> str:
    if not orbs:
        return "Artifacts: 0 active | field is empty"
    values = sorted((o.value for o in orbs), reverse=True)
    value_text = " ".join(f"+{v}" for v in values)
    return f"Artifacts: {len(orbs)}/{settings.MAX_ORBS_ON_FIELD} active | {value_text}"


def draw_mp_bar(
    surf: pygame.Surface,
    rect: pygame.Rect,
    value: int,
    max_value: int,
    color: tuple[float, float, float],
) -> None:
    fill_w = int(rect.width * clamp01(value / max(1, max_value)))
    fill_rect = pygame.Rect(rect.x, rect.y, fill_w, rect.height)
    pygame.draw.rect(surf, (14, 17, 28, 235), rect, border_radius=5)
    if fill_w > 0:
        pygame.draw.rect(surf, (*rgb_bytes(color), 225), fill_rect, border_radius=5)
        shine = pygame.Rect(rect.x, rect.y, fill_w, max(2, rect.height // 3))
        pygame.draw.rect(surf, (255, 255, 255, 45), shine, border_radius=5)
    pygame.draw.rect(surf, (130, 148, 170, 160), rect, width=1, border_radius=5)


def draw_agent_panel(
    surf: pygame.Surface,
    font: pygame.font.Font,
    rect: pygame.Rect,
    *,
    name: str,
    mp: int,
    enemy_mp: int,
    max_mp: int,
    color: tuple[float, float, float],
    is_turn: bool,
    winner: bool,
) -> None:
    base = rgb_bytes(mix_rgb(color, (0.018, 0.022, 0.04), 0.82))
    edge = rgb_bytes(mix_rgb(color, (1.0, 1.0, 1.0), 0.24))
    pygame.draw.rect(surf, (*base, 190), rect, border_radius=8)
    pygame.draw.rect(surf, (*edge, 165), rect, width=1, border_radius=8)

    pygame.draw.circle(surf, (*rgb_bytes(color), 235), (rect.x + 16, rect.y + 16), 6)
    render_fit_text(surf, font, name, (236, 239, 232), (rect.x + 28, rect.y + 7), rect.width - 96)

    if is_turn and not winner:
        turn_rect = pygame.Rect(rect.right - 62, rect.y + 7, 48, 18)
        pygame.draw.rect(surf, (*rgb_bytes(color), 210), turn_rect, border_radius=6)
        label = font.render("TURN", True, (12, 14, 20))
        surf.blit(label, label.get_rect(center=turn_rect.center))

    strategy = strategy_label(mp, enemy_mp)
    render_fit_text(surf, font, f"Strategy: {strategy}", (188, 207, 222), (rect.x + 14, rect.y + 31), rect.width - 28)
    draw_mp_bar(surf, pygame.Rect(rect.x + 14, rect.y + 55, rect.width - 28, 15), mp, max_mp, color)
    render_fit_text(surf, font, f"Magic Power {mp}/{max_mp}", (233, 218, 174), (rect.x + 14, rect.y + 75), rect.width - 28)


def hud_texture(
    font: pygame.font.Font,
    title_font: pygame.font.Font,
    *,
    m_a,
    m_b,
    orbs,
    sim_next: str,
    sim_paused: bool,
    winner: str | None,
    last_line: str,
    sim_timer: float,
) -> tuple[int, int, int]:
    surf = pygame.Surface((WIN_W, HUD_HEIGHT), pygame.SRCALPHA)
    for y in range(HUD_HEIGHT):
        shade = 8 + int(10 * (y / max(1, HUD_HEIGHT - 1)))
        pygame.draw.line(surf, (shade, shade + 2, shade + 12, 246), (0, y), (WIN_W, y))

    panel = pygame.Rect(10, 8, WIN_W - 20, HUD_HEIGHT - 16)
    pygame.draw.rect(surf, (10, 13, 25, 238), panel, border_radius=8)
    pygame.draw.rect(surf, (82, 91, 124, 150), panel, width=1, border_radius=8)
    pygame.draw.line(surf, (*rgb_bytes(ARCANE_BLUE), 115), (18, 10), (WIN_W // 2 - 22, 10), 2)
    pygame.draw.line(surf, (*rgb_bytes(WALL_GLOW), 115), (WIN_W // 2 + 22, 10), (WIN_W - 18, 10), 2)

    title = title_font.render("The Arcane Trial of Hogwarts", True, (238, 226, 196))
    subtitle = font.render("Enchanted Labyrinth Duel", True, (158, 173, 210))
    surf.blit(title, (18, 14))
    surf.blit(subtitle, (20, 38))

    if winner:
        winner_house = "Serpent A" if winner == "A" else "Phoenix B"
        status = f"Duel result: {winner_house} wins | {last_line}"
        status_color = (255, 221, 154)
    elif sim_paused:
        status = "Status: paused | Space resumes the duel"
        status_color = (224, 210, 246)
    else:
        next_house = "Serpent A" if sim_next == "a" else "Phoenix B"
        status = f"Turn: {next_house} | action in ~{max(0, int(sim_timer))} ms"
        status_color = (217, 232, 238)

    render_fit_text(surf, font, status, status_color, (20, 60), 258)
    render_fit_text(surf, font, artifact_indicator(orbs), (182, 224, 218), (20, 80), 258)
    render_fit_text(surf, font, "Drag orbit | Wheel zoom | Space pause | R reset", (143, 158, 184), (20, 101), 258)

    max_mp = max(20, ((max(m_a.mp, m_b.mp, settings.START_MP) + 9) // 10) * 10)
    draw_agent_panel(
        surf,
        font,
        pygame.Rect(298, 20, 270, 90),
        name="Serpent A",
        mp=m_a.mp,
        enemy_mp=m_b.mp,
        max_mp=max_mp,
        color=SERPENT_MAGIC,
        is_turn=sim_next == "a",
        winner=winner is not None,
    )
    draw_agent_panel(
        surf,
        font,
        pygame.Rect(586, 20, 270, 90),
        name="Phoenix B",
        mp=m_b.mp,
        enemy_mp=m_a.mp,
        max_mp=max_mp,
        color=PHOENIX_MAGIC,
        is_turn=sim_next == "b",
        winner=winner is not None,
    )

    w, h = surf.get_size()
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    # OpenGL expects first row = texture bottom; pygame is top-first → flip
    try:
        raw = pygame.image.tobytes(surf, "RGBA", True)
    except AttributeError:
        raw = pygame.image.tostring(surf, "RGBA", True)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, raw)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    return tex, w, h


def draw_hud_overlay(tex: int, tw: int, th: int) -> None:
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex)
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    # y=0 at bottom, y=WIN_H at top — matches upright texture after flipped upload
    glOrtho(0, WIN_W, 0, WIN_H, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glColor4f(1, 1, 1, 1)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0)
    glVertex2f(0, 0)
    glTexCoord2f(1, 0)
    glVertex2f(WIN_W, 0)
    glTexCoord2f(1, 1)
    glVertex2f(WIN_W, HUD_HEIGHT)
    glTexCoord2f(0, 1)
    glVertex2f(0, HUD_HEIGHT)
    glEnd()
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glDisable(GL_TEXTURE_2D)
    glDisable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)


def set_camera(yaw: float, pitch: float, distance: float) -> None:
    """
    Orbit camera on a sphere around maze centre (yaw = radians around Y, pitch = elevation).
    """
    mid = GRID * TILE * 0.5
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    h_radius = distance * cp
    eye_y = distance * sp + 0.4
    ex = mid + h_radius * math.sin(yaw)
    ez = mid + h_radius * math.cos(yaw)
    gluLookAt(ex, eye_y, ez, mid, 0.35, mid, 0, 1, 0)


def main() -> None:
    pygame.init()
    pygame.display.set_mode((WIN_W, WIN_H), DOUBLEBUF | OPENGL)
    pygame.display.set_caption(f"The Arcane Trial of Hogwarts - Phoenix vs Serpent ({GRID}x{GRID})")
    pygame.key.set_repeat(200, 40)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)
    title_font = pygame.font.SysFont("georgia", 19, bold=True)

    setup_gl()
    quad = gluNewQuadric()
    gluQuadricNormals(quad, GLU_SMOOTH)
    orb_label_font = pygame.font.SysFont("consolas", 22, bold=True)
    orb_labels = build_orb_value_textures(orb_label_font)

    rng = random.Random()
    blocked, m_a, m_b, orbs = new_game(rng)
    sim_timer = SUBSTEP_PAUSE_MS
    sim_next: str = "a"
    sim_paused = False
    winner: str | None = None
    last_line = ""
    substep = "-"
    cam_yaw = 0.65
    cam_pitch = 0.52
    cam_dist = float(GRID) * 1.45
    hud_tex: int | None = None
    hud_tw = hud_th = 0
    running = True

    while running:
        dt = clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    blocked, m_a, m_b, orbs = new_game(rng)
                    sim_timer = SUBSTEP_PAUSE_MS
                    sim_next = "a"
                    sim_paused = False
                    winner = None
                    last_line = ""
                    substep = "-"
                    cam_yaw = 0.65
                    cam_pitch = 0.52
                    cam_dist = float(GRID) * 1.45
                elif event.key == pygame.K_SPACE:
                    sim_paused = not sim_paused
                elif event.key == pygame.K_LEFTBRACKET:
                    cam_yaw -= 0.08
                elif event.key == pygame.K_RIGHTBRACKET:
                    cam_yaw += 0.08
                elif event.key in (pygame.K_QUOTE, pygame.K_BACKQUOTE):
                    cam_pitch = min(1.38, cam_pitch + 0.07)
                elif event.key == pygame.K_SEMICOLON:
                    cam_pitch = max(0.12, cam_pitch - 0.07)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    cam_dist = min(32.0, cam_dist + 0.75)
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    cam_dist = max(4.0, cam_dist - 0.75)
            elif event.type == pygame.MOUSEMOTION:
                if event.buttons[0]:
                    cam_yaw += event.rel[0] * 0.01
                    cam_pitch -= event.rel[1] * 0.01
                    cam_pitch = max(0.12, min(1.38, cam_pitch))
            elif event.type == pygame.MOUSEWHEEL:
                cam_dist = max(4.0, min(32.0, cam_dist - event.y * 0.9))

        if winner is None and not sim_paused:
            sim_timer -= dt
            if sim_timer <= 0:
                if sim_next == "a":
                    pa = plan_move_agent_a(
                        m_a.pos,
                        m_b.pos,
                        m_a.mp,
                        m_b.mp,
                        orbs,
                        blocked,
                        grid_w=GRID,
                        grid_h=GRID,
                        rng=rng,
                        prev_pos=m_a.prev_pos,
                    )
                    commit_move(m_a, pa)
                    orbs = collect_for_mage(m_a, orbs)
                    orbs = respawn_orbs_fill(rng, blocked, m_a, m_b, orbs)
                    last_line = f"1/2 Serpent A -> {m_a.pos}"
                    cap = check_capture(m_a, m_b)
                    if cap == "A":
                        winner, last_line = "A", "A wins (higher MP)"
                    elif cap == "B":
                        winner, last_line = "B", "B wins (higher MP)"
                    elif cap == "tie":
                        separate_equal_duel(blocked, m_a, m_b, rng)
                        last_line = "Equal MP - separated"
                    if winner is None:
                        sim_next = "b"
                        sim_timer = SUBSTEP_PAUSE_MS
                else:
                    pb = monte_carlo_plan_move(
                        m_b.pos,
                        m_a.pos,
                        m_b.mp,
                        m_a.mp,
                        orbs,
                        blocked,
                        grid_w=GRID,
                        grid_h=GRID,
                        rng=rng,
                        rollouts=settings.MONTE_CARLO_ROLLOUTS,
                        depth=settings.MONTE_CARLO_DEPTH,
                    )
                    commit_move(m_b, pb)
                    orbs = collect_for_mage(m_b, orbs)
                    orbs = respawn_orbs_fill(rng, blocked, m_a, m_b, orbs)
                    last_line = f"2/2 Phoenix B -> {m_b.pos}"
                    cap = check_capture(m_a, m_b)
                    if cap == "A":
                        winner, last_line = "A", "A wins (higher MP)"
                    elif cap == "B":
                        winner, last_line = "B", "B wins (higher MP)"
                    elif cap == "tie":
                        separate_equal_duel(blocked, m_a, m_b, rng)
                        last_line = "Equal MP - separated"
                    if winner is None:
                        sim_next = "a"
                        sim_timer = SUBSTEP_PAUSE_MS + PAUSE_BETWEEN_ROUNDS_MS

        if winner is None:
            next_house = "Serpent A" if sim_next == "a" else "Phoenix B"
            wait_note = f" | wait ~{max(0, int(sim_timer))} ms -> {next_house}"
            if sim_paused:
                wait_note = " | PAUSED (Space)"
            substep = (
                f"Serpent A: {mp_mode_label(m_a.mp, m_b.mp)} | Phoenix B: {mp_mode_label(m_b.mp, m_a.mp)} "
                f"| MP A={m_a.mp} B={m_b.mp}{wait_note}"
            )

        # OpenGL viewport (x,y) is lower-left. Reserve the bottom strip for HUD only —
        # do NOT draw 3D there (old code used y=0 and the bar covered the board corner).
        glViewport(0, 0, WIN_W, WIN_H)
        glClearColor(BG_CLEAR[0], BG_CLEAR[1], BG_CLEAR[2], 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(0, HUD_HEIGHT, WIN_W, WIN_H - HUD_HEIGHT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(48, WIN_W / (WIN_H - HUD_HEIGHT), 0.1, 80.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        set_camera(cam_yaw, cam_pitch, cam_dist)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if winner is None:
            active_pos = m_a.pos if sim_next == "a" else m_b.pos
            active_color = SERPENT_MAGIC if sim_next == "a" else PHOENIX_MAGIC
            danger_pos = m_b.pos if sim_next == "a" else m_a.pos
            danger_color = PHOENIX_MAGIC if sim_next == "a" else SERPENT_MAGIC
        else:
            active_pos = None
            active_color = ARCANE_BLUE
            danger_pos = None
            danger_color = PHOENIX_MAGIC

        draw_world(
            blocked,
            m_a,
            m_b,
            orbs,
            quad,
            cam_yaw=cam_yaw,
            cam_pitch=cam_pitch,
            time_seconds=pygame.time.get_ticks() / 1000.0,
            orb_labels=orb_labels,
            active_pos=active_pos,
            active_color=active_color,
            danger_pos=danger_pos,
            danger_color=danger_color,
        )

        if hud_tex is not None:
            glDeleteTextures(int(hud_tex))
        tid, hud_tw, hud_th = hud_texture(
            font,
            title_font,
            m_a=m_a,
            m_b=m_b,
            orbs=orbs,
            sim_next=sim_next,
            sim_paused=sim_paused,
            winner=winner,
            last_line=last_line or "The labyrinth is listening.",
            sim_timer=sim_timer,
        )
        hud_tex = int(tid)
        glViewport(0, 0, WIN_W, WIN_H)
        draw_hud_overlay(int(hud_tex), hud_tw, hud_th)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
