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


def cell_center(gx: int, gy: int) -> tuple[float, float, float]:
    return (gx + 0.5) * TILE, 0.0, (gy + 0.5) * TILE


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


def draw_floor_tile(gx: int, gy: int, rgb: tuple[float, float, float]) -> None:
    x0, x1 = gx * TILE, (gx + 1) * TILE
    z0, z1 = gy * TILE, (gy + 1) * TILE
    y = 0.005
    glColor3f(*rgb)
    glBegin(GL_QUADS)
    glNormal3f(0, 1, 0)
    glVertex3f(x0, y, z0)
    glVertex3f(x1, y, z0)
    glVertex3f(x1, y, z1)
    glVertex3f(x0, y, z1)
    glEnd()


def build_orb_value_textures(font: pygame.font.Font) -> dict[int, tuple[int, int, int]]:
    """Textures for +2 / +4 / +6 labels (tex_id, pixel_w, pixel_h)."""
    out: dict[int, tuple[int, int, int]] = {}
    for val in settings.POWER_VALUES:
        surf = font.render(f"+{val}", True, (40, 25, 70), (255, 240, 210))
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


def draw_world(
    blocked,
    m_a,
    m_b,
    orbs,
    quad,
    *,
    cam_yaw: float,
    cam_pitch: float,
    orb_labels: dict[int, tuple[int, int, int]],
) -> None:
    for gy in range(GRID):
        for gx in range(GRID):
            if blocked[gy][gx]:
                cx, _, cz = cell_center(gx, gy)
                draw_box(cx, WALL_H * 0.5, cz, TILE * 0.48, WALL_H * 0.5, TILE * 0.48, WALL)
            else:
                draw_floor_tile(gx, gy, PATH)

    cx, _, cz = cell_center(*m_a.pos)
    draw_box(cx, 0.38, cz, 0.22, 0.36, 0.22, CA)
    cx, _, cz = cell_center(*m_b.pos)
    draw_box(cx, 0.38, cz, 0.22, 0.36, 0.22, CB)

    glColor3f(*ORB_COL)
    for o in orbs:
        cx, _, cz = cell_center(*o.pos)
        glPushMatrix()
        glTranslatef(cx, 0.52, cz)
        gluSphere(quad, 0.2, 14, 14)
        glPopMatrix()

        if o.value not in orb_labels:
            continue
        tid, tw, th = orb_labels[o.value]
        glPushMatrix()
        glTranslatef(cx, 0.88, cz)
        # Face the orbit camera (same yaw/pitch as set_camera)
        glRotatef(-math.degrees(cam_yaw), 0.0, 1.0, 0.0)
        glRotatef(-math.degrees(cam_pitch), 1.0, 0.0, 0.0)
        glDisable(GL_LIGHTING)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor3f(1.0, 1.0, 1.0)
        glBindTexture(GL_TEXTURE_2D, tid)
        half_h = 0.24
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


def setup_gl() -> None:
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.12, 0.12, 0.14, 1.0)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 0.95, 0.88, 1.0))
    pos = (float(GRID * 0.8), float(GRID * 1.1), float(GRID * 0.6), 1.0)
    glLightfv(GL_LIGHT0, GL_POSITION, pos)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)


def hud_texture(lines: list[str], font: pygame.font.Font) -> tuple[int, int, int]:
    surf = pygame.Surface((WIN_W, HUD_HEIGHT), pygame.SRCALPHA)
    surf.fill((24, 24, 30, 255))
    y = 6
    for line in lines:
        t = font.render(line, True, (220, 220, 230))
        surf.blit(t, (12, y))
        y += 22
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
    pygame.display.set_caption(f"3D Maze — Mages A vs B ({GRID}×{GRID})")
    pygame.key.set_repeat(200, 40)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 17)

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
    substep = "—"
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
                    substep = "—"
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
                    last_line = f"1/2 Mage A → {m_a.pos}"
                    cap = check_capture(m_a, m_b)
                    if cap == "A":
                        winner, last_line = "A", "A wins (higher MP)"
                    elif cap == "B":
                        winner, last_line = "B", "B wins (higher MP)"
                    elif cap == "tie":
                        separate_equal_duel(blocked, m_a, m_b, rng)
                        last_line = "Equal MP — separated"
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
                    last_line = f"2/2 Mage B → {m_b.pos}"
                    cap = check_capture(m_a, m_b)
                    if cap == "A":
                        winner, last_line = "A", "A wins (higher MP)"
                    elif cap == "B":
                        winner, last_line = "B", "B wins (higher MP)"
                    elif cap == "tie":
                        separate_equal_duel(blocked, m_a, m_b, rng)
                        last_line = "Equal MP — separated"
                    if winner is None:
                        sim_next = "a"
                        sim_timer = SUBSTEP_PAUSE_MS + PAUSE_BETWEEN_ROUNDS_MS

        if winner is None:
            wait_note = f" | wait ~{max(0, int(sim_timer))} ms → {'A' if sim_next == 'a' else 'B'}"
            if sim_paused:
                wait_note = " | PAUSED (Space)"
            substep = (
                f"A: {mp_mode_label(m_a.mp, m_b.mp)} | B: {mp_mode_label(m_b.mp, m_a.mp)} "
                f"| MP A={m_a.mp} B={m_b.mp}{wait_note}"
            )

        # OpenGL viewport (x,y) is lower-left. Reserve the bottom strip for HUD only —
        # do NOT draw 3D there (old code used y=0 and the bar covered the board corner).
        glViewport(0, 0, WIN_W, WIN_H)
        glClearColor(0.12, 0.12, 0.14, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(0, HUD_HEIGHT, WIN_W, WIN_H - HUD_HEIGHT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(48, WIN_W / (WIN_H - HUD_HEIGHT), 0.1, 80.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        set_camera(cam_yaw, cam_pitch, cam_dist)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_world(
            blocked,
            m_a,
            m_b,
            orbs,
            quad,
            cam_yaw=cam_yaw,
            cam_pitch=cam_pitch,
            orb_labels=orb_labels,
        )

        lines = [
            "LMB: orbit  |  wheel / - =: zoom  |  [ ] yaw  |  ' ; pitch  |  Space: pause sim  |  R  Esc",
            f"Slow steps: {int(SUBSTEP_PAUSE_MS)} ms after each mage + {int(PAUSE_BETWEEN_ROUNDS_MS)} ms between rounds",
            substep,
            last_line or "—",
        ]
        if winner:
            lines.append(f"Winner: {winner}")
        if hud_tex is not None:
            glDeleteTextures(int(hud_tex))
        tid, hud_tw, hud_th = hud_texture(lines, font)
        hud_tex = int(tid)
        glViewport(0, 0, WIN_W, WIN_H)
        draw_hud_overlay(int(hud_tex), hud_tw, hud_th)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
