# Hogwarts Maze AI Game Setup (Python + Pygame)

This guide sets up your AI lab project on the **D drive** and redirects Python “app data” (temp/cache/user installs/pycache) to **`D:\py`** so you **don’t use `C:`** for development artifacts.

## 0) Force Python app data onto `D:\py` (recommended)

Run this once in PowerShell (it will create folders and set **user** environment variables):

```powershell
# Create all app-data folders on D:
mkdir "D:\py" -Force | Out-Null
mkdir "D:\py\tmp" -Force | Out-Null
mkdir "D:\py\pip-cache" -Force | Out-Null
mkdir "D:\py\python-user-base" -Force | Out-Null
mkdir "D:\py\pycache" -Force | Out-Null

# Redirect Windows temp (many tools use this)
setx TEMP "D:\py\tmp"
setx TMP  "D:\py\tmp"

# Redirect pip cache
setx PIP_CACHE_DIR "D:\py\pip-cache"

# Redirect Python user installs (only used if you ever do: pip install --user ...)
setx PYTHONUSERBASE "D:\py\python-user-base"

# Redirect .pyc bytecode cache (keeps __pycache__ off your project)
setx PYTHONPYCACHEPREFIX "D:\py\pycache"
```

Then **close and reopen** PowerShell (so the new environment variables take effect).

## 1) Create project folder on D drive

Use PowerShell:

```powershell
D:
mkdir "D:\AI\arcana_trial"
cd "D:\AI\arcana_trial"
```

## 2) Create and activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 3) Install required packages

```powershell
python -m pip install --upgrade pip
pip install pygame-ce numpy
```

Optional (for fuzzy logic rules):

```powershell
pip install scikit-fuzzy scipy packaging
```

Quick sanity check (should show `D:\py\...` paths):

```powershell
python -c "import os,sys; print('TEMP=',os.getenv('TEMP')); print('TMP=',os.getenv('TMP')); print('PIP_CACHE_DIR=',os.getenv('PIP_CACHE_DIR')); print('PYTHONUSERBASE=',os.getenv('PYTHONUSERBASE')); print('PYTHONPYCACHEPREFIX=',os.getenv('PYTHONPYCACHEPREFIX')); print('executable=',sys.executable)"
```

## 4) Suggested project structure

```text
D:\AI\hogwarts-maze-game
├── main.py
├── requirements.txt
├── assets\
│   ├── sprites\
│   └── sounds\
├── core\
│   ├── maze.py
│   ├── agent.py
│   ├── artifact.py
│   └── game_loop.py
├── ai\
│   ├── astar.py
│   ├── minimax.py
│   └── fuzzy_logic.py
└── config\
    └── settings.py
```

## 5) Save dependencies

```powershell
pip freeze > requirements.txt
```

## 6) Minimal Pygame test

Create `main.py`:

```python
import pygame
import sys

pygame.init()
screen = pygame.display.set_mode((900, 700))
pygame.display.set_caption("Hogwarts Enchanted Maze")
clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((20, 20, 35))
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
```

Run:

```powershell
python main.py
```

## 7) AI modules mapping (your lab requirements)

- `ai/astar.py`: grid navigation for artifact hunt, chase, and escape.
- `ai/minimax.py`: duel move prediction and counter-move selection.
- `ai/fuzzy_logic.py`: behavior mode selection:
  - Aggressive Duelist
  - Defensive Escape
  - Tactical Avoidance

## 8) Recommended next implementation order

1. Build maze grid + walls + random artifact spawning.
2. Add two agents (Phoenix and Serpent) with MP state.
3. Implement A* movement.
4. Add fuzzy behavior selector from MP and distance.
5. Add Minimax duel decision layer.
6. Integrate full game loop and win conditions.

## Note about `pygame` on newer Python

If `pip install pygame` fails on Python 3.14+, use:

```powershell
pip install pygame-ce
```

`pygame-ce` is a drop-in replacement imported the same way:

```python
import pygame
```

## 9) Quick start command (always on D drive)

```powershell
D:
cd "D:\AI\hogwarts-maze-game"
.\.venv\Scripts\Activate.ps1
python main.py
```
