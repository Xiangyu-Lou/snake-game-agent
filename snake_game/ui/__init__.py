"""pygame front-end: the interactive app, the board renderer and the environment viewer.

Needs the optional ``pygame-ce`` dependency (``pip install -e ".[ui]"``); nothing
in the engine or the environments imports it unless rendering is requested.
"""

from .app import SPEEDS, SnakeApp, play
from .renderer import BoardRenderer
from .viewer import EnvViewer

__all__ = ["SPEEDS", "BoardRenderer", "EnvViewer", "SnakeApp", "play"]
