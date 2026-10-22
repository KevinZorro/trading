"""En Windows, torch tiene que importarse antes que numpy y pandas.

Medido en el PC de Kevin: ``import torch; import numpy, pandas`` funciona y
``import numpy, pandas; import torch`` falla con ``WinError 1114`` al cargar
``c10.dll``. ``agents.cli`` importaba torch de forma diferida, después de
numpy, y la calibración (ADR 0010) no podía entrenar. ``agents/__init__`` importa
este módulo primero, antes de cualquier otro, para que el orden no dependa de
quien lo invoque. En otras plataformas, o sin el grupo ``rl``, no hace nada.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Callable
from typing import Any


def preload_torch(
    platform: str = sys.platform,
    find_spec: Callable[[str], Any] = importlib.util.find_spec,
    import_module: Callable[[str], Any] = importlib.import_module,
) -> bool:
    """Importa torch si es Windows y está instalado. Devuelve si lo importó."""
    if platform != "win32" or find_spec("torch") is None:
        return False
    import_module("torch")
    return True


preload_torch()
