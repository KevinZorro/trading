"""Torch se importa antes que numpy, pero solo en Windows y si está instalado."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from agents._torch_first import preload_torch

RAIZ = Path(__file__).resolve().parents[2]


def test_en_windows_con_torch_lo_importa() -> None:
    importados: list[str] = []
    assert preload_torch("win32", lambda nombre: object(), importados.append)
    assert importados == ["torch"]


def test_en_windows_sin_torch_no_hace_nada() -> None:
    importados: list[str] = []
    assert not preload_torch("win32", lambda nombre: None, importados.append)
    assert importados == []


def test_fuera_de_windows_no_importa_nada() -> None:
    importados: list[str] = []
    assert not preload_torch("linux", lambda nombre: object(), importados.append)
    assert importados == []


def test_el_paquete_lo_importa_antes_que_numpy() -> None:
    """``agents`` carga el módulo antes que cualquier otro que traiga numpy."""
    codigo = (
        "import sys; import agents; "
        "orden = [m for m in sys.modules if m in ('agents._torch_first', 'numpy')]; "
        "print(orden)"
    )
    salida = subprocess.run(
        [sys.executable, "-c", codigo],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(RAIZ / "src")},
    ).stdout.strip()
    assert salida == "['agents._torch_first', 'numpy']"
