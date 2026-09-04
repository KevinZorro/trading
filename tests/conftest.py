"""Guardas globales de la suite: sin red y con aleatoriedad sembrada.

Este archivo se importa antes de recoger cualquier test, asi que los bloqueos
que instala cubren tambien el codigo que corre en tiempo de importacion de un
modulo de test.

Por que un guard de red y no solo disciplina: un test que descarga precios de
una API pasa hoy y falla dentro de seis meses cuando la API cambia, o -peor-
sigue pasando con datos distintos y vuelve irreproducible un resultado del
estudio. Los datos entran al repositorio por ``data/loaders`` desde archivos
locales; cualquier socket de red en la suite es un bug, no una dependencia.
"""

from __future__ import annotations

import random
import socket
from typing import Any

import numpy as np
import pytest

# Semilla de las fuentes globales de aleatoriedad. El proyecto usa
# `np.random.default_rng(seed)` explicito en todas partes, asi que esto solo
# cubre un uso accidental del RNG global: lo vuelve reproducible en vez de
# silenciosamente distinto en cada corrida.
GLOBAL_SEED = 20240101


class NetworkAccessError(RuntimeError):
    """Un test intento abrir una conexion de red."""


# AF_UNIX no sale de la maquina: pytest-xdist y algunas herramientas lo usan
# para comunicarse entre procesos. Bloquearlo no aporta aislamiento y si rompe
# el tooling.
_FAMILIAS_PERMITIDAS = {getattr(socket, "AF_UNIX", None)} - {None}

_connect_original = socket.socket.connect
_connect_ex_original = socket.socket.connect_ex
_create_connection_original = socket.create_connection


def _explotar(destino: Any) -> NetworkAccessError:
    return NetworkAccessError(
        f"un test intento conectarse a {destino!r}. La suite corre sin red: "
        "los datos se cargan desde archivos locales o se generan con "
        "`data.synthetic`. Si necesitas un dataset nuevo, versionalo o "
        "generalo con una semilla fija."
    )


def _connect_bloqueado(self: socket.socket, address: Any) -> None:
    if self.family in _FAMILIAS_PERMITIDAS:
        _connect_original(self, address)
        return
    raise _explotar(address)


def _connect_ex_bloqueado(self: socket.socket, address: Any) -> int:
    if self.family in _FAMILIAS_PERMITIDAS:
        return _connect_ex_original(self, address)
    raise _explotar(address)


def _create_connection_bloqueado(address: Any, *args: Any, **kwargs: Any) -> Any:
    raise _explotar(address)


socket.socket.connect = _connect_bloqueado  # type: ignore[method-assign]
socket.socket.connect_ex = _connect_ex_bloqueado  # type: ignore[method-assign]
socket.create_connection = _create_connection_bloqueado  # type: ignore[assignment]


@pytest.fixture(autouse=True)
def _aleatoriedad_sembrada() -> None:
    """Siembra las fuentes globales antes de cada test."""
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)  # noqa: NPY002
