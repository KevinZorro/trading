"""Autoterminacion de la sombra ante un cuelgue (ADR 0007, PR 6).

``restart: unless-stopped`` solo reinicia si el proceso *termina*; un proceso
colgado (por ejemplo, un socket sin respuesta) queda vivo, sin escribir y sin
que Docker haga nada. ``StallGuard`` cierra ese hueco: si pasa ``limit`` sin
progreso, llama a ``on_stall`` (en produccion, ``os._exit(1)``) y Docker lo
levanta. El watchdog no necesita el socket de Docker.

ASSUMPTION: salir de golpe es seguro porque cada snapshot ya esta en disco con
``fsync``; lo unico que se pierde es el que estaba en curso, que de todos modos
no habia terminado de escribirse.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable

import numpy as np

from sim.clock import Clock

CODIGO_SALIDA = 70  # distinto de 0: Docker lo cuenta como caida


class StallGuard:
    def __init__(
        self, clock: Clock, limit: np.timedelta64, on_stall: Callable[[], None]
    ) -> None:
        self._clock = clock
        self._limit = limit
        self._on_stall = on_stall
        self._last = clock.now()

    def touch(self) -> None:
        self._last = self._clock.now()

    def check(self) -> bool:
        """``True`` (y llama a ``on_stall``) si el plazo vencio sin ``touch``."""
        if self._clock.now() - self._last > self._limit:
            self._on_stall()
            return True
        return False

    def start(self, stop: threading.Event, period_s: float = 30.0) -> threading.Thread:
        def vigilar() -> None:  # pragma: no cover - hilo real
            while not stop.wait(period_s):
                self.check()

        hilo = threading.Thread(target=vigilar, name="stall-guard", daemon=True)
        hilo.start()
        return hilo


def exit_process() -> None:  # pragma: no cover - mata el proceso
    os._exit(
        CODIGO_SALIDA
    )  # os._exit: el hilo principal esta colgado, no hay cierre limpio
