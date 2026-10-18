"""Watchdog de la sombra (ADR 0007, PR 6): vigila el heartbeat y avisa.

Corre en un contenedor aparte con el directorio de datos en solo lectura: un
proceso colgado no puede vigilarse a si mismo. Avisa en la transicion a fallo,
recuerda cada ``REPETIR`` mientras siga caido y avisa al recuperarse. Si un
aviso no sale, no cuenta como enviado y se reintenta en el siguiente ciclo.

El tiempo entra por el ``Clock`` y la espera por ``sleep``.
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
from collections.abc import Callable, Mapping

import numpy as np

from live.alerts import Notifier, build_notifier
from live.config import ShadowConfig
from live.health import Health, check
from live.logs import configure_json_logging
from sim.clock import Clock, SystemClock

CICLO_S = 300.0
REPETIR = np.timedelta64(6, "h")

log = logging.getLogger(__name__)


class Watchdog:
    def __init__(
        self,
        config: ShadowConfig,
        clock: Clock,
        notifier: Notifier,
        *,
        sleep: Callable[[float], object],
    ) -> None:
        self._config = config
        self._clock = clock
        self._notifier = notifier
        self._sleep = sleep
        self._motivos: tuple[str, ...] = ()  # lo ya avisado; vacio = sano
        self._ultimo_aviso: np.datetime64 | None = None

    def step(self) -> Health:
        ahora = self._clock.now()
        salud = check(self._config.data_dir, ahora, self._config)
        if salud.ok:
            if self._motivos and self._notifier.send(
                "Sombra recuperada: heartbeat sano."
            ):
                self._motivos = ()
                self._ultimo_aviso = None
            return salud
        nuevo = salud.reasons != self._motivos
        vencido = self._ultimo_aviso is None or ahora - self._ultimo_aviso >= REPETIR
        if nuevo or vencido:
            texto = "ALERTA sombra: " + "; ".join(salud.reasons)
            if self._notifier.send(texto):
                self._motivos = salud.reasons
                self._ultimo_aviso = ahora
        return salud

    def run(self, stop: Callable[[], bool]) -> None:
        while not stop():
            self.step()
            self._sleep(CICLO_S)


def main(environ: Mapping[str, str]) -> int:  # pragma: no cover - proceso real
    configure_json_logging(sys.stdout)
    config = ShadowConfig.from_env(environ)
    parar = threading.Event()
    for senal in (signal.SIGTERM, signal.SIGINT):
        signal.signal(senal, lambda *_: parar.set())
    Watchdog(config, SystemClock(), build_notifier(config), sleep=parar.wait).run(
        parar.is_set
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    import os

    raise SystemExit(main(os.environ))
