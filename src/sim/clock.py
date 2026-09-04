"""El tiempo entra por aqui y solo por aqui.

Ninguna otra parte del proyecto consulta el reloj del sistema. La razon no es
estetica: una capa de riesgo con limite de perdida diaria necesita saber cuando
empieza un dia, y si lo pregunta a ``datetime.now()`` entonces el backtest y el
sistema en vivo ejecutan codigo distinto. En backtest "ahora" es el timestamp de
la barra que se esta procesando, no la hora de la maquina que corre el
experimento; sin esta inyeccion, correr el mismo backtest un martes y un
domingo daria resultados distintos.

Hay un test que verifica que ``SystemClock`` es el unico lugar de ``src/`` que
consulta el reloj del sistema. Si alguien agrega un ``datetime.now()`` en otro
modulo, falla.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Clock(Protocol):
    """Fuente de tiempo inyectable.

    Devuelve ``datetime64[ns]`` en UTC, el mismo tipo que usan los timestamps de
    ``BarSeries``, para que comparar la hora actual contra una barra no requiera
    conversiones.
    """

    def now(self) -> np.datetime64: ...


class SimulatedClock:
    """Reloj del backtest. Lo avanza el motor, barra por barra.

    ``advance_to`` es deliberadamente publico pero solo lo llama el motor: una
    estrategia recibe ``MarketView`` y ``AccountSnapshot``, nunca el reloj, asi
    que no puede adelantarlo para espiar el futuro. Y aunque lo adelantara, el
    reloj no da acceso a datos: la barra siguiente sigue fuera de la vista.
    """

    __slots__ = ("_now",)

    def __init__(self, initial: np.datetime64) -> None:
        self._now = np.datetime64(initial, "ns")

    def now(self) -> np.datetime64:
        return self._now

    def advance_to(self, timestamp: np.datetime64) -> None:
        """Mueve el reloj a la barra actual. Dentro de una corrida no retrocede.

        Que el tiempo no retroceda es lo que hace confiable un limite de perdida
        diaria: si el reloj pudiera ir hacia atras, el contador del dia se
        reiniciaria a mitad de la corrida y el limite dejaria de morder.
        """
        nuevo = np.datetime64(timestamp, "ns")
        if nuevo < self._now:
            raise ValueError(
                f"el reloj no puede retroceder: esta en {self._now} y se pidio "
                f"{nuevo}. Para empezar una corrida nueva usa reset_to."
            )
        self._now = nuevo

    def reset_to(self, timestamp: np.datetime64) -> None:
        """Rebobina para empezar otra corrida. Explicito a proposito.

        Es la unica forma de mover el reloj hacia atras, y existe porque un
        ``Simulator`` se puede correr mas de una vez (un barrido de semillas
        reusa la instancia). Separarlo de ``advance_to`` mantiene la garantia de
        arriba dentro de cada corrida sin impedir que empiece la siguiente.
        """
        self._now = np.datetime64(timestamp, "ns")


class SystemClock:
    """Reloj de pared. **Unico punto del repositorio que lo consulta.**

    No se usa en simulacion. Existe para que el runner en vivo de la Etapa 7
    inyecte esto donde el backtest inyecta ``SimulatedClock``, sin una sola rama
    condicional por entorno.
    """

    __slots__ = ()

    def now(self) -> np.datetime64:
        return np.datetime64(datetime.now(UTC).replace(tzinfo=None), "ns")
