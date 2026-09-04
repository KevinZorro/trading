"""La interfaz contra la que se opera, en simulacion y en vivo.

``Simulator`` implementa este protocolo; en la Etapa 7 lo implementara tambien
un adaptador de broker. El runner y la capa de riesgo hablan con
``ExecutionVenue`` y no saben cual de los dos tienen enfrente. Esa es toda la
paridad backtest-live: no hay ramas condicionales por entorno porque no hay
nada que ramificar.

Dos formas que el protocolo impone a proposito, porque son las que hacen que el
codigo escrito contra el simulador funcione en vivo sin reescribirse:

1. **``submit_order`` acusa recepcion, no ejecucion.** Devuelve ``OrderAck``,
   no ``Fill``. En vivo la orden viaja, el venue la acepta y el fill llega
   despues, por otro canal. Un simulador que devolviera el fill en el mismo
   llamado ensenaria a escribir estrategias que no pueden existir en vivo. La
   validacion queda partida en dos, como en un broker real: la de **admision**
   (identificador duplicado, cantidad no finita) ocurre al enviar, y la de
   **ejecucion** (volumen de la barra, minimos del instrumento, cash) al
   llenar.

2. **``reconcile`` existe desde el primer dia.** El estado autoritativo lo
   reporta el venue, nunca la memoria de la estrategia. En vivo, entre dos
   arranques del proceso pudo pasar cualquier cosa: un fill parcial, una
   cancelacion del broker, una liquidacion forzada. Arrancar asumiendo la
   posicion que uno recuerda es como se pierde una cuenta.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from sim.orders import Fill, MarketOrder, RejectReason


@dataclass(frozen=True)
class OrderAck:
    """Acuse de **recepcion**. No dice nada sobre si la orden se lleno."""

    client_order_id: str
    accepted: bool
    reject_reason: RejectReason | None = None
    detail: str = ""
    is_duplicate: bool = False

    def __post_init__(self) -> None:
        if self.accepted and self.reject_reason is not None:
            raise ValueError("una orden aceptada no puede llevar motivo de rechazo")
        if not self.accepted and self.reject_reason is None:
            raise ValueError("una orden rechazada debe declarar el motivo")


@dataclass(frozen=True)
class CancelAck:
    """Resultado de intentar cancelar. ``cancelled=False`` no es un error.

    Una orden que ya se ejecuto no se puede cancelar, y eso es informacion, no
    una falla: en vivo es la carrera normal entre la cancelacion que sale y el
    fill que entra.
    """

    client_order_id: str
    cancelled: bool
    detail: str = ""


@dataclass(frozen=True)
class VenueState:
    """Estado autoritativo del venue en un instante.

    Lo que devuelve ``reconcile``. La estrategia adopta esto como verdad; su
    propia memoria del estado no cuenta.
    """

    timestamp: np.datetime64
    cash: float
    positions: dict[str, float] = field(default_factory=dict)
    open_order_ids: tuple[str, ...] = ()

    def position(self, symbol: str) -> float:
        return self.positions.get(symbol, 0.0)


@runtime_checkable
class ExecutionVenue(Protocol):
    """Lo minimo que hay que poder hacerle a un venue para operar contra el."""

    def submit_order(self, order: MarketOrder) -> OrderAck:
        """Envia una orden. La orden **debe** traer ``client_order_id``.

        Reenviar un identificador ya visto no ejecuta de nuevo: devuelve el
        mismo acuse con ``is_duplicate=True``.
        """
        ...

    def cancel_order(self, client_order_id: str) -> CancelAck:
        """Intenta cancelar una orden viva."""
        ...

    def get_positions(self) -> dict[str, float]:
        """Posiciones segun el venue. Copia, no la estructura interna."""
        ...

    def get_fills(self, since: int = 0) -> list[Fill]:
        """Fills desde el indice ``since`` del log. Sirve de numero de secuencia."""
        ...

    def reconcile(self) -> VenueState:
        """Foto completa del estado autoritativo. Se llama siempre al arrancar."""
        ...
