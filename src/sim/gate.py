"""El punto donde algo puede vetar una orden antes de que llegue al venue.

Esto es una **costura**, no una politica. La politica concreta vive en
``src/risk/`` y no la conoce nadie de ``sim/``: el simulador no debe saber que
existe un limite de perdida diaria, del mismo modo que un broker real no lo
sabe. La dependencia va en un solo sentido, ``risk`` -> ``sim``, y por eso el
runner en vivo de la Etapa 7 puede insertar exactamente la misma capa de riesgo
entre la estrategia y el adaptador de broker.

Un veto **no** produce un ``Fill``. La orden no llego al venue, asi que no hay
participacion, ni precio de referencia, ni motivo del venue que reportar.
Mezclarlo con los rechazos del venue en la misma lista haria indistinguible
"el venue no pudo llenarla" de "nuestra propia capa de riesgo no la dejo salir",
que son dos diagnosticos completamente distintos. Va en su propia lista,
``SimResult.gate_rejections``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from sim.orders import Fill, MarketOrder
from sim.view import AccountSnapshot


@dataclass(frozen=True)
class GateDecision:
    """Veredicto sobre una orden. El motivo es obligatorio si se rechaza."""

    approved: bool
    reason: str = ""
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.approved and not self.reason:
            raise ValueError("un rechazo debe declarar el motivo")


@dataclass(frozen=True)
class GateRejection:
    """Una orden que nunca salio. Queda registrada con su motivo.

    El proyecto prohibe que una estrategia se autocensure en silencio: hace
    indistinguible "no quiso operar" de "no pudo". Lo mismo vale para la capa de
    riesgo, con mas razon: si un limite esta bloqueando al agente en cada barra,
    eso tiene que verse en el log del experimento y no deducirse de que la curva
    de equity esta plana.
    """

    t_decision: int
    timestamp_decision: np.datetime64
    symbol: str
    client_order_id: str
    qty_requested: float
    reason: str
    detail: str = ""
    tag: str = ""


@runtime_checkable
class OrderGate(Protocol):
    """Filtro entre la decision de la estrategia y el envio al venue.

    ``observe_bar`` y ``observe_fill`` existen porque un limite de riesgo suele
    depender de estado acumulado -la perdida del dia, el turnover del dia- que
    el gate no puede reconstruir mirando solo la orden que tiene enfrente.
    """

    def check(self, order: MarketOrder, account: AccountSnapshot) -> GateDecision: ...

    def observe_bar(self, account: AccountSnapshot) -> None: ...

    def observe_fill(self, fill: Fill) -> None: ...
