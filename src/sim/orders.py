"""Ordenes y registro de ejecuciones con costos desglosados.

La descomposicion del log es deliberada y es el corazon de la atribucion de P&L
del proyecto. Frente al precio que la estrategia vio al decidir (``close[t]``),
la diferencia hasta el efectivo pagado se parte en cuatro terminos::

    shortfall = gap + spread + slippage + comision

- ``gap``: ``qty * (open[t+1] - close[t])``. Es un efecto de **valuacion**,
  no un costo: nace de que la ejecucion ocurre una barra despues de la senal.
  Lleva signo y **puede ser favorable**. En el regimen de riesgo alto domina
  sobre todo lo demas.
- ``spread``, ``slippage``, ``comision``: costos de transaccion propiamente
  dichos. Nunca negativos.

Mezclar el gap con los costos hace imposible distinguir despues "el agente
opera demasiado" de "el agente opera en el peor momento del ciclo overnight".
Por eso ``total_costs_paid`` suma solo los tres ultimos terminos y el gap se
reporta siempre en su propia linea.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from data.instruments import RejectReason as InstrumentRejectReason


class OrderStatus(StrEnum):
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class RejectReason(StrEnum):
    """Motivos de rechazo del venue (heredados) y del simulador."""

    MIN_QTY = "MIN_QTY"
    MIN_NOTIONAL = "MIN_NOTIONAL"
    ZERO_AFTER_ROUNDING = "ZERO_AFTER_ROUNDING"
    INSUFFICIENT_CASH = "INSUFFICIENT_CASH"
    INSUFFICIENT_POSITION = "INSUFFICIENT_POSITION"
    SHORT_NOT_ALLOWED = "SHORT_NOT_ALLOWED"
    NO_VOLUME = "NO_VOLUME"
    # Rechazos de admision: ocurren al enviar, no al llenar.
    MISSING_CLIENT_ORDER_ID = "MISSING_CLIENT_ORDER_ID"
    VENUE_NOT_RUNNING = "VENUE_NOT_RUNNING"

    @classmethod
    def from_instrument(cls, reason: InstrumentRejectReason) -> RejectReason:
        return cls(reason.value)


@dataclass(frozen=True)
class MarketOrder:
    """Orden de mercado en cantidad con signo.

    Es el unico primitivo de ejecucion de la Etapa 1. ``qty > 0`` compra,
    ``qty < 0`` vende. Los helpers de dimensionamiento (por peso, por posicion
    objetivo) resuelven a esto usando datos de la barra de decision.

    ASSUMPTION: no hay ordenes limit. Modelar fills de limit sin libro de
    ordenes real produce ejecuciones optimistas imposibles de auditar.
    """

    qty: float
    tag: str = ""
    # Lo genera el emisor, no el venue. Vacio significa "todavia no asignado":
    # el runner lo completa antes de enviar y el venue rechaza lo que llegue
    # sin el. Ver `sim.ids` para por que el emisor y no el venue.
    client_order_id: str = ""

    def __post_init__(self) -> None:
        if not np.isfinite(self.qty):
            raise ValueError(f"qty no finita: {self.qty}")

    @property
    def side(self) -> str:
        return "BUY" if self.qty > 0 else "SELL"


@dataclass(frozen=True)
class Fill:
    """Una linea del log de ejecuciones, con los costos desglosados."""

    t_decision: int
    t_fill: int
    timestamp_decision: np.datetime64
    timestamp_fill: np.datetime64
    symbol: str
    status: OrderStatus
    qty_requested: float
    qty_filled: float
    decision_price: float
    ref_price: float
    fill_price: float
    gap: float
    spread_cost: float
    slippage_cost: float
    commission: float
    participation: float
    reject_reason: RejectReason | None = None
    tag: str = ""
    client_order_id: str = ""

    @property
    def transaction_costs(self) -> float:
        """Costos de transaccion. **No incluye el gap**, que no es un costo."""
        return self.spread_cost + self.slippage_cost + self.commission

    @property
    def implementation_shortfall(self) -> float:
        """Desviacion total frente a ejecutar al precio de decision."""
        return self.gap + self.transaction_costs

    @property
    def notional(self) -> float:
        return abs(self.qty_filled) * self.fill_price

    @property
    def is_executed(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.PARTIAL)
