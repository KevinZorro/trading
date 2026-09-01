"""Especificacion de instrumento: las reglas de ejecucion del venue.

La fraccionalidad, el lote y los minimos NO son configuracion global del
simulador: son propiedades del instrumento. Una accion de EE.UU. en un broker
sin fraccionales y BTCUSDT en Binance viven en el mismo experimento y tienen
restricciones distintas. El barrido de capital de la Etapa 6 mide exactamente
este parametro, asi que se modela de forma explicita desde el principio.

Precision que importa: en cripto la restriccion que muerde es el **nocional
minimo** (~5-10 USD), no la cantidad minima (BTCUSDT admite 1e-5 BTC). Si solo
se modelara ``min_order_qty`` se concluiria que 10 USD opera bien en BTC cuando
en realidad el venue rechaza la orden. Ambas se evaluan siempre y por separado.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

# Tolerancia relativa para comparar cantidades y nocionales contra los minimos.
# Sin ella, 0.1 + 0.2 en binario rechaza una orden que el venue aceptaria.
_EPS = 1e-9


class RejectReason(StrEnum):
    """Motivos de rechazo atribuibles al instrumento.

    El simulador anade sus propios motivos (cash insuficiente, short no
    permitido); estos son los que dependen solo del venue.
    """

    MIN_QTY = "MIN_QTY"
    MIN_NOTIONAL = "MIN_NOTIONAL"
    ZERO_AFTER_ROUNDING = "ZERO_AFTER_ROUNDING"


CommissionKind = Literal["fixed", "percent", "per_share"]


@dataclass(frozen=True)
class CommissionSchema:
    """Descripcion declarativa del esquema de comisiones del venue.

    Es solo datos: ``src/sim`` construye el modelo ejecutable a partir de esto.
    Se mantiene en ``data`` para que la spec del instrumento sea serializable y
    viaje junto al resultado del experimento sin arrastrar dependencias.

    - ``fixed``: ``value`` por orden.
    - ``percent``: ``value`` como fraccion del nocional (0.001 == 10 bps).
    - ``per_share``: ``value`` por unidad negociada.
    """

    kind: CommissionKind = "percent"
    value: float = 0.0
    minimum: float = 0.0
    max_pct_notional: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("fixed", "percent", "per_share"):
            raise ValueError(f"kind de comision desconocido: {self.kind!r}")
        if self.value < 0:
            raise ValueError("value de comision no puede ser negativo")
        if self.minimum < 0:
            raise ValueError("minimum de comision no puede ser negativo")
        if self.max_pct_notional is not None and self.max_pct_notional <= 0:
            raise ValueError("max_pct_notional debe ser positivo o None")


@dataclass(frozen=True)
class InstrumentSpec:
    """Reglas de negociacion de un simbolo en un venue concreto."""

    symbol: str
    venue: str
    tick_size: float
    lot_size: float
    allow_fractional: bool
    qty_precision: int
    min_order_qty: float
    min_notional: float
    commission_schema: CommissionSchema
    asset_class: str = "equity"

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol vacio")
        if self.tick_size <= 0:
            raise ValueError("tick_size debe ser positivo")
        if self.lot_size <= 0:
            raise ValueError("lot_size debe ser positivo")
        if self.qty_precision < 0:
            raise ValueError("qty_precision no puede ser negativa")
        if self.min_order_qty < 0 or self.min_notional < 0:
            raise ValueError("los minimos no pueden ser negativos")
        if not self.allow_fractional and self.lot_size < 1.0:
            # ASSUMPTION: sin fraccionales el lote representa unidades enteras.
            raise ValueError(
                "allow_fractional=False exige lot_size >= 1 (unidades enteras)"
            )

    # -- redondeo -------------------------------------------------------

    def round_qty(self, qty: float) -> float:
        """Redondea una cantidad con signo al lote negociable, **hacia cero**.

        Nunca hacia arriba: redondear hacia arriba con capital bajo fabrica
        ejecuciones que el cash no soporta.
        """
        if not math.isfinite(qty):
            raise ValueError(f"qty no finita: {qty}")
        sign = -1.0 if qty < 0 else 1.0
        magnitude = abs(qty)
        lots = math.floor(magnitude / self.lot_size + _EPS)
        rounded = lots * self.lot_size
        if self.allow_fractional:
            scale = 10**self.qty_precision
            rounded = math.floor(rounded * scale + _EPS) / scale
        return sign * rounded

    def round_price(self, price: float) -> float:
        """Redondea un precio al tick mas cercano."""
        if not math.isfinite(price):
            raise ValueError(f"precio no finito: {price}")
        return round(price / self.tick_size) * self.tick_size

    def is_price_on_tick(self, price: float, rel_tol: float = 1e-6) -> bool:
        ratio = price / self.tick_size
        return abs(ratio - round(ratio)) <= max(rel_tol * abs(ratio), _EPS)

    # -- admisibilidad --------------------------------------------------

    def check_tradable(
        self, qty: float, price: float
    ) -> tuple[float, RejectReason | None]:
        """Devuelve ``(qty_redondeada, motivo_de_rechazo | None)``.

        Evalua **las dos** restricciones por separado y reporta cual mordio.
        Un rechazo es un rechazo: no se redimensiona la orden en silencio.
        """
        if price <= 0:
            raise ValueError(f"precio no positivo en check_tradable: {price}")
        rounded = self.round_qty(qty)
        if rounded == 0.0:
            return 0.0, RejectReason.ZERO_AFTER_ROUNDING
        if abs(rounded) + _EPS < self.min_order_qty:
            return rounded, RejectReason.MIN_QTY
        if abs(rounded) * price + _EPS < self.min_notional:
            return rounded, RejectReason.MIN_NOTIONAL
        return rounded, None


def us_equity_spec(
    symbol: str,
    *,
    venue: str = "XNAS",
    commission: CommissionSchema | None = None,
    allow_fractional: bool = False,
) -> InstrumentSpec:
    """Accion de EE.UU. en un broker retail tipico.

    ASSUMPTION: sin fraccionales por defecto y sin nocional minimo; la
    restriccion que muerde con capital bajo es la unidad entera.
    """
    return InstrumentSpec(
        symbol=symbol,
        venue=venue,
        tick_size=0.01,
        lot_size=1.0 if not allow_fractional else 1e-6,
        allow_fractional=allow_fractional,
        qty_precision=0 if not allow_fractional else 6,
        min_order_qty=1.0 if not allow_fractional else 1e-6,
        min_notional=0.0,
        commission_schema=commission
        or CommissionSchema(
            kind="per_share", value=0.005, minimum=1.0, max_pct_notional=0.01
        ),
        asset_class="equity",
    )


def binance_spot_spec(
    symbol: str = "BTCUSDT",
    *,
    tick_size: float = 0.01,
    step_size: float = 1e-5,
    min_notional: float = 10.0,
    commission: CommissionSchema | None = None,
) -> InstrumentSpec:
    """Par spot de Binance.

    ASSUMPTION: valores por defecto tomados de los filtros tipicos de BTCUSDT
    (PRICE_FILTER 0.01, LOT_SIZE 1e-5, NOTIONAL 10 USDT) y comision taker de
    10 bps sin descuentos. Los filtros reales cambian por par y en el tiempo;
    para el experimento final se cargan del snapshot de ``exchangeInfo``
    vigente en cada fecha.
    """
    return InstrumentSpec(
        symbol=symbol,
        venue="BINANCE",
        tick_size=tick_size,
        lot_size=step_size,
        allow_fractional=True,
        qty_precision=8,
        min_order_qty=step_size,
        min_notional=min_notional,
        commission_schema=commission or CommissionSchema(kind="percent", value=0.001),
        asset_class="crypto",
    )
