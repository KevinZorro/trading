"""Modelos de costo de transaccion: spread, slippage y comisiones.

Invariantes del modulo:

1. Spread y slippage son **siempre adversos**. Nunca mejoran el precio de
   referencia. Un modelo de slippage con ruido centrado en cero a veces regala
   un precio mejor que el de referencia, y eso no existe en un mercado.
2. Todo modelo se evalua con informacion disponible **hasta la barra de
   decision** ``t``, aunque el fill ocurra en ``t+1``. En particular, estimar el
   spread desde el rango de la propia barra de ejecucion seria lookahead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from data.instruments import CommissionSchema, InstrumentSpec


@dataclass(frozen=True)
class SpreadContext:
    """Informacion disponible para estimar el spread de una ejecucion.

    ``high``, ``low`` y ``close`` llegan hasta la barra de decision ``t``
    inclusive. La barra de ejecucion ``t+1`` no esta y no debe estarlo.
    """

    ref_price: float
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray


@runtime_checkable
class SpreadModel(Protocol):
    name: str

    def half_spread(self, ctx: SpreadContext) -> float:
        """Medio spread en unidades monetarias. Siempre >= 0."""
        ...


@dataclass(frozen=True)
class ZeroSpread:
    """Sin spread. Solo para tests de invariantes contables."""

    name: str = "zero"

    def half_spread(self, ctx: SpreadContext) -> float:
        return 0.0


@dataclass(frozen=True)
class FixedBpsSpread:
    """Spread fijo en puntos basicos sobre el precio de referencia."""

    bps: float
    name: str = "fixed_bps"

    def __post_init__(self) -> None:
        if self.bps < 0:
            raise ValueError("bps no puede ser negativo")

    def half_spread(self, ctx: SpreadContext) -> float:
        return ctx.ref_price * self.bps / 2e4


@dataclass(frozen=True)
class CorwinSchultzSpread:
    """Estimador de Corwin-Schultz (2012) a partir de dos barras consecutivas.

    Se prefiere al rango crudo ``high - low``, que sobreestima el spread justo
    en las barras volatiles, que son las que mas pesan en el tier de riesgo alto.
    El estimador separa la parte del rango atribuible a volatilidad de la
    atribuible al spread usando que la primera escala con el tiempo y la
    segunda no.

    ASSUMPTION: la formula supone continuidad de precios entre las dos barras.
    Ante un gap grande el estimador puede volverse negativo; en ese caso se
    trunca a cero, que es la convencion de los autores. Se aplica ademas un
    tope (``max_bps``) para que una barra patologica no genere un costo absurdo.
    """

    max_bps: float = 200.0
    fallback: SpreadModel | None = None
    name: str = "corwin_schultz"

    _K = 3.0 - 2.0 * math.sqrt(2.0)

    def half_spread(self, ctx: SpreadContext) -> float:
        if len(ctx.high) < 2:
            # Sin dos barras no hay estimador; se usa el fallback declarado.
            if self.fallback is None:
                return 0.0
            return self.fallback.half_spread(ctx)

        h1, h0 = float(ctx.high[-1]), float(ctx.high[-2])
        l1, l0 = float(ctx.low[-1]), float(ctx.low[-2])

        beta = math.log(h0 / l0) ** 2 + math.log(h1 / l1) ** 2
        gamma = math.log(max(h0, h1) / min(l0, l1)) ** 2
        alpha = (math.sqrt(2.0 * beta) - math.sqrt(beta)) / self._K - math.sqrt(
            gamma / self._K
        )
        spread_pct = 2.0 * (math.exp(alpha) - 1.0) / (1.0 + math.exp(alpha))
        spread_pct = min(max(spread_pct, 0.0), self.max_bps / 1e4)
        return ctx.ref_price * spread_pct / 2.0


@runtime_checkable
class SlippageModel(Protocol):
    name: str

    def impact(self, participation: float) -> float:
        """Impacto como fraccion del precio de referencia. Siempre >= 0."""
        ...


@dataclass(frozen=True)
class NoSlippage:
    name: str = "none"

    def impact(self, participation: float) -> float:
        return 0.0


@dataclass(frozen=True)
class LinearSlippage:
    """Impacto lineal en la participacion sobre el volumen de la barra."""

    k: float
    name: str = "linear"

    def __post_init__(self) -> None:
        if self.k < 0:
            raise ValueError("k no puede ser negativo: el slippage es adverso")

    def impact(self, participation: float) -> float:
        return self.k * max(participation, 0.0)


@dataclass(frozen=True)
class SqrtSlippage:
    """Impacto en raiz de la participacion (ley de raiz cuadrada)."""

    k: float
    name: str = "sqrt"

    def __post_init__(self) -> None:
        if self.k < 0:
            raise ValueError("k no puede ser negativo: el slippage es adverso")

    def impact(self, participation: float) -> float:
        return self.k * math.sqrt(max(participation, 0.0))


@runtime_checkable
class CommissionModel(Protocol):
    name: str

    def compute(self, qty: float, price: float) -> float:
        """Comision en unidades monetarias. Siempre >= 0."""
        ...


@dataclass(frozen=True)
class SchemaCommission:
    """Comision derivada del :class:`CommissionSchema` del instrumento.

    Vive aqui y no en ``data`` porque es la parte ejecutable; el esquema es solo
    datos y viaja serializado junto al resultado del experimento.
    """

    schema: CommissionSchema
    name: str = "schema"

    def compute(self, qty: float, price: float) -> float:
        notional = abs(qty) * price
        if self.schema.kind == "fixed":
            fee = self.schema.value
        elif self.schema.kind == "percent":
            fee = notional * self.schema.value
        else:  # per_share
            fee = abs(qty) * self.schema.value
        fee = max(fee, self.schema.minimum)
        if self.schema.max_pct_notional is not None:
            fee = min(fee, notional * self.schema.max_pct_notional)
        return fee


@dataclass(frozen=True)
class ZeroCommission:
    name: str = "zero"

    def compute(self, qty: float, price: float) -> float:
        return 0.0


def commission_from_instrument(instrument: InstrumentSpec) -> CommissionModel:
    return SchemaCommission(schema=instrument.commission_schema)
