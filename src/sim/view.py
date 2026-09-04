"""La ventana de mercado que ve la estrategia.

Restriccion estructural del proyecto: una estrategia que decide en ``t`` no
puede acceder a datos de ``t+1``. No se logra con disciplina ni con revisiones
de codigo, se logra con la forma del objeto:

- El simulador conduce el bucle y **entrega** la ventana; la estrategia nunca
  recibe la serie completa ni un cursor que pueda adelantar.
- ``MarketView`` guarda slices ``[0..t]``. Los datos de ``t+1`` en adelante no
  estan referenciados por el objeto.
- Los ``lookback`` son no negativos por contrato: ``lookback=0`` es la barra
  actual y ``lookback=-1`` (el error clasico "la barra siguiente") lanza
  :class:`LookaheadError` en vez de devolver el futuro.
- ``history`` devuelve **copias** de solo lectura, para que no se pueda navegar
  al array subyacente por ``ndarray.base``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.errors import LookaheadError
from data.instruments import InstrumentSpec
from data.schema import OHLCV_FIELDS, BarSeries


class MarketView:
    """Vista inmutable de la serie hasta la barra ``t`` inclusive."""

    __slots__ = ("_fields", "_instrument", "_t", "_timestamp")

    def __init__(self, series: BarSeries, t: int) -> None:
        if not 0 <= t < len(series):
            raise IndexError(f"t={t} fuera de rango para n={len(series)}")
        # Slices, no la serie: el objeto no referencia nada posterior a t.
        self._fields = {f: series.field(f)[: t + 1] for f in OHLCV_FIELDS}
        self._timestamp = series.timestamp[: t + 1]
        self._instrument = series.instrument
        self._t = t

    # -- identidad ------------------------------------------------------

    @property
    def t(self) -> int:
        """Indice de la barra actual dentro de la serie completa."""
        return self._t

    @property
    def timestamp(self) -> np.datetime64:
        return self._timestamp[-1]

    @property
    def instrument(self) -> InstrumentSpec:
        return self._instrument

    @property
    def symbol(self) -> str:
        return self._instrument.symbol

    def __len__(self) -> int:
        """Numero de barras visibles: ``t + 1``."""
        return self._t + 1

    def __repr__(self) -> str:
        return f"MarketView(symbol={self.symbol!r}, t={self._t}, close={self.close()})"

    # -- acceso escalar -------------------------------------------------

    def _at(self, field: str, lookback: int) -> float:
        if field not in self._fields:
            raise KeyError(f"campo desconocido: {field!r}")
        if lookback < 0:
            raise LookaheadError(
                f"lookback={lookback} pide una barra posterior a t={self._t}. "
                "El futuro no esta disponible en el instante de la decision."
            )
        if lookback > self._t:
            raise IndexError(
                f"lookback={lookback} excede la historia disponible "
                f"({len(self)} barras)"
            )
        return float(self._fields[field][-1 - lookback])

    def open(self, lookback: int = 0) -> float:
        return self._at("open", lookback)

    def high(self, lookback: int = 0) -> float:
        return self._at("high", lookback)

    def low(self, lookback: int = 0) -> float:
        return self._at("low", lookback)

    def close(self, lookback: int = 0) -> float:
        return self._at("close", lookback)

    def volume(self, lookback: int = 0) -> float:
        return self._at("volume", lookback)

    # -- acceso vectorial -----------------------------------------------

    def history(self, field: str, n: int) -> np.ndarray:
        """Ultimas ``n`` barras de ``field``, terminando en ``t``.

        Devuelve una copia de solo lectura: ni se puede mutar la serie ni se
        puede alcanzar el array completo a traves de ``ndarray.base``.
        """
        if field not in self._fields:
            raise KeyError(f"campo desconocido: {field!r}")
        if n <= 0:
            raise ValueError("n debe ser positivo")
        if n > len(self):
            raise IndexError(
                f"se pidieron {n} barras y solo hay {len(self)} "
                f"disponibles en t={self._t}"
            )
        out = np.array(self._fields[field][-n:], dtype=np.float64, copy=True)
        out.flags.writeable = False
        return out

    def log_returns(self, n: int) -> np.ndarray:
        """Retornos logaritmicos de cierre de las ultimas ``n`` barras.

        Los precios crudos no son estacionarios; esta es la forma preferida de
        consumir la serie desde una estrategia.
        """
        closes = self.history("close", n + 1)
        out = np.log(closes[1:] / closes[:-1])
        out.flags.writeable = False
        return out


@dataclass(frozen=True)
class AccountSnapshot:
    """Estado de la cuenta al cierre de la barra ``t``.

    Es una foto inmutable: la estrategia no puede modificar la contabilidad.
    """

    t: int
    cash: float
    position: float
    mark_price: float
    equity: float
    pending_qty: float

    @property
    def position_value(self) -> float:
        return self.position * self.mark_price

    def max_affordable_qty(self, price: float, *, safety: float = 0.999) -> float:
        """Cantidad bruta que el cash soporta a ``price``, con margen de seguridad.

        El margen existe porque el precio de ejecucion sera peor que ``price``
        (gap, spread, slippage) y la comision se cobra encima. Una orden que no
        deja holgura para eso se rechaza por cash insuficiente: el simulador no
        la redimensiona en silencio.
        """
        if price <= 0:
            raise ValueError("price debe ser positivo")
        return max(self.cash, 0.0) * safety / price
