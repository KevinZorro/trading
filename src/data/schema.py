"""Esquema de barras OHLCV.

Los precios almacenados son **sin ajustar**: son los que se pudieron ejecutar en
su momento. Los eventos corporativos viven en columnas propias y solo entran en
el calculo de retornos (ver :mod:`data.adjustments`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from data.errors import SchemaError
from data.instruments import InstrumentSpec

PRICE_FIELDS: tuple[str, ...] = ("open", "high", "low", "close")
OHLCV_FIELDS: tuple[str, ...] = (*PRICE_FIELDS, "volume")
EVENT_FIELDS: tuple[str, ...] = ("split_factor", "cash_dividend")
REQUIRED_COLUMNS: tuple[str, ...] = ("timestamp", *OHLCV_FIELDS, "symbol")
OPTIONAL_COLUMNS: tuple[str, ...] = EVENT_FIELDS

# Nombres que delatan precios ajustados retroactivamente. Si aparecen en la
# ruta de ejecucion, el loader falla en vez de ejecutar contra precios ficticios.
ADJUSTED_COLUMN_HINTS: tuple[str, ...] = (
    "adj_close",
    "adjclose",
    "adjusted_close",
    "adj close",
    "adj_open",
    "adj_high",
    "adj_low",
)


def _freeze(array: np.ndarray, name: str, dtype: Any) -> np.ndarray:
    """Copia a un array contiguo de solo lectura.

    La inmutabilidad no es estetica: el ``MarketView`` del simulador entrega
    slices de estos arrays a la estrategia, y un array escribible permitiria a
    una estrategia corromper la serie bajo los pies del motor.
    """
    out = np.ascontiguousarray(np.asarray(array, dtype=dtype))
    if out.ndim != 1:
        raise SchemaError(f"{name} debe ser unidimensional, tiene ndim={out.ndim}")
    out.flags.writeable = False
    return out


@dataclass(frozen=True)
class BarSeries:
    """Serie inmutable de barras de un unico simbolo.

    Todos los arrays tienen la misma longitud y ``writeable=False``.
    ``timestamp`` es ``datetime64[ns]`` en UTC y estrictamente creciente.
    """

    instrument: InstrumentSpec
    freq: str
    timestamp: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    split_factor: np.ndarray
    cash_dividend: np.ndarray
    source: str = "unknown"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "timestamp", _freeze(self.timestamp, "timestamp", "datetime64[ns]")
        )
        for field in OHLCV_FIELDS + EVENT_FIELDS:
            object.__setattr__(
                self, field, _freeze(getattr(self, field), field, np.float64)
            )
        n = len(self.timestamp)
        for field in OHLCV_FIELDS + EVENT_FIELDS:
            if len(getattr(self, field)) != n:
                raise SchemaError(
                    f"longitud inconsistente en {field}: "
                    f"{len(getattr(self, field))} != {n}"
                )
        if n == 0:
            raise SchemaError("BarSeries vacia")

    # -- acceso ---------------------------------------------------------

    @property
    def symbol(self) -> str:
        return self.instrument.symbol

    def __len__(self) -> int:
        return len(self.timestamp)

    def field(self, name: str) -> np.ndarray:
        """Devuelve un campo por nombre; el array sigue siendo de solo lectura."""
        if name not in OHLCV_FIELDS + EVENT_FIELDS + ("timestamp",):
            raise SchemaError(f"campo desconocido: {name!r}")
        return getattr(self, name)

    def slice(self, start: int, stop: int) -> BarSeries:
        """Sub-serie ``[start, stop)``. Usado por el walk-forward."""
        n = len(self)
        if not (0 <= start < stop <= n):
            raise SchemaError(f"slice invalido ({start}, {stop}) sobre n={n}")
        kwargs = {f: getattr(self, f)[start:stop] for f in OHLCV_FIELDS + EVENT_FIELDS}
        return replace(self, timestamp=self.timestamp[start:stop], **kwargs)

    def has_corporate_actions(self) -> bool:
        return bool(
            np.any(self.split_factor != 1.0) or np.any(self.cash_dividend != 0.0)
        )

    def meta(self) -> dict[str, Any]:
        """Metadatos serializables para guardar junto a cada resultado."""
        return {
            "symbol": self.symbol,
            "venue": self.instrument.venue,
            "asset_class": self.instrument.asset_class,
            "freq": self.freq,
            "n_bars": len(self),
            "start": str(self.timestamp[0]),
            "end": str(self.timestamp[-1]),
            "source": self.source,
            "has_corporate_actions": self.has_corporate_actions(),
        }
