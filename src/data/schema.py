"""Esquema de barras OHLCV.

Los precios almacenados son **sin ajustar**: son los que se pudieron ejecutar en
su momento. Los eventos corporativos viven en columnas propias y solo entran en
el calculo de retornos (ver :mod:`data.adjustments`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import numpy.typing as npt

from data.errors import SchemaError
from data.instruments import InstrumentSpec

# Alias de los arrays de la serie. Todas las columnas numericas son 1-D,
# float64 y de solo lectura; nombrarlas evita repetir la anotacion completa
# en cada firma y hace que mypy strict verifique de verdad los bordes.
FloatArray = npt.NDArray[np.float64]
TimeArray = npt.NDArray[np.datetime64]

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


@dataclass(frozen=True)
class ExogColumn:
    """Una columna exogena (no es precio) que el entorno agrega a la observacion.

    ``scaled`` dice si lleva z-score ajustado en train (``False`` para indicadores
    0/1). Una columna exogena puede venir enmascarada (``NaN``): el escalador la
    ajusta sin los ``NaN`` y la deja en 0 despues de escalar.
    """

    name: str
    scaled: bool = True


def _freeze(array: npt.ArrayLike, name: str, dtype: Any) -> npt.NDArray[Any]:
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
    timestamp: TimeArray
    open: FloatArray
    high: FloatArray
    low: FloatArray
    close: FloatArray
    volume: FloatArray
    split_factor: FloatArray
    cash_dividend: FloatArray
    source: str = "unknown"
    # Columnas exogenas alineadas con ``timestamp``: la fila ``i`` solo contiene lo
    # conocido al cierre de la barra ``i`` (lo garantiza quien la construye, p. ej.
    # ``features.news.with_news``). ``NaN`` = enmascarado. Sin ellas, la serie y su
    # serializacion son las de siempre.
    exog: FloatArray | None = None
    exog_names: tuple[str, ...] = ()

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
        self._validate_exog(n)

    def _validate_exog(self, n: int) -> None:
        if self.exog is None:
            if self.exog_names:
                raise SchemaError("exog_names sin matriz exog")
            return
        matriz = np.ascontiguousarray(np.asarray(self.exog, dtype=np.float64))
        if matriz.ndim != 2 or matriz.shape != (n, len(self.exog_names)):
            raise SchemaError(
                f"exog debe ser ({n}, {len(self.exog_names)}) y es {matriz.shape}"
            )
        if len(set(self.exog_names)) != len(self.exog_names):
            raise SchemaError("exog_names tiene nombres repetidos")
        if np.isinf(matriz).any():
            raise SchemaError("exog tiene valores infinitos (solo NaN = enmascarado)")
        matriz.flags.writeable = False
        object.__setattr__(self, "exog", matriz)
        object.__setattr__(self, "exog_names", tuple(self.exog_names))

    # -- acceso ---------------------------------------------------------

    @property
    def symbol(self) -> str:
        return self.instrument.symbol

    def __len__(self) -> int:
        return len(self.timestamp)

    def field(self, name: str) -> npt.NDArray[Any]:
        """Devuelve un campo por nombre; el array sigue siendo de solo lectura."""
        if name not in OHLCV_FIELDS + EVENT_FIELDS + ("timestamp",):
            raise SchemaError(f"campo desconocido: {name!r}")
        column: npt.NDArray[Any] = getattr(self, name)
        return column

    def slice(self, start: int, stop: int) -> BarSeries:
        """Sub-serie ``[start, stop)``. Usado por el walk-forward."""
        n = len(self)
        if not (0 <= start < stop <= n):
            raise SchemaError(f"slice invalido ({start}, {stop}) sobre n={n}")
        kwargs = {f: getattr(self, f)[start:stop] for f in OHLCV_FIELDS + EVENT_FIELDS}
        exog = None if self.exog is None else self.exog[start:stop]
        return replace(self, timestamp=self.timestamp[start:stop], exog=exog, **kwargs)

    def has_corporate_actions(self) -> bool:
        return bool(
            np.any(self.split_factor != 1.0) or np.any(self.cash_dividend != 0.0)
        )

    def meta(self) -> dict[str, Any]:
        """Metadatos serializables para guardar junto a cada resultado."""
        extra: dict[str, Any] = (
            {"exog_names": list(self.exog_names)} if self.exog_names else {}
        )
        return {
            **extra,
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
