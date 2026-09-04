"""Validacion de barras. Falla ruidosamente o no falla.

Ningun chequeo se degrada a warning y ninguno repara los datos. Si una barra es
incoherente, el experimento no puede correr: repararla en silencio produce
resultados que nadie puede auditar despues.
"""

from __future__ import annotations

import pandas as pd

from data.calendars import Calendar
from data.errors import (
    AdjustedPriceError,
    CalendarGapError,
    DataValidationError,
    SchemaError,
)
from data.instruments import InstrumentSpec
from data.schema import (
    ADJUSTED_COLUMN_HINTS,
    EVENT_FIELDS,
    OHLCV_FIELDS,
    PRICE_FIELDS,
    REQUIRED_COLUMNS,
)

_MAX_REPORTED = 10


def _sample(values) -> str:
    values = list(values)
    head = ", ".join(str(v) for v in values[:_MAX_REPORTED])
    if len(values) > _MAX_REPORTED:
        head += f", ... (+{len(values) - _MAX_REPORTED} mas)"
    return head


def check_no_adjusted_prices(frame: pd.DataFrame) -> None:
    """Rechaza datasets con precios ajustados retroactivamente.

    El ajuste retroactivo reescribe el pasado con informacion posterior: usar
    ``adj_close`` como precio de ejecucion es lookahead de manual. Los eventos
    corporativos deben venir en ``split_factor`` / ``cash_dividend``.
    """
    lowered = {str(c).strip().lower(): c for c in frame.columns}
    offenders = [lowered[h] for h in ADJUSTED_COLUMN_HINTS if h in lowered]
    if offenders:
        raise AdjustedPriceError(
            f"columnas de precio ajustado en la ruta de ejecucion: {offenders}. "
            "Cargar precios sin ajustar y pasar los eventos corporativos en "
            "split_factor / cash_dividend."
        )


def check_schema(frame: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise SchemaError(f"columnas requeridas ausentes: {missing}")
    for column in OHLCV_FIELDS + tuple(f for f in EVENT_FIELDS if f in frame.columns):
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise SchemaError(f"{column} debe ser numerica, es {frame[column].dtype}")


def check_timestamps(frame: pd.DataFrame) -> None:
    ts = frame["timestamp"]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        raise SchemaError(f"timestamp debe ser datetime, es {ts.dtype}")
    if ts.dt.tz is None:
        raise SchemaError(
            "timestamp es tz-naive; se exige tz-aware en UTC para evitar "
            "desalineaciones entre fuentes de precio y de noticias"
        )
    if str(ts.dt.tz) != "UTC":
        raise SchemaError(f"timestamp debe estar en UTC, esta en {ts.dt.tz}")
    if ts.isna().any():
        raise DataValidationError("hay timestamps nulos")
    duplicated = ts[ts.duplicated(keep=False)]
    if not duplicated.empty:
        raise DataValidationError(
            f"timestamps duplicados: {_sample(duplicated.unique())}"
        )
    if not ts.is_monotonic_increasing:
        offenders = ts[ts.diff() <= pd.Timedelta(0)]
        raise DataValidationError(
            f"timestamps no estrictamente crecientes en: {_sample(offenders)}"
        )


def check_symbol(frame: pd.DataFrame, instrument: InstrumentSpec) -> None:
    symbols = frame["symbol"].unique()
    if len(symbols) != 1:
        raise DataValidationError(
            f"se esperaba un unico simbolo, hay {len(symbols)}: {_sample(symbols)}"
        )
    if str(symbols[0]) != instrument.symbol:
        raise DataValidationError(
            f"simbolo del dataset ({symbols[0]!r}) != instrumento "
            f"({instrument.symbol!r})"
        )


def check_prices(frame: pd.DataFrame) -> None:
    for column in OHLCV_FIELDS:
        if frame[column].isna().any():
            bad = frame.loc[frame[column].isna(), "timestamp"]
            raise DataValidationError(f"NaN en {column} en: {_sample(bad)}")

    for column in PRICE_FIELDS:
        bad = frame.loc[frame[column] <= 0, "timestamp"]
        if not bad.empty:
            raise DataValidationError(f"{column} no positivo en: {_sample(bad)}")

    bad = frame.loc[frame["volume"] < 0, "timestamp"]
    if not bad.empty:
        raise DataValidationError(f"volumen negativo en: {_sample(bad)}")

    bad = frame.loc[frame["high"] < frame["low"], "timestamp"]
    if not bad.empty:
        raise DataValidationError(f"high < low en: {_sample(bad)}")

    for column in ("open", "close"):
        outside = frame[column].gt(frame["high"]) | frame[column].lt(frame["low"])
        bad = frame.loc[outside, "timestamp"]
        if not bad.empty:
            raise DataValidationError(
                f"{column} fuera del rango [low, high] en: {_sample(bad)}"
            )


def check_corporate_actions(frame: pd.DataFrame) -> None:
    if "split_factor" in frame.columns:
        bad = frame.loc[frame["split_factor"] <= 0, "timestamp"]
        if not bad.empty:
            raise DataValidationError(f"split_factor no positivo en: {_sample(bad)}")
    if "cash_dividend" in frame.columns:
        bad = frame.loc[frame["cash_dividend"] < 0, "timestamp"]
        if not bad.empty:
            raise DataValidationError(f"cash_dividend negativo en: {_sample(bad)}")


def check_calendar(frame: pd.DataFrame, calendar: Calendar, freq: str) -> None:
    """Compara las barras observadas contra las esperadas por el calendario."""
    actual = pd.DatetimeIndex(frame["timestamp"])
    expected = calendar.expected_index(actual[0], actual[-1], freq)
    missing = expected.difference(actual)
    extra = actual.difference(expected)
    problems = []
    if len(missing):
        problems.append(f"{len(missing)} barras faltantes: {_sample(missing)}")
    if len(extra):
        problems.append(
            f"{len(extra)} barras fuera del calendario '{calendar.name}': "
            f"{_sample(extra)}"
        )
    if problems:
        raise CalendarGapError(
            "el dataset no cuadra con el calendario declarado. " + " | ".join(problems)
        )


def check_tick_conformity(frame: pd.DataFrame, instrument: InstrumentSpec) -> None:
    """Verifica que los precios sean multiplos del tick del instrumento.

    ASSUMPTION: desactivado por defecto. Los datos agregados de proveedores
    (VWAP, barras consolidadas de varios venues) violan el tick de forma
    rutinaria sin que eso invalide la serie; se activa cuando la fuente
    pretende ser el book de un unico venue.
    """
    for column in PRICE_FIELDS:
        bad_mask = ~frame[column].map(instrument.is_price_on_tick)
        bad = frame.loc[bad_mask, "timestamp"]
        if not bad.empty:
            raise DataValidationError(
                f"{column} no es multiplo de tick_size={instrument.tick_size} "
                f"en: {_sample(bad)}"
            )


def validate_bars(
    frame: pd.DataFrame,
    *,
    instrument: InstrumentSpec,
    calendar: Calendar | None = None,
    freq: str | None = None,
    enforce_tick_size: bool = False,
) -> None:
    """Corre la bateria completa. Lanza al primer problema encontrado.

    ``calendar`` opcional solo para permitir validar fragmentos sinteticos; en
    la ruta de ingesta real se pasa siempre.
    """
    check_no_adjusted_prices(frame)
    check_schema(frame)
    check_timestamps(frame)
    check_symbol(frame, instrument)
    check_prices(frame)
    check_corporate_actions(frame)
    if enforce_tick_size:
        check_tick_conformity(frame, instrument)
    if calendar is not None:
        if freq is None:
            raise ValueError("validar contra un calendario exige declarar freq")
        check_calendar(frame, calendar, freq)
