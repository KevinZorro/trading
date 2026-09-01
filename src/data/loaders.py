"""Carga de barras desde CSV/parquet hacia :class:`BarSeries`.

La unica puerta de entrada de datos externos al proyecto. Valida antes de
construir: no existe una ``BarSeries`` que no haya pasado la bateria completa.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.calendars import Calendar
from data.errors import SchemaError
from data.instruments import InstrumentSpec
from data.schema import EVENT_FIELDS, OHLCV_FIELDS, BarSeries
from data.validation import validate_bars

_EVENT_DEFAULTS = {"split_factor": 1.0, "cash_dividend": 0.0}


def _read_any(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise SchemaError(f"extension no soportada: {suffix!r} (usar .csv o .parquet)")


def bars_from_frame(
    frame: pd.DataFrame,
    *,
    instrument: InstrumentSpec,
    freq: str,
    calendar: Calendar | None = None,
    column_map: dict[str, str] | None = None,
    enforce_tick_size: bool = False,
    source: str = "frame",
) -> BarSeries:
    """Valida un DataFrame y lo congela en una ``BarSeries``."""
    frame = frame.rename(columns=column_map) if column_map else frame.copy()

    if "timestamp" in frame.columns:
        ts = pd.to_datetime(frame["timestamp"], utc=False, errors="raise")
        # No se localiza a UTC en silencio: un timestamp naive es ambiguo y
        # asumir una zona horaria aqui desalinea precios y noticias mas tarde.
        frame["timestamp"] = ts

    for field, default in _EVENT_DEFAULTS.items():
        if field not in frame.columns:
            frame[field] = default

    frame = frame.reset_index(drop=True)
    validate_bars(
        frame,
        instrument=instrument,
        calendar=calendar,
        freq=freq if calendar is not None else None,
        enforce_tick_size=enforce_tick_size,
    )

    return BarSeries(
        instrument=instrument,
        freq=freq,
        timestamp=frame["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[ns]"),
        source=source,
        **{f: frame[f].to_numpy(dtype="float64") for f in OHLCV_FIELDS + EVENT_FIELDS},
    )


def load_bars(
    path: str | Path,
    *,
    instrument: InstrumentSpec,
    freq: str,
    calendar: Calendar | None = None,
    column_map: dict[str, str] | None = None,
    enforce_tick_size: bool = False,
) -> BarSeries:
    """Carga barras de un fichero. Los precios deben venir **sin ajustar**."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = _read_any(path)
    return bars_from_frame(
        frame,
        instrument=instrument,
        freq=freq,
        calendar=calendar,
        column_map=column_map,
        enforce_tick_size=enforce_tick_size,
        source=str(path),
    )


def bars_to_frame(series: BarSeries) -> pd.DataFrame:
    """Vuelca una ``BarSeries`` a DataFrame (fixtures, inspeccion, persistencia)."""
    data = {"timestamp": pd.DatetimeIndex(series.timestamp).tz_localize("UTC")}
    for field in OHLCV_FIELDS + EVENT_FIELDS:
        data[field] = series.field(field)
    data["symbol"] = series.symbol
    return pd.DataFrame(data)
