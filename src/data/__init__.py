"""Ingesta, validacion y generacion de datos de barras.

Contrato del modulo:

- Los precios OHLC almacenados son **sin ajustar**: son los que realmente se
  pudieron ejecutar. Los eventos corporativos viven en columnas separadas
  (``split_factor``, ``cash_dividend``) y solo se usan para calcular retornos.
- Toda serie va acompanada de un :class:`InstrumentSpec`, porque las
  restricciones de ejecucion (lote, nocional minimo, fraccionalidad) son
  propiedad del instrumento, no configuracion global del simulador.
"""

from data.adjustments import (
    backward_adjusted_close,
    total_return_index,
    total_return_log_returns,
)
from data.calendars import AlwaysOpen, Calendar, ExplicitCalendar, WeekdayCalendar
from data.errors import (
    AdjustedPriceError,
    CalendarGapError,
    DataError,
    DataValidationError,
    LookaheadError,
    SchemaError,
)
from data.instruments import (
    CommissionSchema,
    InstrumentSpec,
    RejectReason,
    binance_spot_spec,
    us_equity_spec,
)
from data.loaders import bars_from_frame, load_bars
from data.schema import OHLCV_FIELDS, PRICE_FIELDS, REQUIRED_COLUMNS, BarSeries
from data.synthetic import generate_gbm_sv
from data.validation import validate_bars

__all__ = [
    "OHLCV_FIELDS",
    "PRICE_FIELDS",
    "REQUIRED_COLUMNS",
    "AdjustedPriceError",
    "AlwaysOpen",
    "BarSeries",
    "Calendar",
    "CalendarGapError",
    "CommissionSchema",
    "DataError",
    "DataValidationError",
    "ExplicitCalendar",
    "InstrumentSpec",
    "LookaheadError",
    "RejectReason",
    "SchemaError",
    "WeekdayCalendar",
    "backward_adjusted_close",
    "bars_from_frame",
    "binance_spot_spec",
    "generate_gbm_sv",
    "load_bars",
    "total_return_index",
    "total_return_log_returns",
    "us_equity_spec",
    "validate_bars",
]
