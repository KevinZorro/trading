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
from data.fixtures import (
    DEFAULT_N_BARS,
    LEVELS,
    SNR_SWEEP_BETAS,
    Ceilings,
    Fixture,
    FixtureCosts,
    FixtureError,
    HysteresisPolicy,
    SignalSpec,
    calibrate_costs,
    clairvoyant_states,
    evaluate_states,
    fixture_instrument,
    generate_signal_bars,
    level_0_deterministic,
    level_1_noisy,
    level_2_costly,
    level_3_regime_flip,
    level_4_control,
    myopic_states,
    solve_optimal_policy,
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
    "DEFAULT_N_BARS",
    "LEVELS",
    "OHLCV_FIELDS",
    "PRICE_FIELDS",
    "REQUIRED_COLUMNS",
    "SNR_SWEEP_BETAS",
    "AdjustedPriceError",
    "AlwaysOpen",
    "BarSeries",
    "Calendar",
    "CalendarGapError",
    "Ceilings",
    "CommissionSchema",
    "DataError",
    "DataValidationError",
    "ExplicitCalendar",
    "Fixture",
    "FixtureCosts",
    "FixtureError",
    "HysteresisPolicy",
    "InstrumentSpec",
    "LookaheadError",
    "RejectReason",
    "SchemaError",
    "SignalSpec",
    "WeekdayCalendar",
    "backward_adjusted_close",
    "bars_from_frame",
    "binance_spot_spec",
    "calibrate_costs",
    "clairvoyant_states",
    "evaluate_states",
    "fixture_instrument",
    "generate_gbm_sv",
    "generate_signal_bars",
    "level_0_deterministic",
    "level_1_noisy",
    "level_2_costly",
    "level_3_regime_flip",
    "level_4_control",
    "load_bars",
    "myopic_states",
    "solve_optimal_policy",
    "total_return_index",
    "total_return_log_returns",
    "us_equity_spec",
    "validate_bars",
]
