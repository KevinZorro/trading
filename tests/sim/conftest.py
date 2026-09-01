from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.instruments import CommissionSchema, InstrumentSpec, us_equity_spec
from data.loaders import bars_from_frame
from data.schema import BarSeries
from sim.engine import SimConfig


@pytest.fixture
def equity() -> InstrumentSpec:
    """Accion entera, sin comision: la base para tests de invariantes."""
    return us_equity_spec("TEST", commission=CommissionSchema(kind="fixed", value=0.0))


@pytest.fixture
def fractional() -> InstrumentSpec:
    """Instrumento fraccional sin minimos: aisla la contabilidad del redondeo."""
    return InstrumentSpec(
        symbol="FRAC",
        venue="TEST",
        tick_size=0.01,
        lot_size=1e-8,
        allow_fractional=True,
        qty_precision=8,
        min_order_qty=0.0,
        min_notional=0.0,
        commission_schema=CommissionSchema(kind="fixed", value=0.0),
        asset_class="crypto",
    )


@pytest.fixture
def frictionless(equity: InstrumentSpec) -> SimConfig:
    return SimConfig(initial_cash=100_000.0)


def make_series(
    instrument: InstrumentSpec,
    closes: list[float] | np.ndarray,
    *,
    opens: list[float] | np.ndarray | None = None,
    volume: float | np.ndarray = 1_000_000.0,
    freq: str = "1D",
) -> BarSeries:
    """Serie deterministica y verificable a mano.

    Por defecto ``open == close`` de la misma barra, lo que anula el gap y
    permite testear los costos de transaccion aislados. Los tests de gap pasan
    ``opens`` explicitamente.
    """
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    opens_arr = closes.copy() if opens is None else np.asarray(opens, dtype=float)
    volume_arr = (
        np.full(n, float(volume))
        if np.isscalar(volume)
        else np.asarray(volume, dtype=float)
    )
    highs = np.maximum(opens_arr, closes) * 1.001
    lows = np.minimum(opens_arr, closes) * 0.999
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01T21:00:00Z", periods=n, freq=freq),
            "open": opens_arr,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volume_arr,
            "symbol": instrument.symbol,
        }
    )
    return bars_from_frame(frame, instrument=instrument, freq=freq)
