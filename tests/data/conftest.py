from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.instruments import InstrumentSpec, us_equity_spec

SESSION_TIME = "21:00:00"


@pytest.fixture
def equity() -> InstrumentSpec:
    return us_equity_spec("TEST")


def make_frame(
    n: int = 10,
    *,
    symbol: str = "TEST",
    start: str = f"2020-01-01T{SESSION_TIME}Z",
    freq: str = "1D",
    business_days: bool = True,
) -> pd.DataFrame:
    """Barras validas y aburridas: precio 100, rango de 1, volumen constante."""
    if business_days:
        index = pd.bdate_range(start=pd.Timestamp(start), periods=n, tz="UTC")
    else:
        index = pd.date_range(start=pd.Timestamp(start), periods=n, freq=freq, tz="UTC")
    close = 100.0 + np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": index,
            "open": close - 0.25,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
            "symbol": symbol,
        }
    )
