from __future__ import annotations

import numpy as np
import pytest

from data.instruments import CommissionSchema, InstrumentSpec, us_equity_spec
from data.schema import BarSeries
from envs.observation import ObservationBuilder, fit_scaler_on_train
from features.scaler import FeatureScaler
from sim.costs import FixedBpsSpread, LinearSlippage
from sim.engine import SimConfig

from ..sim.conftest import make_series


@pytest.fixture
def spec() -> InstrumentSpec:
    """Fraccional sin minimos: aisla el entorno del redondeo del instrumento."""
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
def spec_con_minimos() -> InstrumentSpec:
    """Accion entera con comision minima: el caso donde el rechazo importa."""
    return us_equity_spec(
        "TEST",
        commission=CommissionSchema(kind="per_share", value=0.005, minimum=1.0),
    )


def serie_deterministica(
    instrumento: InstrumentSpec, n: int = 120, seed: int = 3
) -> BarSeries:
    """Camino reproducible, con gap entre cierre y apertura siguiente."""
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n)))
    opens = closes * (1.0 + rng.normal(0.0, 0.002, n))
    return make_series(instrumento, closes, opens=opens, volume=200_000.0)


@pytest.fixture
def serie(spec: InstrumentSpec) -> BarSeries:
    return serie_deterministica(spec)


@pytest.fixture
def builder() -> ObservationBuilder:
    return ObservationBuilder()


@pytest.fixture
def scaler(serie: BarSeries, builder: ObservationBuilder) -> FeatureScaler:
    return fit_scaler_on_train(serie.slice(0, 80), builder)


@pytest.fixture
def sim_config() -> SimConfig:
    return SimConfig(
        initial_cash=100_000.0,
        spread=FixedBpsSpread(bps=8.0),
        slippage=LinearSlippage(k=0.4),
    )
