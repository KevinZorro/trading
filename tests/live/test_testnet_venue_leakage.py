"""Anti-leakage del venue de testnet: tiempo solo por ``Clock``."""

from __future__ import annotations

from pathlib import Path

import pytest

from live.testnet_venue import BinanceTestnetVenue
from sim.orders import MarketOrder
from tests.live.test_testnet_venue import AHORA, POST, Servidor, grabacion, ok, venue

pytestmark = pytest.mark.leakage


def test_la_decision_no_se_fecha_despues_del_fill(tmp_path: Path) -> None:
    v: BinanceTestnetVenue = venue(
        tmp_path, Servidor(**{POST: [ok(grabacion("order_buy.json"))]})
    )
    v.submit_order(MarketOrder(0.001, "", "rec7a-buy-1792379461"))
    (f,) = v.get_fills()
    assert f.timestamp_decision == AHORA
    assert f.timestamp_decision <= f.timestamp_fill
    assert f.t_decision == f.t_fill == 1
