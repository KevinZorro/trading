from __future__ import annotations

import pytest

from data.instruments import CommissionSchema, InstrumentSpec


@pytest.fixture
def fractional() -> InstrumentSpec:
    """Instrumento fraccional sin minimos: aisla el riesgo del redondeo.

    Duplica a proposito la fixture de ``tests/sim``: los tests de riesgo no
    deben romperse si aquella cambia por un motivo del simulador.
    """
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
