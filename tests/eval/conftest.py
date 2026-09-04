from __future__ import annotations

import pytest

from data.instruments import CommissionSchema, InstrumentSpec, us_equity_spec


@pytest.fixture
def equity_spec() -> InstrumentSpec:
    """Accion entera con comision por accion y minimo: el caso realista."""
    return us_equity_spec(
        "TEST",
        commission=CommissionSchema(kind="per_share", value=0.005, minimum=1.0),
    )


@pytest.fixture
def sin_comision() -> InstrumentSpec:
    return us_equity_spec("TEST", commission=CommissionSchema(kind="fixed", value=0.0))
