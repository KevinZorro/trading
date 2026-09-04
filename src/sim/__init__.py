"""Motor de simulacion: ejecucion, costos y contabilidad.

Tres invariantes que el diseno hace estructurales, no cuestion de disciplina:

1. **Sin lookahead.** El motor conduce el bucle y entrega un
   :class:`~sim.view.MarketView` limitado a ``[0..t]``. No hay ``step()``
   publico ni acceso a la serie completa desde la estrategia.
2. **Ejecucion diferida.** Decision con el cierre de ``t``, ejecucion al open
   de ``t+1``.
3. **Gap separado de costos.** El log distingue el desplazamiento de valuacion
   entre ``close[t]`` y ``open[t+1]`` de lo que cuesta transaccionar.
"""

from sim.costs import (
    CommissionModel,
    CorwinSchultzSpread,
    FixedBpsSpread,
    LinearSlippage,
    NoSlippage,
    SchemaCommission,
    SlippageModel,
    SpreadContext,
    SpreadModel,
    SqrtSlippage,
    ZeroCommission,
    ZeroSpread,
    commission_from_instrument,
)
from sim.engine import (
    LONG_ONLY_ACTION_RANGE,
    SimConfig,
    SimResult,
    Simulator,
    Strategy,
)
from sim.orders import Fill, MarketOrder, OrderStatus, RejectReason
from sim.portfolio import ACCOUNTING_TOL, AccountingError, Ledger
from sim.view import AccountSnapshot, MarketView

__all__ = [
    "ACCOUNTING_TOL",
    "LONG_ONLY_ACTION_RANGE",
    "AccountSnapshot",
    "AccountingError",
    "CommissionModel",
    "CorwinSchultzSpread",
    "Fill",
    "FixedBpsSpread",
    "Ledger",
    "LinearSlippage",
    "MarketOrder",
    "MarketView",
    "NoSlippage",
    "OrderStatus",
    "RejectReason",
    "SchemaCommission",
    "SimConfig",
    "SimResult",
    "Simulator",
    "SlippageModel",
    "SpreadContext",
    "SpreadModel",
    "SqrtSlippage",
    "Strategy",
    "ZeroCommission",
    "ZeroSpread",
    "commission_from_instrument",
]
