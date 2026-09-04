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

from sim.clock import Clock, SimulatedClock, SystemClock
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
from sim.gate import GateDecision, GateRejection, OrderGate
from sim.ids import OrderIdGenerator, PrefixedSequentialIds, SequentialIds
from sim.orders import Fill, MarketOrder, OrderStatus, RejectReason
from sim.portfolio import ACCOUNTING_TOL, AccountingError, Ledger
from sim.venue import CancelAck, ExecutionVenue, OrderAck, VenueState
from sim.view import AccountSnapshot, MarketView

__all__ = [
    "ACCOUNTING_TOL",
    "LONG_ONLY_ACTION_RANGE",
    "AccountSnapshot",
    "AccountingError",
    "CancelAck",
    "Clock",
    "CommissionModel",
    "CorwinSchultzSpread",
    "ExecutionVenue",
    "Fill",
    "FixedBpsSpread",
    "GateDecision",
    "GateRejection",
    "Ledger",
    "LinearSlippage",
    "MarketOrder",
    "MarketView",
    "NoSlippage",
    "OrderAck",
    "OrderGate",
    "OrderIdGenerator",
    "OrderStatus",
    "PrefixedSequentialIds",
    "RejectReason",
    "SchemaCommission",
    "SequentialIds",
    "SimConfig",
    "SimResult",
    "SimulatedClock",
    "Simulator",
    "SlippageModel",
    "SpreadContext",
    "SpreadModel",
    "SqrtSlippage",
    "Strategy",
    "SystemClock",
    "VenueState",
    "ZeroCommission",
    "ZeroSpread",
    "commission_from_instrument",
]
