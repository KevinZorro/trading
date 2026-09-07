"""Metricas y reportes sobre corridas del simulador.

Dos reglas del modulo que no son negociables:

1. **Se reportan siempre las dos series de equity**, la marcada a close y la de
   liquidacion, con la brecha entre ellas como metrica explicita. Esa brecha
   mide la friccion no realizada: cuanto del equity marcado desaparece al
   intentar convertirlo en efectivo.
2. **El gap de ejecucion nunca se suma a los costos de transaccion.** Es
   valuacion, lleva signo y puede ser favorable; agregarlo destruye la lectura
   por regimen que se midio en la Etapa 1.
"""

from eval.distribution import (
    MIN_SEEDS,
    DeflatedSharpe,
    DistributionError,
    SeedDistribution,
    deflated_sharpe,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    summarize,
)
from eval.metrics import (
    DISPERSION_NULA_REL,
    Drawdown,
    MetricError,
    annual_volatility,
    cagr,
    calmar,
    excess_returns,
    first_ruin_index,
    max_drawdown,
    periodic_rate,
    sharpe,
    simple_returns,
    sortino,
    total_return,
    truncate_at_ruin,
    years_elapsed,
)
from eval.report import (
    CostReport,
    EquityMetrics,
    FrictionGap,
    RunReport,
    TurnoverStats,
    evaluate_run,
    evaluate_series,
    friction_gap,
)
from eval.trades import (
    POSITION_TOL,
    OpenPosition,
    RoundTrip,
    RoundTripLog,
    TradeStats,
    round_trips,
    trade_stats,
)
from eval.walkforward import WalkForwardError, Window, coverage, rolling_windows

__all__ = [
    "DISPERSION_NULA_REL",
    "MIN_SEEDS",
    "POSITION_TOL",
    "CostReport",
    "DeflatedSharpe",
    "DistributionError",
    "Drawdown",
    "EquityMetrics",
    "FrictionGap",
    "MetricError",
    "OpenPosition",
    "RoundTrip",
    "RoundTripLog",
    "RunReport",
    "SeedDistribution",
    "TradeStats",
    "TurnoverStats",
    "WalkForwardError",
    "Window",
    "annual_volatility",
    "cagr",
    "calmar",
    "coverage",
    "deflated_sharpe",
    "evaluate_run",
    "evaluate_series",
    "excess_returns",
    "expected_max_sharpe",
    "first_ruin_index",
    "friction_gap",
    "max_drawdown",
    "periodic_rate",
    "probabilistic_sharpe_ratio",
    "rolling_windows",
    "round_trips",
    "sharpe",
    "simple_returns",
    "sortino",
    "summarize",
    "total_return",
    "trade_stats",
    "truncate_at_ruin",
    "years_elapsed",
]
