"""Reporte de una corrida sobre las dos series de equity.

El proyecto mantiene dos series en paralelo y las dos se reportan siempre:

- ``equity_mark``: cash mas posicion valuada al close. Metrica primaria.
- ``equity_liquidation``: cash mas el producto neto de cerrar la posicion en esa
  barra, con spread, slippage y comision de salida.

La **brecha entre ambas** es una metrica de primer nivel, no una curiosidad:
mide la friccion no realizada, es decir cuanto del equity marcado se evapora al
intentar retirarlo. Una estrategia con una posicion enorme en un instrumento
iliquido tiene un ``equity_mark`` que no puede cobrar.

Sobre el orden de los drawdowns entre las dos series: ``equity_liquidation <=
equity_mark`` en todas las barras (con ``allow_short=False``), pero eso **no**
implica que su drawdown sea mayor. Si la posicion es grande en el pico y chica o
nula en el valle, la friccion deprime el pico mas que el valle y el drawdown de
la serie de liquidacion resulta *menor*. Es contraintuitivo y hay un test que lo
fija.

El ``gap`` de ejecucion nunca se suma a los costos. Ver ``CostReport``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from data.schema import FloatArray
from eval.metrics import (
    Drawdown,
    MetricError,
    annual_volatility,
    cagr,
    calmar,
    max_drawdown,
    sharpe,
    sortino,
    total_return,
    truncate_at_ruin,
    years_elapsed,
)
from eval.trades import RoundTripLog, TradeStats, round_trips, trade_stats
from sim.engine import SimResult

# Minimo de observaciones para estimar una desviacion estandar. Con dos puntos
# hay un solo retorno y la varianza muestral no existe.
_MIN_OBS_DISPERSION = 3


@dataclass(frozen=True)
class EquityMetrics:
    """Metricas de una unica serie de equity.

    ``annualization_reliable`` es falso cuando el periodo cubre menos de un ano.
    El CAGR y el Sharpe se calculan igual -en walk-forward todas las ventanas
    son cortas y hace falta que todas se anualicen del mismo modo para poder
    compararlas- pero la extrapolacion queda marcada en vez de disimulada.
    """

    label: str
    n_bars: int
    years: float
    annualization_reliable: bool
    initial_equity: float
    final_equity: float
    total_return: float
    cagr: float
    annual_volatility: float
    sharpe: float | None
    sortino: float | None
    drawdown: Drawdown
    calmar: float
    ruined_at: int | None


def evaluate_series(
    equity: FloatArray,
    *,
    label: str,
    bars_per_year: float,
    risk_free: float = 0.0,
) -> EquityMetrics:
    """Metricas de una serie de equity cualquiera, venga o no del simulador."""
    serie, ruina = truncate_at_ruin(equity)
    if len(serie) < 2:
        raise MetricError(
            f"la serie {label!r} queda con {len(serie)} observaciones utiles "
            "tras cortar en la ruina; no hay retorno que medir"
        )

    hay_dispersion = len(serie) >= _MIN_OBS_DISPERSION
    years = years_elapsed(len(serie), bars_per_year)
    return EquityMetrics(
        label=label,
        n_bars=len(serie),
        years=years,
        annualization_reliable=years >= 1.0,
        initial_equity=float(serie[0]),
        final_equity=float(serie[-1]),
        total_return=total_return(serie),
        cagr=cagr(serie, bars_per_year),
        annual_volatility=annual_volatility(serie, bars_per_year),
        sharpe=sharpe(serie, bars_per_year, risk_free) if hay_dispersion else None,
        sortino=sortino(serie, bars_per_year, risk_free) if hay_dispersion else None,
        drawdown=max_drawdown(serie),
        calmar=calmar(serie, bars_per_year),
        ruined_at=ruina,
    )


@dataclass(frozen=True)
class FrictionGap:
    """Brecha entre el equity marcado y el de liquidacion.

    Mide la friccion no realizada: lo que cuesta convertir el equity marcado en
    efectivo. ``min_gap`` se expone a proposito: con ``allow_short=False`` no
    puede ser negativo, y si lo es hay un bug de signo en el motor. Se reporta
    en vez de lanzar para que el numero se vea.
    """

    mean_gap: float
    max_gap: float
    min_gap: float
    final_gap: float
    max_gap_pct: float
    final_gap_pct: float


def friction_gap(
    equity_mark: FloatArray, equity_liquidation: FloatArray
) -> FrictionGap:
    marca = np.asarray(equity_mark, dtype=np.float64)
    liquida = np.asarray(equity_liquidation, dtype=np.float64)
    if marca.shape != liquida.shape:
        raise MetricError(
            f"las series tienen longitudes distintas: {marca.shape} y {liquida.shape}"
        )
    brecha = marca - liquida
    con_marca_positiva = np.where(marca > 0, marca, np.nan)
    porcentual = brecha / con_marca_positiva
    finito = porcentual[np.isfinite(porcentual)]
    return FrictionGap(
        mean_gap=float(np.mean(brecha)),
        max_gap=float(np.max(brecha)),
        min_gap=float(np.min(brecha)),
        final_gap=float(brecha[-1]),
        max_gap_pct=float(np.max(finito)) if len(finito) else 0.0,
        final_gap_pct=float(porcentual[-1]) if np.isfinite(porcentual[-1]) else 0.0,
    )


@dataclass(frozen=True)
class CostReport:
    """Desglose de la friccion de ejecucion, con el gap en linea propia.

    **El gap nunca se agrega a los costos.** Es un efecto de valuacion -la senal
    es del close de ``t`` y la ejecucion del open de ``t+1``-, lleva signo y
    puede ser favorable. En la Etapa 1 se midio que en el tier de riesgo alto el
    gap fue favorable y compenso un cuarto de los costos, invirtiendo el orden
    aparente de los regimenes. Sumarlo a ``total_costs_paid`` habria borrado ese
    hallazgo y habria hecho indistinguible "el agente opera demasiado" de "el
    agente opera en el peor momento del ciclo overnight".

    ``implementation_shortfall`` si es la suma de los cuatro terminos, pero se
    llama por su nombre y viene acompanado de sus componentes.
    """

    spread: float
    slippage: float
    commission: float
    total_costs_paid: float
    gap: float
    implementation_shortfall: float
    interest_earned: float
    n_fills: int
    n_rejected: int
    costs_pct_of_initial: float
    costs_bps_of_turnover: float
    gap_pct_of_initial: float


@dataclass(frozen=True)
class TurnoverStats:
    """Cuanto se opero, en tres normalizaciones distintas.

    ``notional`` es la suma del nocional de los fills ejecutados. ``ratio`` lo
    normaliza por el equity medio: cuantas veces se dio vuelta la cuenta.
    ``annualized`` lleva ese ratio a base anual, que es como se compara entre
    corridas de largo distinto.
    """

    notional: float
    ratio: float
    annualized: float
    n_fills: int
    n_rejected: int


@dataclass(frozen=True)
class RunReport:
    """Reporte completo de una corrida."""

    strategy_name: str
    mark: EquityMetrics
    liquidation: EquityMetrics
    friction: FrictionGap
    costs: CostReport
    turnover: TurnoverStats
    trades: TradeStats
    risk_free: float
    bars_per_year: float
    config: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        """Forma serializable, para guardar junto a la config de la corrida."""
        return {
            "strategy_name": self.strategy_name,
            "risk_free": self.risk_free,
            "bars_per_year": self.bars_per_year,
            "mark": _metrics_dict(self.mark),
            "liquidation": _metrics_dict(self.liquidation),
            "friction": vars(self.friction).copy(),
            "costs": vars(self.costs).copy(),
            "turnover": vars(self.turnover).copy(),
            "trades": _trades_dict(self.trades),
            "config": dict(self.config),
        }

    def render(self) -> str:
        """Reporte de texto. Las dos series lado a lado y el gap aparte."""
        return _render(self)


def _metrics_dict(m: EquityMetrics) -> dict[str, object]:
    salida = vars(m).copy()
    salida["drawdown"] = vars(m.drawdown).copy()
    return salida


def _trades_dict(t: TradeStats) -> dict[str, object]:
    salida = vars(t).copy()
    abierta = t.open_position
    salida["open_position"] = (
        None
        if abierta is None
        else {
            "t_open": abierta.t_open,
            "timestamp_open": str(abierta.timestamp_open),
            "qty": abierta.qty,
            "cost_basis": abierta.cost_basis,
            "commission": abierta.commission,
            "n_fills": abierta.n_fills,
        }
    )
    return salida


def evaluate_run(result: SimResult, *, risk_free: float | None = None) -> RunReport:
    """Reporte completo a partir del resultado del simulador.

    ``risk_free`` toma por defecto el ``cash_rate`` de la corrida. El motor
    devenga interes sobre el cash ocioso, asi que ese rendimiento ya esta dentro
    del equity; descontar cero le regalaria alfa a una estrategia que pasa la
    mitad del tiempo fuera del mercado cobrando esa misma tasa.
    """
    bars_per_year = float(str(result.config["bars_per_year"]))
    tasa = float(str(result.config["cash_rate"])) if risk_free is None else risk_free
    inicial = float(str(result.config["initial_cash"]))

    marca = evaluate_series(
        result.equity,
        label="equity_mark",
        bars_per_year=bars_per_year,
        risk_free=tasa,
    )
    liquida = evaluate_series(
        result.equity_liquidation,
        label="equity_liquidation",
        bars_per_year=bars_per_year,
        risk_free=tasa,
    )

    desglose = result.cost_breakdown()
    nocional = float(desglose["turnover_notional"])
    costos_pagados = float(desglose["total_costs_paid"])
    equity_medio = float(np.mean(np.asarray(result.equity, dtype=np.float64)))

    costos = CostReport(
        spread=float(desglose["spread"]),
        slippage=float(desglose["slippage"]),
        commission=float(desglose["commission"]),
        total_costs_paid=costos_pagados,
        gap=float(desglose["gap"]),
        implementation_shortfall=float(desglose["implementation_shortfall"]),
        interest_earned=float(desglose["interest_earned"]),
        n_fills=int(desglose["n_fills"]),
        n_rejected=int(desglose["n_rejected"]),
        costs_pct_of_initial=costos_pagados / inicial if inicial > 0 else 0.0,
        costs_bps_of_turnover=(
            costos_pagados / nocional * 1e4 if nocional > 0 else 0.0
        ),
        gap_pct_of_initial=(float(desglose["gap"]) / inicial if inicial > 0 else 0.0),
    )

    ratio = nocional / equity_medio if equity_medio > 0 else 0.0
    turnover = TurnoverStats(
        notional=nocional,
        ratio=ratio,
        annualized=ratio / marca.years if marca.years > 0 else 0.0,
        n_fills=int(desglose["n_fills"]),
        n_rejected=int(desglose["n_rejected"]),
    )

    log: RoundTripLog = round_trips(result.fills)

    return RunReport(
        strategy_name=result.strategy_name,
        mark=marca,
        liquidation=liquida,
        friction=friction_gap(result.equity, result.equity_liquidation),
        costs=costos,
        turnover=turnover,
        trades=trade_stats(log),
        risk_free=tasa,
        bars_per_year=bars_per_year,
        config=dict(result.config),
    )


# ---------------------------------------------------------------------------
# Renderizado
# ---------------------------------------------------------------------------


def _num(valor: float | None, *, decimales: int = 4, pct: bool = False) -> str:
    if valor is None:
        return "n/d"
    if math.isinf(valor):
        return "+inf" if valor > 0 else "-inf"
    if math.isnan(valor):
        return "nan"
    if pct:
        return f"{valor * 100:.2f}%"
    return f"{valor:,.{decimales}f}"


def _fila(etiqueta: str, izq: str, der: str) -> str:
    """Fila de la tabla comparativa: etiqueta y las dos series."""
    return f"  {etiqueta:<26}{izq:>18}{der:>18}"


def _dato(etiqueta: str, valor: str, nota: str = "") -> str:
    """Linea de una sola serie, con una nota opcional entre parentesis."""
    base = f"    {etiqueta:<26}{valor:>16}"
    return f"{base}   ({nota})" if nota else base


def _render(r: RunReport) -> str:
    m, lq = r.mark, r.liquidation
    lineas: list[str] = []
    lineas.append(f"Reporte de corrida: {r.strategy_name}")
    lineas.append(
        f"  {m.n_bars} barras, {m.years:.3f} anos, "
        f"bars_per_year={r.bars_per_year:g}, risk_free={r.risk_free:g}"
    )
    if not m.annualization_reliable:
        lineas.append(
            "  AVISO: el periodo cubre menos de un ano. El CAGR, el Sharpe y el "
            "Calmar estan anualizados por extrapolacion."
        )
    if m.ruined_at is not None:
        lineas.append(f"  AVISO: equity_mark toca cero en la barra {m.ruined_at}.")
    if lq.ruined_at is not None:
        lineas.append(
            f"  AVISO: equity_liquidation toca cero en la barra {lq.ruined_at}: "
            "cerrar la posicion no cubre el costo de salida."
        )

    lineas.append("")
    lineas.append(f"  {'':<26}{'equity_mark':>18}{'equity_liquid.':>18}")
    lineas.append("  " + "-" * 62)
    lineas.append(
        _fila(
            "equity inicial",
            _num(m.initial_equity, decimales=2),
            _num(lq.initial_equity, decimales=2),
        )
    )
    lineas.append(
        _fila(
            "equity final",
            _num(m.final_equity, decimales=2),
            _num(lq.final_equity, decimales=2),
        )
    )
    lineas.append(
        _fila(
            "retorno total",
            _num(m.total_return, pct=True),
            _num(lq.total_return, pct=True),
        )
    )
    lineas.append(_fila("CAGR", _num(m.cagr, pct=True), _num(lq.cagr, pct=True)))
    lineas.append(
        _fila(
            "volatilidad anual",
            _num(m.annual_volatility, pct=True),
            _num(lq.annual_volatility, pct=True),
        )
    )
    lineas.append(
        _fila("Sharpe", _num(m.sharpe, decimales=3), _num(lq.sharpe, decimales=3))
    )
    lineas.append(
        _fila("Sortino", _num(m.sortino, decimales=3), _num(lq.sortino, decimales=3))
    )
    lineas.append(
        _fila(
            "max drawdown",
            _num(m.drawdown.depth, pct=True),
            _num(lq.drawdown.depth, pct=True),
        )
    )
    lineas.append(
        _fila(
            "  pico -> valle (barras)",
            str(m.drawdown.duration_bars),
            str(lq.drawdown.duration_bars),
        )
    )
    lineas.append(
        _fila(
            "  bajo agua (barras)",
            str(m.drawdown.underwater_bars),
            str(lq.drawdown.underwater_bars),
        )
    )
    lineas.append(
        _fila("Calmar", _num(m.calmar, decimales=3), _num(lq.calmar, decimales=3))
    )

    f = r.friction
    lineas.append("")
    lineas.append("  Friccion no realizada (equity_mark - equity_liquidation)")
    lineas.append("  " + "-" * 62)
    lineas.append(_dato("brecha media", _num(f.mean_gap, decimales=2)))
    lineas.append(
        _dato(
            "brecha maxima", _num(f.max_gap, decimales=2), _num(f.max_gap_pct, pct=True)
        )
    )
    lineas.append(
        _dato(
            "brecha final",
            _num(f.final_gap, decimales=2),
            _num(f.final_gap_pct, pct=True),
        )
    )
    lineas.append(_dato("brecha minima", _num(f.min_gap, decimales=2)))
    if f.min_gap < 0:
        lineas.append(
            "    AVISO: brecha negativa. Con allow_short=False es imposible; "
            "hay un bug de signo en el motor."
        )

    c = r.costs
    lineas.append("")
    lineas.append("  Costos de transaccion (desglosados, nunca agregados)")
    lineas.append("  " + "-" * 62)
    lineas.append(_dato("spread", _num(c.spread, decimales=2)))
    lineas.append(_dato("slippage", _num(c.slippage, decimales=2)))
    lineas.append(_dato("comision", _num(c.commission, decimales=2)))
    lineas.append(
        _dato(
            "TOTAL costos pagados",
            _num(c.total_costs_paid, decimales=2),
            f"{_num(c.costs_bps_of_turnover, decimales=1)} bps del nocional",
        )
    )

    lineas.append("")
    lineas.append("  Gap de ejecucion (valuacion, NO es un costo; lleva signo)")
    lineas.append("  " + "-" * 62)
    lineas.append(
        _dato(
            "gap close[t]->open[t+1]",
            _num(c.gap, decimales=2),
            f"{_num(c.gap_pct_of_initial, pct=True)} del capital inicial",
        )
    )
    lineas.append(
        _dato(
            "implementation shortfall",
            _num(c.implementation_shortfall, decimales=2),
            "= costos + gap",
        )
    )
    lineas.append(_dato("interes devengado", _num(c.interest_earned, decimales=2)))

    t = r.trades
    lineas.append("")
    lineas.append("  Round-trips cerrados (flat a flat)")
    lineas.append("  " + "-" * 62)
    lineas.append(_dato("trades cerrados", str(t.n_round_trips)))
    lineas.append(_dato("ganados / perdidos", f"{t.n_wins} / {t.n_losses}"))
    if t.n_scratches:
        lineas.append(_dato("sin resultado (P&L = 0)", str(t.n_scratches)))
    lineas.append(_dato("win rate", _num(t.win_rate, pct=True)))
    lineas.append(_dato("profit factor", _num(t.profit_factor, decimales=3)))
    lineas.append(_dato("P&L neto realizado", _num(t.net_pnl, decimales=2)))
    lineas.append(_dato("barras en posicion", _num(t.avg_bars_held, decimales=1)))
    if t.open_position is not None:
        lineas.append(
            f"    posicion abierta al cierre: {t.open_position.qty:g} unidades, "
            f"base {t.open_position.cost_basis:,.2f}. No entra al win rate."
        )

    lineas.append("")
    lineas.append(
        f"  Turnover: {_num(r.turnover.notional, decimales=2)} nocional, "
        f"{_num(r.turnover.ratio, decimales=2)}x el equity medio, "
        f"{_num(r.turnover.annualized, decimales=2)}x anualizado"
    )
    lineas.append(
        f"  Fills ejecutados: {r.turnover.n_fills}"
        f"   rechazados: {r.turnover.n_rejected}"
    )
    return "\n".join(lineas)
