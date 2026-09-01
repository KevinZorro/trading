"""Motor de simulacion con latencia de una barra.

Secuencia de cada barra ``t``:

1. Se devenga interes sobre el cash del periodo anterior.
2. Se ejecuta la orden encolada en ``t-1``, al **open de t**.
3. Se marca el equity al **close de t**.
4. Se pide decision a la estrategia con una ventana que llega hasta ``t``.
   La orden resultante se encola para ``t+1``.

No existe un ``step()`` publico: el motor conduce el bucle. Un caller que
pudiera adelantar el cursor por su cuenta podria leer la barra siguiente antes
de decidir, que es justo lo que la Etapa 2 no debe poder hacer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from data.instruments import RejectReason as InstrumentRejectReason
from data.schema import BarSeries
from sim.costs import (
    CommissionModel,
    SlippageModel,
    SpreadContext,
    SpreadModel,
    ZeroSpread,
    NoSlippage,
    commission_from_instrument,
)
from sim.orders import Fill, MarketOrder, OrderStatus, RejectReason
from sim.portfolio import Ledger
from sim.view import AccountSnapshot, MarketView

# Rango del espacio de accion continuo mientras allow_short sea False. Vive aqui
# para que el entorno de la Etapa 2 lo importe en vez de redeclararlo: si el
# agente aprende sobre [-1, 1] y el simulador recorta en silencio a [0, 1], el
# agente optimiza sobre un rango que no existe.
LONG_ONLY_ACTION_RANGE: tuple[float, float] = (0.0, 1.0)


@runtime_checkable
class Strategy(Protocol):
    """Interfaz unica de decision. La misma que usaran los agentes de RL.

    ``on_bar`` recibe lo que se sabe en ``t`` y devuelve la orden a ejecutar en
    ``t+1``, o ``None`` para no operar.
    """

    name: str

    def reset(self, seed: int | None = None) -> None: ...

    def on_bar(
        self, view: MarketView, account: AccountSnapshot
    ) -> MarketOrder | None: ...


@dataclass(frozen=True)
class SimConfig:
    """Configuracion de una corrida. Se serializa junto al resultado."""

    initial_cash: float
    spread: SpreadModel = field(default_factory=ZeroSpread)
    slippage: SlippageModel = field(default_factory=NoSlippage)
    commission: CommissionModel | None = None
    max_participation: float = 0.10
    allow_short: bool = False
    cash_rate: float = 0.0
    bars_per_year: float = 252.0
    latency_bars: int = 1
    check_accounting: bool = True

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash debe ser positivo")
        if not 0.0 < self.max_participation <= 1.0:
            raise ValueError("max_participation debe estar en (0, 1]")
        if self.allow_short:
            raise NotImplementedError(
                "allow_short=True no esta implementado. Modelar mal el costo de "
                "prestamo es peor que no permitir cortos; el flag existe para "
                "que la interfaz no cambie cuando se implemente."
            )
        if self.latency_bars != 1:
            raise NotImplementedError(
                "solo se soporta latencia de una barra (decision en t, "
                "ejecucion al open de t+1)"
            )
        if self.cash_rate < 0:
            raise ValueError("cash_rate no puede ser negativo")

    @property
    def rate_per_bar(self) -> float:
        if self.cash_rate == 0.0:
            return 0.0
        return (1.0 + self.cash_rate) ** (1.0 / self.bars_per_year) - 1.0

    def describe(self) -> dict[str, object]:
        return {
            "initial_cash": self.initial_cash,
            "spread": self.spread.name,
            "slippage": self.slippage.name,
            "commission": self.commission.name if self.commission else "instrument",
            "max_participation": self.max_participation,
            "allow_short": self.allow_short,
            "cash_rate": self.cash_rate,
            "bars_per_year": self.bars_per_year,
            "latency_bars": self.latency_bars,
        }


@dataclass(frozen=True)
class SimResult:
    """Salida de una corrida: series por barra, log de fills y configuracion."""

    equity: np.ndarray
    equity_liquidation: np.ndarray
    cash: np.ndarray
    position: np.ndarray
    interest: np.ndarray
    timestamp: np.ndarray
    fills: list[Fill]
    config: dict[str, object]
    series_meta: dict[str, object]
    strategy_name: str

    @property
    def executed_fills(self) -> list[Fill]:
        return [f for f in self.fills if f.is_executed]

    @property
    def rejected_fills(self) -> list[Fill]:
        return [f for f in self.fills if f.status is OrderStatus.REJECTED]

    def cost_breakdown(self) -> dict[str, float]:
        """Desglose de P&L de ejecucion.

        ``gap`` va separado a proposito: es valuacion (la senal es de ``t`` y la
        ejecucion de ``t+1``), no un costo de transaccion. Sumarlos impediria
        distinguir un agente que opera demasiado de uno que opera en el peor
        momento del ciclo overnight.
        """
        executed = self.executed_fills
        spread = sum(f.spread_cost for f in executed)
        slippage = sum(f.slippage_cost for f in executed)
        commission = sum(f.commission for f in executed)
        gap = sum(f.gap for f in executed)
        return {
            "spread": spread,
            "slippage": slippage,
            "commission": commission,
            "total_costs_paid": spread + slippage + commission,
            "gap": gap,
            "implementation_shortfall": spread + slippage + commission + gap,
            "interest_earned": float(self.interest.sum()),
            "turnover_notional": sum(f.notional for f in executed),
            "n_fills": len(executed),
            "n_rejected": len(self.rejected_fills),
        }


class Simulator:
    """Ejecuta una estrategia contra una serie de barras."""

    def __init__(self, series: BarSeries, config: SimConfig) -> None:
        if len(series) < 2:
            raise ValueError("se necesitan al menos 2 barras para simular")
        self._series = series
        self._config = config
        self._commission = config.commission or commission_from_instrument(
            series.instrument
        )

    # -- ejecucion ------------------------------------------------------

    def _spread_context(self, t_decision: int, ref_price: float) -> SpreadContext:
        """Contexto para el spread: solo barras hasta la decision.

        Estimar el spread con el rango de la barra de ejecucion seria usar
        informacion que no existia cuando la orden se envio.
        """
        upto = t_decision + 1
        return SpreadContext(
            ref_price=ref_price,
            high=self._series.high[:upto],
            low=self._series.low[:upto],
            close=self._series.close[:upto],
        )

    def _reject(
        self,
        *,
        order: MarketOrder,
        t_decision: int,
        t_fill: int,
        decision_price: float,
        ref_price: float,
        reason: RejectReason,
        participation: float = 0.0,
    ) -> Fill:
        return Fill(
            t_decision=t_decision,
            t_fill=t_fill,
            timestamp_decision=self._series.timestamp[t_decision],
            timestamp_fill=self._series.timestamp[t_fill],
            symbol=self._series.symbol,
            status=OrderStatus.REJECTED,
            qty_requested=order.qty,
            qty_filled=0.0,
            decision_price=decision_price,
            ref_price=ref_price,
            fill_price=ref_price,
            gap=0.0,
            spread_cost=0.0,
            slippage_cost=0.0,
            commission=0.0,
            participation=participation,
            reject_reason=reason,
            tag=order.tag,
        )

    def _execute(
        self, order: MarketOrder, t_decision: int, t_fill: int, ledger: Ledger
    ) -> Fill:
        series = self._series
        config = self._config
        instrument = series.instrument
        symbol = series.symbol

        decision_price = float(series.close[t_decision])
        ref_price = float(series.open[t_fill])
        bar_volume = float(series.volume[t_fill])
        sign = 1.0 if order.qty > 0 else -1.0

        reject = lambda reason, participation=0.0: self._reject(  # noqa: E731
            order=order,
            t_decision=t_decision,
            t_fill=t_fill,
            decision_price=decision_price,
            ref_price=ref_price,
            reason=reason,
            participation=participation,
        )

        if bar_volume <= 0.0:
            return reject(RejectReason.NO_VOLUME)

        # 1. Capacidad de la barra: llenado parcial si la orden la supera.
        #    El remanente se cancela, no se arrastra: arrastrarlo convierte una
        #    orden de mercado en algo sin analogo real.
        capacity = config.max_participation * bar_volume
        qty_capped = sign * min(abs(order.qty), capacity)
        partial = abs(qty_capped) < abs(order.qty) - 1e-12

        # 2. Restricciones del instrumento (lote, cantidad y nocional minimos).
        qty_filled, instrument_reason = instrument.check_tradable(
            qty_capped, ref_price
        )
        participation = abs(qty_filled) / bar_volume if bar_volume else 0.0
        if instrument_reason is not None:
            return reject(
                RejectReason.from_instrument(instrument_reason), participation
            )

        # 3. Restricciones de la cuenta.
        position = ledger.position(symbol)
        if qty_filled < 0 and abs(qty_filled) > position + 1e-12:
            reason = (
                RejectReason.SHORT_NOT_ALLOWED
                if position <= 0
                else RejectReason.INSUFFICIENT_POSITION
            )
            return reject(reason, participation)

        # 4. Precio de ejecucion. Spread y slippage son SIEMPRE adversos:
        #    se suman en la direccion del signo de la orden, nunca en contra.
        ctx = self._spread_context(t_decision, ref_price)
        half_spread = self._config.spread.half_spread(ctx)
        impact = self._config.slippage.impact(participation)
        if half_spread < 0 or impact < 0:
            raise ValueError(
                "spread y slippage deben ser no negativos: el precio ejecutado "
                "nunca puede ser mejor que el de referencia"
            )
        slippage_per_unit = ref_price * impact
        fill_price = ref_price + sign * (half_spread + slippage_per_unit)
        commission = self._commission.compute(qty_filled, fill_price)

        # 5. El cash debe cubrir nocional + comision. Sin redimensionado
        #    silencioso: una orden que no cabe se rechaza y queda en el log.
        if qty_filled > 0:
            required = qty_filled * fill_price + commission
            if required > ledger.cash + 1e-9:
                return reject(RejectReason.INSUFFICIENT_CASH, participation)

        ledger.apply_fill(symbol, qty_filled, fill_price, commission)

        return Fill(
            t_decision=t_decision,
            t_fill=t_fill,
            timestamp_decision=series.timestamp[t_decision],
            timestamp_fill=series.timestamp[t_fill],
            symbol=symbol,
            status=OrderStatus.PARTIAL if partial else OrderStatus.FILLED,
            qty_requested=order.qty,
            qty_filled=qty_filled,
            decision_price=decision_price,
            ref_price=ref_price,
            fill_price=fill_price,
            # Valuacion: la senal es de t y la ejecucion de t+1. Lleva signo.
            gap=qty_filled * (ref_price - decision_price),
            # Costos de transaccion: no negativos por construccion.
            spread_cost=abs(qty_filled) * half_spread,
            slippage_cost=abs(qty_filled) * slippage_per_unit,
            commission=commission,
            participation=participation,
            tag=order.tag,
        )

    def _liquidation_equity(self, ledger: Ledger, t: int) -> float:
        """Equity si se cerrara la posicion al cierre de ``t``.

        Serie secundaria: el equity marcado a close ignora que salir cuesta.
        En el tier de riesgo alto, donde el spread y el impacto son grandes,
        el drawdown medido sobre esta serie es el que representa lo que el
        operador realmente podria retirar.
        """
        position = ledger.position(self._series.symbol)
        if position == 0.0:
            return ledger.cash
        close = float(self._series.close[t])
        bar_volume = float(self._series.volume[t])
        participation = abs(position) / bar_volume if bar_volume > 0 else 1.0
        ctx = self._spread_context(t, close)
        half_spread = self._config.spread.half_spread(ctx)
        impact = self._config.slippage.impact(participation)
        exit_price = close - (half_spread + close * impact)
        exit_price = max(exit_price, 0.0)
        commission = self._commission.compute(position, exit_price)
        return ledger.cash + position * exit_price - commission

    # -- bucle principal ------------------------------------------------

    def run(self, strategy: Strategy, *, seed: int | None = None) -> SimResult:
        """Corre la estrategia completa. Unica entrada publica del motor."""
        series = self._series
        config = self._config
        symbol = series.symbol
        n = len(series)

        strategy.reset(seed)
        ledger = Ledger(cash=config.initial_cash)

        equity = np.empty(n)
        equity_liq = np.empty(n)
        cash_series = np.empty(n)
        position_series = np.empty(n)
        interest_series = np.zeros(n)
        fills: list[Fill] = []

        pending: MarketOrder | None = None
        pending_t: int = -1

        for t in range(n):
            # 1. Interes sobre el cash del periodo anterior.
            if t > 0:
                interest_series[t] = ledger.accrue_interest(config.rate_per_bar)

            # 2. Ejecucion de lo decidido en t-1, al open de t.
            if pending is not None:
                fills.append(self._execute(pending, pending_t, t, ledger))
                pending = None

            # 3. Marca a close de t.
            close = float(series.close[t])
            equity[t] = ledger.mark_to_market({symbol: close})
            equity_liq[t] = self._liquidation_equity(ledger, t)
            cash_series[t] = ledger.cash
            position_series[t] = ledger.position(symbol)
            if config.check_accounting:
                ledger.check_identity(equity[t], {symbol: close})

            # 4. Decision con datos hasta t. La ventana no referencia t+1.
            view = MarketView(series, t)
            account = AccountSnapshot(
                t=t,
                cash=ledger.cash,
                position=ledger.position(symbol),
                mark_price=close,
                equity=equity[t],
                pending_qty=0.0,
            )
            order = strategy.on_bar(view, account)
            if order is not None:
                if not isinstance(order, MarketOrder):
                    raise TypeError(
                        f"on_bar debe devolver MarketOrder o None, devolvio "
                        f"{type(order).__name__}"
                    )
                if order.qty == 0.0:
                    order = None
            if order is not None:
                if t == n - 1:
                    # Una orden decidida en la ultima barra no tiene barra de
                    # ejecucion. Se registra como expirada en vez de perderse.
                    fills.append(
                        Fill(
                            t_decision=t,
                            t_fill=t,
                            timestamp_decision=series.timestamp[t],
                            timestamp_fill=series.timestamp[t],
                            symbol=symbol,
                            status=OrderStatus.EXPIRED,
                            qty_requested=order.qty,
                            qty_filled=0.0,
                            decision_price=close,
                            ref_price=close,
                            fill_price=close,
                            gap=0.0,
                            spread_cost=0.0,
                            slippage_cost=0.0,
                            commission=0.0,
                            participation=0.0,
                            tag=order.tag,
                        )
                    )
                else:
                    pending = order
                    pending_t = t

        return SimResult(
            equity=equity,
            equity_liquidation=equity_liq,
            cash=cash_series,
            position=position_series,
            interest=interest_series,
            timestamp=series.timestamp,
            fills=fills,
            config=config.describe(),
            series_meta=series.meta(),
            strategy_name=getattr(strategy, "name", type(strategy).__name__),
        )
