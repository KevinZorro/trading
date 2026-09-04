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

``Simulator`` implementa ademas :class:`~sim.venue.ExecutionVenue`. Los metodos
del protocolo (``submit_order``, ``cancel_order``, ``get_positions``,
``get_fills``, ``reconcile``) son publicos y no comprometen la garantia de
arriba: ninguno recibe ni devuelve una barra, ninguno mueve el cursor ``_t``, y
``MarketView`` sigue siendo el unico canal hacia los datos de mercado. Lo que si
hacen es partir la validacion en dos tiempos, como un broker real: **admision**
al enviar y **ejecucion** al llenar.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

import numpy as np

from data.schema import BarSeries, FloatArray, TimeArray
from sim.clock import Clock, SimulatedClock
from sim.costs import (
    CommissionModel,
    NoSlippage,
    SlippageModel,
    SpreadContext,
    SpreadModel,
    ZeroSpread,
    commission_from_instrument,
)
from sim.gate import GateRejection, OrderGate
from sim.ids import OrderIdGenerator, SequentialIds
from sim.orders import Fill, MarketOrder, OrderStatus, RejectReason
from sim.portfolio import Ledger
from sim.venue import CancelAck, OrderAck, VenueState
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
    # Semilla del generador de client_order_id. Fijo por configuracion, no
    # aleatorio: dos corridas del mismo backtest deben producir exactamente los
    # mismos identificadores o sus logs no son comparables.
    run_id: str = "sim"

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
        if not self.run_id:
            raise ValueError("run_id no puede ser vacio")

    @property
    def rate_per_bar(self) -> float:
        if self.cash_rate == 0.0:
            return 0.0
        return float((1.0 + self.cash_rate) ** (1.0 / self.bars_per_year) - 1.0)

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
            "run_id": self.run_id,
        }


@dataclass(frozen=True)
class SimResult:
    """Salida de una corrida: series por barra, log de fills y configuracion."""

    equity: FloatArray
    equity_liquidation: FloatArray
    cash: FloatArray
    position: FloatArray
    interest: FloatArray
    timestamp: TimeArray
    fills: list[Fill]
    config: dict[str, object]
    series_meta: dict[str, object]
    strategy_name: str
    # Ordenes que la capa de riesgo veto antes de que llegaran al venue. Lista
    # aparte a proposito: no son fills y no son rechazos del venue. Ver
    # `sim.gate`.
    gate_rejections: list[GateRejection] = field(default_factory=list)

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
            "n_gate_rejected": len(self.gate_rejections),
        }


class Simulator:
    """Ejecuta una estrategia contra una serie de barras.

    Implementa :class:`~sim.venue.ExecutionVenue`, asi que el mismo runner y la
    misma capa de riesgo que operan contra esto pueden operar contra un broker
    real sin cambiar una linea.
    """

    def __init__(
        self,
        series: BarSeries,
        config: SimConfig,
        *,
        clock: Clock | None = None,
        order_ids: OrderIdGenerator | None = None,
    ) -> None:
        if len(series) < 2:
            raise ValueError("se necesitan al menos 2 barras para simular")
        self._series = series
        self._config = config
        self._commission = config.commission or commission_from_instrument(
            series.instrument
        )
        # El reloj arranca en la primera barra. En vivo se inyecta SystemClock
        # aqui mismo y nada mas cambia.
        self._clock: Clock = clock or SimulatedClock(series.timestamp[0])
        self._order_ids: OrderIdGenerator = order_ids or SequentialIds(config.run_id)

        # Estado de venue. Vive en la instancia porque el protocolo lo consulta
        # entre llamados; `run` lo reinicia al empezar cada corrida.
        self._ledger = Ledger(cash=config.initial_cash)
        self._fills: list[Fill] = []
        self._pending: MarketOrder | None = None
        self._pending_t: int = -1
        self._acks: dict[str, OrderAck] = {}
        self._t: int = -1

    def _reset_venue(self) -> None:
        """Vuelve al estado de arranque. Solo lo llama ``run``."""
        self._ledger = Ledger(cash=self._config.initial_cash)
        self._fills = []
        self._pending = None
        self._pending_t = -1
        self._acks = {}
        self._t = -1
        if isinstance(self._order_ids, SequentialIds):
            self._order_ids.reset()
        if isinstance(self._clock, SimulatedClock):
            # Rebobinar el reloj: una instancia de Simulator se puede correr mas
            # de una vez y la segunda corrida empieza en la primera barra.
            self._clock.reset_to(self._series.timestamp[0])

    # No se expone el reloj como propiedad publica: `SimulatedClock.advance_to`
    # es mutable y no hay razon para que nadie mas que el motor lo mueva. Quien
    # inyecta el reloj ya tiene su propia referencia.

    # -- ExecutionVenue -------------------------------------------------

    def submit_order(self, order: MarketOrder) -> OrderAck:
        """Acusa **recepcion**. El fill, si lo hay, ocurre al open de ``t+1``.

        Aqui solo se valida la admision. Que la orden quepa en el volumen de la
        barra, respete los minimos del instrumento y tenga cash detras se decide
        al llenar, en ``_execute``, porque en el momento del envio esos datos
        todavia no existen: son de la barra siguiente. Adelantarlos seria
        lookahead.
        """
        if not order.client_order_id:
            return OrderAck(
                client_order_id="",
                accepted=False,
                reject_reason=RejectReason.MISSING_CLIENT_ORDER_ID,
                detail=(
                    "el identificador lo genera el emisor, no el venue: sin el, "
                    "un reintento tras un timeout duplica la posicion"
                ),
            )

        previo = self._acks.get(order.client_order_id)
        if previo is not None:
            # Idempotencia. Este es el caso que en vivo evita la posicion doble
            # cuando la respuesta al primer envio se perdio en la red.
            return replace(previo, is_duplicate=True)

        if self._t < 0 or self._t >= len(self._series):
            ack = OrderAck(
                client_order_id=order.client_order_id,
                accepted=False,
                reject_reason=RejectReason.VENUE_NOT_RUNNING,
                detail="el venue no esta procesando una barra",
            )
            self._acks[order.client_order_id] = ack
            return ack

        ack = OrderAck(client_order_id=order.client_order_id, accepted=True)
        self._acks[order.client_order_id] = ack
        self._pending = order
        self._pending_t = self._t
        return ack

    def cancel_order(self, client_order_id: str) -> CancelAck:
        """Cancela la orden encolada si el identificador coincide."""
        if self._pending is not None and (
            self._pending.client_order_id == client_order_id
        ):
            self._pending = None
            self._pending_t = -1
            return CancelAck(client_order_id=client_order_id, cancelled=True)
        return CancelAck(
            client_order_id=client_order_id,
            cancelled=False,
            detail="no hay una orden viva con ese identificador",
        )

    def get_positions(self) -> dict[str, float]:
        """Copia de las posiciones. Nadie muta la contabilidad desde afuera."""
        return dict(self._ledger.positions)

    def get_fills(self, since: int = 0) -> list[Fill]:
        """Fills desde el indice ``since``. El indice es el numero de secuencia."""
        if since < 0:
            raise ValueError("since no puede ser negativo")
        return list(self._fills[since:])

    def reconcile(self) -> VenueState:
        """Estado autoritativo. La estrategia adopta esto, no su memoria.

        En simulacion es trivial porque el venue y el libro son el mismo objeto.
        Existe igual desde el primer dia para que el runner de la Etapa 7 no
        tenga que inventar el paso de reconciliacion cuando ya haya dinero real
        en juego.
        """
        vivas = (self._pending.client_order_id,) if self._pending is not None else ()
        return VenueState(
            timestamp=self._clock.now(),
            cash=self._ledger.cash,
            positions=dict(self._ledger.positions),
            open_order_ids=vivas,
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
            client_order_id=order.client_order_id,
        )

    def _execute(self, order: MarketOrder, t_decision: int, t_fill: int) -> Fill:
        ledger = self._ledger
        series = self._series
        config = self._config
        instrument = series.instrument
        symbol = series.symbol

        decision_price = float(series.close[t_decision])
        ref_price = float(series.open[t_fill])
        bar_volume = float(series.volume[t_fill])
        sign = 1.0 if order.qty > 0 else -1.0

        def reject(reason: RejectReason, participation: float = 0.0) -> Fill:
            return self._reject(
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
        qty_filled, instrument_reason = instrument.check_tradable(qty_capped, ref_price)
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
            client_order_id=order.client_order_id,
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

    def run(
        self,
        strategy: Strategy,
        *,
        seed: int | None = None,
        gate: OrderGate | None = None,
    ) -> SimResult:
        """Corre la estrategia completa. Unica entrada publica del motor.

        ``gate`` es la capa de riesgo. Se aplica **entre** la decision de la
        estrategia y el envio al venue, que es exactamente el mismo punto en el
        que la aplicara el runner en vivo. Sin gate el comportamiento es el de
        siempre.
        """
        series = self._series
        config = self._config
        symbol = series.symbol
        n = len(series)

        strategy.reset(seed)
        self._reset_venue()

        # Reconciliacion de arranque. En simulacion no puede sorprender a nadie;
        # se hace igual para que el runner en vivo no estrene este paso el dia
        # que haya dinero real.
        estado_inicial = self.reconcile()
        if estado_inicial.positions:
            raise ValueError(
                f"el venue arranca con posiciones abiertas: "
                f"{estado_inicial.positions!r}"
            )

        equity = np.empty(n)
        equity_liq = np.empty(n)
        cash_series = np.empty(n)
        position_series = np.empty(n)
        interest_series = np.zeros(n)
        vetadas: list[GateRejection] = []

        for t in range(n):
            self._t = t
            if isinstance(self._clock, SimulatedClock):
                self._clock.advance_to(series.timestamp[t])

            # 1. Interes sobre el cash del periodo anterior.
            if t > 0:
                interest_series[t] = self._ledger.accrue_interest(config.rate_per_bar)

            # 2. Ejecucion de lo decidido en t-1, al open de t.
            if self._pending is not None:
                fill = self._execute(self._pending, self._pending_t, t)
                self._fills.append(fill)
                self._pending = None
                self._pending_t = -1
                if gate is not None:
                    gate.observe_fill(fill)

            # 3. Marca a close de t.
            close = float(series.close[t])
            equity[t] = self._ledger.mark_to_market({symbol: close})
            equity_liq[t] = self._liquidation_equity(self._ledger, t)
            cash_series[t] = self._ledger.cash
            position_series[t] = self._ledger.position(symbol)
            if config.check_accounting:
                self._ledger.check_identity(equity[t], {symbol: close})

            # 4. Decision con datos hasta t. La ventana no referencia t+1.
            view = MarketView(series, t)
            account = AccountSnapshot(
                t=t,
                cash=self._ledger.cash,
                position=self._ledger.position(symbol),
                mark_price=close,
                equity=equity[t],
                pending_qty=0.0,
            )
            if gate is not None:
                gate.observe_bar(account)

            order = strategy.on_bar(view, account)
            if order is not None:
                if not isinstance(order, MarketOrder):
                    raise TypeError(
                        f"on_bar debe devolver MarketOrder o None, devolvio "
                        f"{type(order).__name__}"
                    )
                if order.qty == 0.0:
                    order = None

            if order is None:
                continue

            # 5. El emisor pone el identificador antes de enviar. Una estrategia
            #    puede traer el suyo; si no, lo asigna el runner.
            if not order.client_order_id:
                order = replace(order, client_order_id=self._order_ids.next_id())

            # 6. Capa de riesgo. Rechaza, nunca redimensiona: redimensionar en
            #    silencio hace indistinguible "el agente pidio esto" de "el
            #    limite lo recorto hasta aca".
            if gate is not None:
                decision = gate.check(order, account)
                if not decision.approved:
                    vetadas.append(
                        GateRejection(
                            t_decision=t,
                            timestamp_decision=series.timestamp[t],
                            symbol=symbol,
                            client_order_id=order.client_order_id,
                            qty_requested=order.qty,
                            reason=decision.reason,
                            detail=decision.detail,
                            tag=order.tag,
                        )
                    )
                    continue

            # 7. Envio al venue. El acuse no es un fill.
            self.submit_order(order)

        # Una orden encolada en la ultima barra no tiene barra de ejecucion. Se
        # registra como expirada en vez de perderse.
        if self._pending is not None:
            expirada = self._pending
            t_exp = self._pending_t
            cierre = float(series.close[t_exp])
            self._fills.append(
                Fill(
                    t_decision=t_exp,
                    t_fill=t_exp,
                    timestamp_decision=series.timestamp[t_exp],
                    timestamp_fill=series.timestamp[t_exp],
                    symbol=symbol,
                    status=OrderStatus.EXPIRED,
                    qty_requested=expirada.qty,
                    qty_filled=0.0,
                    decision_price=cierre,
                    ref_price=cierre,
                    fill_price=cierre,
                    gap=0.0,
                    spread_cost=0.0,
                    slippage_cost=0.0,
                    commission=0.0,
                    participation=0.0,
                    tag=expirada.tag,
                    client_order_id=expirada.client_order_id,
                )
            )
            self._pending = None
            self._pending_t = -1

        return SimResult(
            equity=equity,
            equity_liquidation=equity_liq,
            cash=cash_series,
            position=position_series,
            interest=interest_series,
            timestamp=series.timestamp,
            fills=list(self._fills),
            config=config.describe(),
            series_meta=series.meta(),
            strategy_name=getattr(strategy, "name", type(strategy).__name__),
            gate_rejections=vetadas,
        )
