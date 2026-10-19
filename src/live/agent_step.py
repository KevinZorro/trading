"""Un paso en vivo del Agente A: decide -> sizer -> riesgo -> venue -> settle.

Cada funcion que decide el resultado es la misma que usa el entorno: la
observacion sale de ``ObservationBuilder.build(Decision)``, la orden de
``TargetWeightSizer.order_for`` y el veto de un ``OrderGate``. Lo que **no** es el
mismo codigo es el bucle: ``Simulator.drive`` conduce el suyo y este paso es otro
(paridad parcial, declarada en la propuesta de la 5b).

Convencion temporal (CLAUDE.md): la barra ``t`` cierra en su ``timestamp``; el paso
corre a las 00:00:30, decide con ``close[t]`` y ejecuta al libro de ese instante, que
es el open de ``t+1``. Una barra mas vieja que ``max_bar_age`` se rechaza: decidir con
ella y ejecutar mas tarde meteria un gap que el simulador nunca ve.

La cuenta autoritativa es la del venue (``reconcile``), no la memoria del paso. El
log del ``FillStore`` guarda tambien la decision, **antes** de enviar: tras una caida
el paso retoma la orden con el mismo ``client_order_id`` (deterministico por barra) y
el venue la deduplica, sin volver a consultar a la politica.

ASSUMPTION: el estado de ``RiskLayer`` (perdida del dia, kill switch) vive en memoria
y no sobrevive a un reinicio. Persistirlo es trabajo de un PR propio antes de que
exista dinero real; en sombra no hay capital en juego.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from data.schema import FloatArray
from envs.observation import ObservationBuilder
from live.fillstore import FillStore, day_of
from live.frozen import FrozenPolicy
from live.marketdata import MarketData
from live.venue import ShadowVenue
from sim.clock import Clock
from sim.costs import SchemaCommission
from sim.engine import LONG_ONLY_ACTION_RANGE, Decision
from sim.gate import GateRejection, OrderGate
from sim.orders import Fill, MarketOrder
from sim.sizing import TargetWeightSizer
from sim.view import AccountSnapshot, MarketView

DECISION = "decision"
MAX_BAR_AGE = np.timedelta64(1, "h")


class StaleBarsError(RuntimeError):
    """La ultima barra cerrada no es la que corresponde decidir ahora."""


@dataclass(frozen=True)
class StepOutcome:
    bar_timestamp: np.datetime64
    raw_action: float | None  # None: se retomo una decision ya registrada
    target_weight: float
    order: MarketOrder | None
    veto: GateRejection | None
    fills: tuple[Fill, ...]
    resumed: bool


def liquidation_equity(
    cash: float, position: float, bids: FloatArray, commission: SchemaCommission
) -> float:
    """Equity si se cerrara la posicion recorriendo el lado bid del libro.

    Es lo que ``Simulator._liquidation_equity`` calcula con un modelo; aca sale del
    libro real. ASSUMPTION: si el libro visible no alcanza, el remanente se valua al
    peor nivel visible. Con 100 USDT el libro de 1000 niveles sobra.
    """
    if position == 0.0:
        return cash
    restante, ingreso = position, 0.0
    for precio, cantidad in bids:
        tomada = min(restante, float(cantidad))
        ingreso += tomada * float(precio)
        restante -= tomada
        if restante <= 0.0:
            break
    if restante > 0.0:
        ingreso += restante * float(bids[-1, 0])
    return cash + ingreso - commission.compute(position, ingreso / position)


def _veto_from(rec: dict[str, Any], symbol: str) -> GateRejection | None:
    if not rec.get("veto_reason"):
        return None
    return GateRejection(
        t_decision=0,
        timestamp_decision=np.datetime64(rec["bar_timestamp"], "ns"),
        symbol=symbol,
        client_order_id=rec["client_order_id"],
        qty_requested=rec["qty"],
        reason=rec["veto_reason"],
        detail=rec["veto_detail"],
        tag=rec["tag"],
    )


class AgentStep:
    def __init__(
        self,
        *,
        agent: FrozenPolicy,
        builder: ObservationBuilder,
        sizer: TargetWeightSizer,
        gate: OrderGate,
        venue: ShadowVenue,
        store: FillStore,
        market: MarketData,
        symbol: str,
        commission: SchemaCommission,
        clock: Clock,
        run_id: str,
        interval: str = "1d",
        max_bar_age: np.timedelta64 = MAX_BAR_AGE,
        depth_limit: int = 1000,
    ) -> None:
        self._agent = agent
        self._builder = builder
        self._sizer = sizer
        self._gate = gate
        self._venue = venue
        self._store = store
        self._market = market
        self._symbol = symbol
        self._commission = commission
        self._clock = clock
        self._run_id = run_id
        self._interval = interval
        self._max_age = max_bar_age
        self._depth_limit = depth_limit

    def _decisions(self) -> list[dict[str, Any]]:
        return [r for r in self._store.replay() if r["type"] == DECISION]

    def run(self) -> StepOutcome:
        bars = self._market.closed_klines(
            self._symbol, self._interval, self._builder.spec.warmup + 1
        )
        t = len(bars) - 1
        ts = bars.timestamp[t]
        edad = self._clock.now() - ts
        if edad < np.timedelta64(0, "ns") or edad > self._max_age:
            raise StaleBarsError(
                f"ultima barra cerrada {ts}, ahora {self._clock.now()}: edad {edad}"
            )
        clave = str(np.datetime64(ts, "ns"))
        previas = self._decisions()
        hecha = next((r for r in previas if r["bar_timestamp"] == clave), None)
        anteriores = [r for r in previas if r["bar_timestamp"] < clave]
        anterior = anteriores[-1] if anteriores else None

        estado = self._venue.reconcile()  # la posicion la reporta el venue
        posicion = estado.position(self._symbol)
        close = float(bars.close[t])
        cuenta = AccountSnapshot(
            t=t,
            cash=estado.cash,
            position=posicion,
            mark_price=close,
            equity=estado.cash + posicion * close,
            pending_qty=0.0,
        )
        self._gate.observe_bar(cuenta)

        if hecha is not None:
            return self._enviar(hecha, close, resumed=True)

        # "no quise" vs "no pude": solo cuenta el fill de la orden de la decision
        # anterior, igual que en el simulador, no el ultimo fill que haya existido.
        ultimo_fill = None
        if anterior is not None and anterior["client_order_id"]:
            ultimo_fill = next(
                (
                    f
                    for f in reversed(self._venue.get_fills())
                    if f.client_order_id == anterior["client_order_id"]
                ),
                None,
            )
        libro = self._market.depth(self._symbol, self._depth_limit)
        decision = Decision(
            view=MarketView(bars, t),
            account=cuenta,
            equity_liquidation=liquidation_equity(
                cuenta.cash, posicion, libro.bids, self._commission
            ),
            last_fill=ultimo_fill,
            last_gate_rejection=(
                _veto_from(anterior, self._symbol) if anterior is not None else None
            ),
        )
        cruda = self._agent.act(self._builder.build(decision))
        if not np.isfinite(cruda):
            raise ValueError(f"accion no finita: {cruda!r}")
        bajo, alto = LONG_ONLY_ACTION_RANGE
        objetivo = min(max(cruda, bajo), alto)
        orden = self._sizer.order_for(objetivo, decision.view, cuenta)

        rec: dict[str, Any] = {
            "type": DECISION,
            "bar_timestamp": clave,
            "raw_action": cruda,
            "target_weight": objetivo,
            "qty": 0.0,
            "tag": "",
            "client_order_id": "",
            "veto_reason": "",
            "veto_detail": "",
        }
        if orden is not None:
            # Deterministico por barra: reintentar tras una caida es idempotente.
            orden = replace(orden, client_order_id=f"{self._run_id}-{clave}")
            rec.update(
                qty=orden.qty, tag=orden.tag, client_order_id=orden.client_order_id
            )
            veredicto = self._gate.check(orden, cuenta)
            if not veredicto.approved:
                rec.update(veto_reason=veredicto.reason, veto_detail=veredicto.detail)
        self._store.append(day_of(self._clock.now()), rec)
        return self._enviar(rec, close, resumed=False)

    def _enviar(
        self, rec: dict[str, Any], close: float, *, resumed: bool
    ) -> StepOutcome:
        orden = (
            MarketOrder(rec["qty"], rec["tag"], rec["client_order_id"])
            if rec["client_order_id"]
            else None
        )
        veto = _veto_from(rec, self._symbol)
        fills: list[Fill] = []
        if orden is not None and veto is None:
            self._venue.submit_order(orden)  # duplicado tras reinicio: no repite
            fills = self._venue.settle(decision_price=close)
            for f in fills:
                self._gate.observe_fill(f)
        return StepOutcome(
            bar_timestamp=np.datetime64(rec["bar_timestamp"], "ns"),
            raw_action=None if resumed else rec["raw_action"],
            target_weight=rec["target_weight"],
            order=orden,
            veto=veto,
            fills=tuple(fills),
            resumed=resumed,
        )
