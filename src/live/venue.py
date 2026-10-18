"""Venue sombra: implementa ``ExecutionVenue`` y llena contra el libro L2 real.

No envia nada a ningun lado. Una orden se acusa al recibirla (``submit_order``)
y se ejecuta despues, en ``settle()``, que toma el libro y recorre sus niveles.
``settle`` es el unico metodo publico fuera del protocolo y juega el papel del
canal de fills de un broker real: ejecutar dentro de ``submit_order`` ensenaria
a escribir estrategias que no pueden existir en vivo (``src/sim/CLAUDE.md``).

Los costos salen del libro, no de un modelo, y se registran separados como en
el log del simulador: ``spread_cost`` es el medio spread, ``slippage_cost`` el
impacto de profundidad y ``commission`` la tarifa del ``InstrumentSpec``. El
``InstrumentSpec`` entra por el constructor y nunca de ``binance_spot_spec``
(ADR 0008).

Todo se persiste con append y ``fsync`` antes de tocar la memoria, y el estado
se reconstruye repitiendo el log: en sombra no hay broker que reporte la
posicion, asi que el log es el estado autoritativo.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from data.instruments import InstrumentSpec
from data.schema import FloatArray
from live.fillstore import FillStore, day_of, fill_from_dict, fill_to_dict
from live.marketdata import BookSnapshot, MarketData
from sim.clock import Clock
from sim.costs import SchemaCommission
from sim.orders import Fill, MarketOrder, OrderStatus, RejectReason
from sim.portfolio import Ledger
from sim.venue import CancelAck, OrderAck, VenueState

DEPTH_LIMIT = 1000


class ShadowVenueError(RuntimeError):
    """Estado imposible del venue: se detiene, no se corrige en silencio."""


@dataclass(frozen=True)
class _Pending:
    order: MarketOrder
    timestamp: np.datetime64  # hora del Clock al aceptar la orden
    seq: int


def _walk_qty(niveles: FloatArray, qty: float) -> tuple[float, float]:
    """``(mejor precio, vwap)`` al consumir ``qty`` unidades del lado.

    Misma forma que ``walk_book`` (``best + suma(q*(p-best))/suma(q)``): un
    llenado en el primer nivel da vwap == best exacto.
    """
    best = float(niveles[0, 0])
    restante = qty
    desvio = 0.0
    for precio, cantidad in niveles:
        tomada = min(restante, float(cantidad))
        desvio += tomada * (float(precio) - best)
        restante -= tomada
        if restante <= 0.0:
            break
    return best, best + desvio / qty


class ShadowVenue:
    def __init__(
        self,
        *,
        market: MarketData,
        instrument: InstrumentSpec,
        clock: Clock,
        store: FillStore,
        initial_cash: float = 100.0,
        depth_limit: int = DEPTH_LIMIT,
    ) -> None:
        if initial_cash <= 0:
            raise ValueError("initial_cash debe ser positivo")
        self._market = market
        self._instrument = instrument
        self._clock = clock
        self._store = store
        self._depth_limit = depth_limit
        self._commission = SchemaCommission(schema=instrument.commission_schema)
        self._ledger = Ledger(cash=initial_cash)
        self._fills: list[Fill] = []
        self._acks: dict[str, OrderAck] = {}
        self._pending: dict[str, _Pending] = {}
        self._seq = 0
        self._replay()

    def _replay(self) -> None:
        """Reconstruye cash, posicion, acuses y ordenes vivas desde el log."""
        for r in self._store.replay():
            if r["type"] == "ack":
                motivo = r["reject_reason"]
                ack = OrderAck(
                    client_order_id=r["client_order_id"],
                    accepted=r["accepted"],
                    reject_reason=RejectReason(motivo) if motivo else None,
                    detail=r["detail"],
                )
                self._acks[ack.client_order_id] = ack
                if ack.accepted:
                    self._seq = r["seq"]
                    self._pending[ack.client_order_id] = _Pending(
                        MarketOrder(r["qty"], r["tag"], ack.client_order_id),
                        np.datetime64(r["timestamp"], "ns"),
                        r["seq"],
                    )
            elif r["type"] == "fill":
                fill = fill_from_dict(r["fill"])
                self._fills.append(fill)
                self._pending.pop(fill.client_order_id, None)
                if fill.is_executed:
                    self._ledger.apply_fill(
                        fill.symbol, fill.qty_filled, fill.fill_price, fill.commission
                    )
            elif r["type"] == "cancel":
                self._pending.pop(r["client_order_id"], None)

    # -- ExecutionVenue -------------------------------------------------

    def submit_order(self, order: MarketOrder) -> OrderAck:
        """Acusa **recepcion**. El fill ocurre en ``settle``."""
        if not order.client_order_id:
            return OrderAck(
                client_order_id="",
                accepted=False,
                reject_reason=RejectReason.MISSING_CLIENT_ORDER_ID,
                detail="el identificador lo genera el emisor, no el venue",
            )
        previo = self._acks.get(order.client_order_id)
        if previo is not None:
            return replace(
                previo, is_duplicate=True
            )  # idempotente, tambien tras reiniciar
        ack = OrderAck(client_order_id=order.client_order_id, accepted=True)
        ahora = self._clock.now()
        self._store.append(
            day_of(ahora),
            {
                "type": "ack",
                "client_order_id": ack.client_order_id,
                "accepted": True,
                "reject_reason": None,
                "detail": "",
                "qty": order.qty,
                "tag": order.tag,
                "timestamp": str(ahora),
                "seq": self._seq + 1,
            },
        )
        self._seq += 1
        self._acks[ack.client_order_id] = ack
        self._pending[ack.client_order_id] = _Pending(order, ahora, self._seq)
        return ack

    def cancel_order(self, client_order_id: str) -> CancelAck:
        if client_order_id not in self._pending:
            return CancelAck(
                client_order_id=client_order_id,
                cancelled=False,
                detail="no hay una orden viva con ese identificador",
            )
        self._store.append(
            day_of(self._clock.now()),
            {"type": "cancel", "client_order_id": client_order_id},
        )
        del self._pending[client_order_id]
        return CancelAck(client_order_id=client_order_id, cancelled=True)

    def get_positions(self) -> dict[str, float]:
        return dict(self._ledger.positions)

    def get_fills(self, since: int = 0) -> list[Fill]:
        if since < 0:
            raise ValueError("since no puede ser negativo")
        return list(self._fills[since:])

    def reconcile(self) -> VenueState:
        return VenueState(
            timestamp=self._clock.now(),
            cash=self._ledger.cash,
            positions=dict(self._ledger.positions),
            open_order_ids=tuple(self._pending),
        )

    # -- ejecucion ------------------------------------------------------

    def settle(self, decision_price: float | None = None) -> list[Fill]:
        """Ejecuta contra el libro actual lo que esta encolado, en orden.

        ``decision_price`` es el ``close[t]`` que vio quien decidio; el venue no
        lo conoce. Sin el, el gap es 0. Una falla de red propaga y deja la orden
        viva: reintentar es seguro porque el fill se persiste antes de aplicarse.
        """
        salida: list[Fill] = []
        for pendiente in list(self._pending.values()):
            libro = self._market.depth(self._instrument.symbol, self._depth_limit)
            if libro.received_at < pendiente.timestamp:
                raise ShadowVenueError(
                    f"libro recibido {libro.received_at} anterior a la orden "
                    f"{pendiente.timestamp}: reloj o mercado inconsistentes"
                )
            fill = self._execute(pendiente, libro, decision_price)
            self._store.append(
                day_of(libro.received_at), {"type": "fill", "fill": fill_to_dict(fill)}
            )
            self._fills.append(fill)
            del self._pending[pendiente.order.client_order_id]
            if fill.is_executed:
                self._ledger.apply_fill(
                    fill.symbol, fill.qty_filled, fill.fill_price, fill.commission
                )
            salida.append(fill)
        return salida

    def _execute(
        self, p: _Pending, libro: BookSnapshot, decision_price: float | None
    ) -> Fill:
        inst = self._instrument
        order = p.order
        mid = (libro.best_bid + libro.best_ask) / 2.0
        decision = mid if decision_price is None else decision_price
        sign = 1.0 if order.qty > 0 else -1.0
        niveles = libro.asks if sign > 0 else libro.bids

        def fill(
            status: OrderStatus,
            qty: float = 0.0,
            price: float = mid,
            spread: float = 0.0,
            slip: float = 0.0,
            comision: float = 0.0,
            part: float = 0.0,
            motivo: RejectReason | None = None,
        ) -> Fill:
            return Fill(
                t_decision=p.seq,
                t_fill=p.seq,
                timestamp_decision=p.timestamp,
                timestamp_fill=libro.received_at,
                symbol=inst.symbol,
                status=status,
                qty_requested=order.qty,
                qty_filled=qty,
                decision_price=decision,
                ref_price=mid,
                fill_price=price,
                gap=qty * (mid - decision),
                spread_cost=spread,
                slippage_cost=slip,
                commission=comision,
                participation=part,
                reject_reason=motivo,
                tag=order.tag,
                client_order_id=order.client_order_id,
            )

        def rechazo(motivo: RejectReason, part: float = 0.0) -> Fill:
            return fill(OrderStatus.REJECTED, part=part, motivo=motivo)

        qty, instrumento = inst.check_tradable(order.qty, mid)
        if instrumento is not None:
            return rechazo(RejectReason.from_instrument(instrumento))

        # ASSUMPTION: si el libro visible no alcanza, se llena lo que hay y el
        # remanente se cancela (misma regla que max_participation en Simulator).
        visible = float(niveles[:, 1].sum())
        parcial = abs(qty) > visible
        if parcial:
            qty, instrumento = inst.check_tradable(sign * visible, mid)
            if instrumento is not None:
                return rechazo(RejectReason.from_instrument(instrumento), 1.0)
        part = abs(qty) / visible

        posicion = self._ledger.position(inst.symbol)
        if qty < 0 and abs(qty) > posicion + 1e-12:
            return rechazo(
                RejectReason.SHORT_NOT_ALLOWED
                if posicion <= 0
                else RejectReason.INSUFFICIENT_POSITION,
                part,
            )

        best, vwap = _walk_qty(niveles, abs(qty))
        comision = self._commission.compute(qty, vwap)
        if qty > 0 and qty * vwap + comision > self._ledger.cash + 1e-9:
            return rechazo(RejectReason.INSUFFICIENT_CASH, part)
        return fill(
            OrderStatus.PARTIAL if parcial else OrderStatus.FILLED,
            qty=qty,
            price=vwap,
            spread=abs(qty) * abs(best - mid),
            slip=abs(qty) * abs(vwap - best),
            comision=comision,
            part=part,
        )
