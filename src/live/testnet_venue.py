"""Venue de testnet: implementa ``ExecutionVenue`` contra Binance spot testnet.

Valida **mecanica** (envio, idempotencia, reconciliacion), nunca costos: el libro
de testnet es artificial (ADR 0007), asi que ``spread_cost`` y ``slippage_cost``
valen 0.0 y ninguna cifra de aca entra a un reporte de costos.

Binance **no** rechaza un ``newClientOrderId`` repetido sobre una orden de mercado
ya llena: la ejecuta otra vez (``order_buy.json`` y ``order_buy_duplicado.json``).
La idempotencia es entonces nuestra:

1. un id ya visto devuelve el acuse previo y no toca la red;
2. antes del POST se guarda un ``intent`` con append + fsync (write-ahead);
3. ante un timeout se consulta por ``origClientOrderId`` antes de reenviar.

Fail-closed: cualquier ``BinanceApiError`` no cubierto por una grabacion se
propaga y detiene el proceso. No hay metodo publico fuera del protocolo: el fill
de una orden de mercado viaja en la respuesta del POST.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from data.instruments import InstrumentSpec
from live.fillstore import FillStore, day_of, fill_from_dict, fill_to_dict
from live.testnet_http import BinanceApiError, BinanceTestnetClient, TransportError
from sim.clock import Clock
from sim.orders import Fill, MarketOrder, OrderStatus, RejectReason
from sim.venue import CancelAck, OrderAck, VenueState

# Regla de Binance para newClientOrderId (err_id_invalido.json).
_ID_VALIDO = re.compile(r"^[a-zA-Z0-9\-_]{1,36}$")
_ORDEN_INEXISTENTE = -2013
_CANCELAR_INEXISTENTE = -2011


class TestnetVenueError(RuntimeError):
    """Estado imposible o respuesta que no sabemos interpretar: se detiene."""

    __test__ = False  # no es una clase de tests para pytest


@dataclass(frozen=True)
class _Intent:
    order: MarketOrder
    qty: float  # redondeada al lote, es lo que viaja
    timestamp: np.datetime64
    seq: int


def _ms(valor: Any) -> np.datetime64:
    return np.datetime64(int(valor), "ms") + np.timedelta64(0, "ns")


class BinanceTestnetVenue:
    def __init__(
        self,
        *,
        client: BinanceTestnetClient,
        instrument: InstrumentSpec,
        base_asset: str,
        quote_asset: str,
        clock: Clock,
        store: FillStore,
    ) -> None:
        self._client = client
        self._instrument = instrument
        self._base = base_asset
        self._quote = quote_asset
        self._clock = clock
        self._store = store
        self._acks: dict[str, OrderAck] = {}
        self._unresolved: dict[str, _Intent] = {}
        self._fills: list[Fill] = []
        self._seq = 0
        self._replay()

    def _replay(self) -> None:
        for r in self._store.replay():
            if r["type"] == "intent":
                id_ = r["client_order_id"]
                self._acks[id_] = OrderAck(client_order_id=id_, accepted=True)
                self._seq = max(self._seq, r["seq"])
                self._unresolved[id_] = _Intent(
                    MarketOrder(r["qty_requested"], r["tag"], id_),
                    r["qty"],
                    np.datetime64(r["timestamp"], "ns"),
                    r["seq"],
                )
            elif r["type"] == "fill":
                fill = fill_from_dict(r["fill"])
                self._fills.append(fill)
                self._unresolved.pop(fill.client_order_id, None)
            elif r["type"] == "error":
                self._unresolved.pop(r["client_order_id"], None)

    # -- ExecutionVenue -------------------------------------------------

    def submit_order(self, order: MarketOrder) -> OrderAck:
        """Acusa recepcion. El fill queda en el log de ``get_fills``."""
        if not order.client_order_id:
            return OrderAck(
                client_order_id="",
                accepted=False,
                reject_reason=RejectReason.MISSING_CLIENT_ORDER_ID,
                detail="el identificador lo genera el emisor, no el venue",
            )
        if not _ID_VALIDO.match(order.client_order_id):
            return OrderAck(
                client_order_id=order.client_order_id,
                accepted=False,
                reject_reason=RejectReason.INVALID_CLIENT_ORDER_ID,
                detail="el identificador debe cumplir ^[a-zA-Z0-9-_]{1,36}$",
            )
        previo = self._acks.get(order.client_order_id)
        if previo is not None:
            return replace(previo, is_duplicate=True)  # sin tocar la red
        ahora = self._clock.now()
        intent = _Intent(
            order, self._instrument.round_qty(order.qty), ahora, self._seq + 1
        )
        self._store.append(
            day_of(ahora),
            {
                "type": "intent",
                "client_order_id": order.client_order_id,
                "qty_requested": order.qty,
                "qty": intent.qty,
                "tag": order.tag,
                "timestamp": str(ahora),
                "seq": intent.seq,
            },
        )
        self._seq = intent.seq
        ack = OrderAck(client_order_id=order.client_order_id, accepted=True)
        self._acks[ack.client_order_id] = ack
        self._unresolved[ack.client_order_id] = intent
        if intent.qty == 0.0:  # como el simulador: se registra, no se envia
            self._record(self._rejected(intent, RejectReason.ZERO_AFTER_ROUNDING))
        else:
            self._send(intent)
        return ack

    def cancel_order(self, client_order_id: str) -> CancelAck:
        try:
            self._client.call(
                "DELETE",
                "/api/v3/order",
                {
                    "symbol": self._instrument.symbol,
                    "origClientOrderId": client_order_id,
                },
            )
        except BinanceApiError as exc:
            if exc.code == _CANCELAR_INEXISTENTE:
                return CancelAck(client_order_id, False, "no hay una orden viva")
            raise
        return CancelAck(client_order_id=client_order_id, cancelled=True)

    def get_positions(self) -> dict[str, float]:
        qty = self._balance(self._account(), self._base)
        return {self._instrument.symbol: qty} if qty != 0.0 else {}

    def get_fills(self, since: int = 0) -> list[Fill]:
        if since < 0:
            raise ValueError("since no puede ser negativo")
        return list(self._fills[since:])

    def reconcile(self) -> VenueState:
        """Resuelve intenciones sin resultado y devuelve la foto del venue.

        Una intencion que el venue no conoce (``-2013``) **no se reenvia**: una
        orden de mercado vieja ejecutaria a un precio que nadie decidio. Se
        registra como no entregada.
        """
        for intent in list(self._unresolved.values()):
            if not self._query(intent):
                self._record(self._rejected(intent, RejectReason.VENUE_NOT_RUNNING))
        cuenta = self._account()
        abiertas = self._client.call(
            "GET", "/api/v3/openOrders", {"symbol": self._instrument.symbol}
        )
        qty = self._balance(cuenta, self._base)
        return VenueState(
            timestamp=self._clock.now(),
            cash=self._balance(cuenta, self._quote),
            positions={self._instrument.symbol: qty} if qty != 0.0 else {},
            open_order_ids=tuple(o["clientOrderId"] for o in abiertas),
        )

    # -- envio y recuperacion ------------------------------------------

    def _params(self, intent: _Intent) -> dict[str, str | int]:
        return {
            "symbol": self._instrument.symbol,
            "side": "BUY" if intent.qty > 0 else "SELL",
            "type": "MARKET",
            "quantity": f"{abs(intent.qty):.{self._instrument.qty_precision}f}",
            "newClientOrderId": intent.order.client_order_id,
            "newOrderRespType": "FULL",
        }

    def _send(self, intent: _Intent) -> None:
        try:
            self._post(intent)
        except TransportError:
            # Estado desconocido: consultar antes de reenviar, nunca a ciegas.
            if not self._query(intent):
                self._post(intent)  # -2013: no salio; un unico reenvio

    def _post(self, intent: _Intent) -> None:
        try:
            respuesta = self._client.call("POST", "/api/v3/order", self._params(intent))
        except BinanceApiError as exc:
            # El venue respondio con error: no hay orden. Se deja constancia para
            # que reconcile no la reenvie, y se detiene (fail-closed).
            self._store.append(
                day_of(self._clock.now()),
                {
                    "type": "error",
                    "client_order_id": intent.order.client_order_id,
                    "code": exc.code,
                    "msg": exc.msg,
                },
            )
            self._unresolved.pop(intent.order.client_order_id, None)
            raise
        self._record(self._fill_from(intent, respuesta))

    def _query(self, intent: _Intent) -> bool:
        """Adopta el estado de la orden en el venue. ``False`` si no existe."""
        try:
            respuesta = self._client.call(
                "GET",
                "/api/v3/order",
                {
                    "symbol": self._instrument.symbol,
                    "origClientOrderId": intent.order.client_order_id,
                },
            )
        except BinanceApiError as exc:
            if exc.code == _ORDEN_INEXISTENTE:
                return False
            raise
        self._record(self._fill_from(intent, respuesta))
        return True

    def _record(self, fill: Fill) -> None:
        self._store.append(
            day_of(fill.timestamp_fill), {"type": "fill", "fill": fill_to_dict(fill)}
        )
        self._fills.append(fill)
        self._unresolved.pop(fill.client_order_id, None)

    # -- mapeo a Fill ---------------------------------------------------

    def _rejected(self, intent: _Intent, motivo: RejectReason) -> Fill:
        return self._build(
            intent, OrderStatus.REJECTED, 0.0, 0.0, 0.0, self._clock.now(), motivo
        )

    def _build(
        self,
        intent: _Intent,
        status: OrderStatus,
        qty: float,
        price: float,
        commission: float,
        t_fill: np.datetime64,
        motivo: RejectReason | None = None,
    ) -> Fill:
        # ASSUMPTION: sin close[t] en el protocolo, decision_price == ref_price
        # == precio del fill y gap == 0. Spread y slippage no se miden en testnet.
        return Fill(
            t_decision=intent.seq,
            t_fill=intent.seq,
            timestamp_decision=intent.timestamp,
            timestamp_fill=t_fill,
            symbol=self._instrument.symbol,
            status=status,
            qty_requested=intent.order.qty,
            qty_filled=qty,
            decision_price=price,
            ref_price=price,
            fill_price=price,
            gap=0.0,
            spread_cost=0.0,
            slippage_cost=0.0,
            commission=commission,
            participation=0.0,
            reject_reason=motivo,
            tag=intent.order.tag,
            client_order_id=intent.order.client_order_id,
        )

    def _fill_from(self, intent: _Intent, r: Mapping[str, Any]) -> Fill:
        ejecutado = float(r["executedQty"])
        estado = r["status"]
        t_fill = _ms(r.get("transactTime", r.get("updateTime")))
        if ejecutado == 0.0:
            if estado in ("EXPIRED", "EXPIRED_IN_MATCH", "CANCELED"):
                return self._build(
                    intent, OrderStatus.REJECTED, 0.0, 0.0, 0.0, t_fill,
                    RejectReason.NO_VOLUME,
                )  # fmt: skip
            raise TestnetVenueError(f"orden sin ejecutar en estado {estado!r}")
        if estado == "FILLED":
            status = OrderStatus.FILLED
        elif estado in ("EXPIRED", "EXPIRED_IN_MATCH", "CANCELED"):
            status = OrderStatus.PARTIAL  # el remanente se cancela, no se arrastra
        else:
            raise TestnetVenueError(f"estado {estado!r} no interpretable")
        precio = float(r["cummulativeQuoteQty"]) / ejecutado
        signo = 1.0 if r["side"] == "BUY" else -1.0
        # ASSUMPTION: una orden recuperada por consulta no trae fills[] y pierde
        # la comision (0 en testnet); myTrades la traeria y queda para el PR 8.
        comision = sum(self._commission(f, precio) for f in r.get("fills", []))
        return self._build(intent, status, signo * ejecutado, precio, comision, t_fill)

    def _commission(self, fill: Mapping[str, Any], precio: float) -> float:
        monto = float(fill["commission"])
        if fill["commissionAsset"] == self._quote:
            return monto
        if fill["commissionAsset"] == self._base:
            return monto * precio
        raise TestnetVenueError(f"comision en {fill['commissionAsset']!r}")

    # -- cuenta ---------------------------------------------------------

    def _account(self) -> Mapping[str, Any]:
        cuenta = self._client.call("GET", "/api/v3/account")
        assert isinstance(cuenta, dict)
        return cuenta

    @staticmethod
    def _balance(cuenta: Mapping[str, Any], asset: str) -> float:
        for b in cuenta["balances"]:
            if b["asset"] == asset:
                return float(b["free"]) + float(b["locked"])
        return 0.0
