"""Observador de fills por el WebSocket de testnet. **Nunca envia ordenes.**

Solo suscribe (``userDataStream.subscribe.signature``) y lee. No recibe el cliente
con el que se envian ordenes ni conoce ``submit_order``: no hay camino de reenvio
a ciegas (Binance ejecuta otra vez un ``newClientOrderId`` repetido).

El contrato sale de las grabaciones reales del PR 8a
(``tests/live/respuestas/testnet_ws``):

* un fill ocurrido con el socket caido **no se repite** al reconectar, asi que
  despues de cada reconexion se reconcilia por REST (``myTrades``) antes de
  confiar en el stream;
* los trades se deduplican por ``trade_id``: el REST y el stream se solapan;
* ``eventStreamTerminated`` sin haber pedido la baja es "stream muerto";
* los eventos llegan con retraso y en otro orden que el REST, asi que nada aca
  asume que la cuenta ya se actualizo cuando se ve un fill.

Cada trade se guarda con append + fsync **antes** de entregarse a quien llama.
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from live.fillstore import FillStore, day_of
from live.guard import assert_testnet_ws_url
from live.testnet_http import (
    BinanceTestnetClient,
    BinanceTestnetCredentials,
    sign,
)
from sim.clock import Clock

log = logging.getLogger(__name__)

WS_URL = "wss://ws-api.testnet.binance.vision/ws-api/v3"
SEGUNDOS_SUSCRIPCION = 10.0  # espera maxima por la respuesta a la suscripcion
BACKOFF_INICIAL_S = 1.0
BACKOFF_MAXIMO_S = 60.0


class StreamError(RuntimeError):
    """Binance rechazo la suscripcion: reintentar no lo arregla, se detiene."""

    __test__ = False


class Socket(Protocol):
    def send(self, texto: str) -> None: ...

    def recv(self, timeout: float) -> str | None:
        """Un mensaje, ``None`` si no llego ninguno; ``ConnectionError`` si cayo."""
        ...

    def close(self) -> None: ...


Connector = Callable[[str], Socket]


def connect_websockets(url: str) -> Socket:  # pragma: no cover - red
    """Conector real; ``websockets`` es un grupo opcional (``--group ws``)."""
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.client import connect

    class _Real:
        def __init__(self) -> None:
            self._ws = connect(assert_testnet_ws_url(url), open_timeout=10)

        def send(self, texto: str) -> None:
            try:
                self._ws.send(texto)
            except ConnectionClosed as exc:
                raise ConnectionError(str(exc)) from exc

        def recv(self, timeout: float) -> str | None:
            try:
                dato = self._ws.recv(timeout=timeout)
            except TimeoutError:
                return None
            except ConnectionClosed as exc:
                raise ConnectionError(str(exc)) from exc
            return dato if isinstance(dato, str) else dato.decode()

        def close(self) -> None:
            self._ws.close()

    return _Real()


@dataclass(frozen=True)
class TradeReport:
    """Un trade ejecutado. ``client_order_id`` es ``None`` si vino por REST."""

    trade_id: int
    order_id: int
    client_order_id: str | None
    symbol: str
    side: str
    qty: float
    price: float
    commission: float
    commission_asset: str
    time: np.datetime64
    source: str  # "ws" | "rest"


def _ms(valor: Any) -> np.datetime64:
    return np.datetime64(int(valor), "ms") + np.timedelta64(0, "ns")


def report_from_event(event: Mapping[str, Any]) -> TradeReport | None:
    """``executionReport`` con un trade. Los ``NEW`` y demas no son fills."""
    if event.get("e") != "executionReport" or event.get("x") != "TRADE":
        return None
    return TradeReport(
        trade_id=int(event["t"]),
        order_id=int(event["i"]),
        client_order_id=str(event["c"]),
        symbol=str(event["s"]),
        side=str(event["S"]),
        qty=float(event["l"]),
        price=float(event["L"]),
        commission=float(event["n"]),
        commission_asset=str(event["N"] or ""),
        time=_ms(event["T"]),
        source="ws",
    )


def report_from_rest(trade: Mapping[str, Any]) -> TradeReport:
    return TradeReport(
        trade_id=int(trade["id"]),
        order_id=int(trade["orderId"]),
        client_order_id=None,
        symbol=str(trade["symbol"]),
        side="BUY" if trade["isBuyer"] else "SELL",
        qty=float(trade["qty"]),
        price=float(trade["price"]),
        commission=float(trade["commission"]),
        commission_asset=str(trade["commissionAsset"]),
        time=_ms(trade["time"]),
        source="rest",
    )


def _a_dict(r: TradeReport) -> dict[str, Any]:
    return {**r.__dict__, "time": str(r.time)}


def _de_dict(d: Mapping[str, Any]) -> TradeReport:
    return TradeReport(**{**d, "time": np.datetime64(d["time"], "ns")})


def subscribe_request(
    credentials: BinanceTestnetCredentials, timestamp_ms: int, id_: str
) -> str:
    """Pedido firmado: HMAC sobre los params ordenados por clave (como el REST)."""
    params: dict[str, str | int] = {
        "apiKey": credentials.api_key,
        "timestamp": timestamp_ms,
    }
    cadena = "&".join(f"{k}={params[k]}" for k in sorted(params))
    params["signature"] = sign(credentials.api_secret, cadena)
    return json.dumps(
        {"id": id_, "method": "userDataStream.subscribe.signature", "params": params}
    )


class TestnetFillStream:
    __test__ = False

    def __init__(
        self,
        *,
        connect: Connector,
        client: BinanceTestnetClient,
        credentials: BinanceTestnetCredentials,
        clock: Clock,
        store: FillStore,
        symbol: str,
        url: str = WS_URL,
    ) -> None:
        self._url = assert_testnet_ws_url(url)
        self._connect = connect
        self._client = client
        self._credentials = credentials
        self._clock = clock
        self._store = store
        self._symbol = symbol
        self._sock: Socket | None = None
        self._cerrado = False
        self._backoff = BACKOFF_INICIAL_S
        self._retry_at = clock.now()
        self._seq = 0
        self._vistos: dict[int, TradeReport] = {}
        for r in store.replay():
            if r["type"] == "ws_trade":
                t = _de_dict(r["trade"])
                self._vistos[t.trade_id] = t

    @property
    def last_trade_id(self) -> int | None:
        return max(self._vistos) if self._vistos else None

    def poll(self) -> list[TradeReport]:
        """Trades nuevos desde la ultima llamada. No bloquea; reconecta solo.

        Con el socket caido devuelve ``[]`` hasta que pase el backoff. Un rechazo
        de la suscripcion (``StreamError``) se propaga: no se reintenta a ciegas.
        """
        nuevos: list[TradeReport] = []
        if self._cerrado:
            return nuevos
        if self._sock is None and not self._reabrir(nuevos):
            return nuevos
        assert self._sock is not None
        try:
            while (texto := self._sock.recv(0.0)) is not None:
                if not self._manejar(texto, nuevos):
                    self._caer("eventStreamTerminated inesperado")
                    break
        except ConnectionError as exc:
            self._caer(str(exc))
        return nuevos

    def close(self) -> None:
        """Cierre pedido por nosotros: no reconecta."""
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        self._cerrado = True

    # -- conexion -------------------------------------------------------

    def _caer(self, motivo: str) -> None:
        log.warning("stream de testnet caido", extra={"motivo": motivo})
        if self._sock is not None:
            with contextlib.suppress(OSError):
                self._sock.close()
        self._sock = None
        self._retry_at = self._clock.now() + np.timedelta64(
            int(self._backoff * 1000), "ms"
        )
        self._backoff = min(self._backoff * 2, BACKOFF_MAXIMO_S)

    def _reabrir(self, nuevos: list[TradeReport]) -> bool:
        if self._clock.now() < self._retry_at:
            return False
        try:
            sock = self._connect(self._url)
            self._sock = sock
            self._suscribir(sock)
        except ConnectionError as exc:
            self._caer(str(exc))
            return False
        self._backoff = BACKOFF_INICIAL_S
        # El venue no repite lo perdido: se reconcilia por REST **despues** de
        # suscribir (sin hueco) y deduplicando por trade_id (con solape).
        self._recuperar(nuevos)
        return True

    def _suscribir(self, sock: Socket) -> None:
        self._seq += 1
        id_ = f"sub-{self._seq}"
        ts = int(self._clock.now().astype("datetime64[ms]").astype(np.int64))
        sock.send(subscribe_request(self._credentials, ts, id_))
        respuesta = sock.recv(SEGUNDOS_SUSCRIPCION)
        if respuesta is None:
            raise ConnectionError("sin respuesta a la suscripcion")
        datos = json.loads(respuesta)
        if datos.get("id") != id_ or datos.get("status") != 200:
            error = datos.get("error", {})
            raise StreamError(
                f"suscripcion rechazada: {error.get('code')} {error.get('msg')}"
            )

    def _recuperar(self, nuevos: list[TradeReport]) -> None:
        params: dict[str, str | int] = {"symbol": self._symbol}
        if self.last_trade_id is not None:
            params["fromId"] = self.last_trade_id + 1
        for trade in self._client.call("GET", "/api/v3/myTrades", params):
            self._aceptar(report_from_rest(trade), nuevos)

    # -- eventos --------------------------------------------------------

    def _manejar(self, texto: str, nuevos: list[TradeReport]) -> bool:
        """``False`` si el stream murio."""
        frame = json.loads(texto)
        if frame.get("status", 200) != 200:
            error = frame.get("error", {})
            raise StreamError(f"error del stream: {error.get('code')} {error}")
        event = frame.get("event", {})
        if event.get("e") == "eventStreamTerminated":
            return False
        reporte = report_from_event(event)
        if reporte is not None:
            self._aceptar(reporte, nuevos)
        return True

    def _aceptar(self, reporte: TradeReport, nuevos: list[TradeReport]) -> None:
        previo = self._vistos.get(reporte.trade_id)
        if previo is not None and (
            previo.client_order_id is not None or reporte.client_order_id is None
        ):
            return  # ya visto; el del stream conserva el client_order_id
        self._store.append(
            day_of(self._clock.now()), {"type": "ws_trade", "trade": _a_dict(reporte)}
        )
        self._vistos[reporte.trade_id] = reporte
        if previo is None:
            nuevos.append(reporte)
