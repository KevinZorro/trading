"""Observador de fills contra los frames reales grabados en el PR 8a.

Ningun test inventa un frame: la respuesta a la suscripcion se toma de la
grabacion y solo se le reescribe el ``id`` (lo genera el emisor), y los casos
raros se derivan de un frame real cambiando campos.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from live.fillstore import FillStore, venue_header
from live.guard import MainnetOrderError
from live.testnet_http import (
    BinanceTestnetClient,
    BinanceTestnetCredentials,
    HttpRequest,
    HttpResponse,
    sign,
)
from live.testnet_ws import (
    StreamError,
    TestnetFillStream,
    TradeReport,
    report_from_event,
)
from sim.clock import SimulatedClock

DIR = Path(__file__).parent / "respuestas" / "testnet_ws"
REST = Path(__file__).parent / "respuestas" / "testnet"
MANIFEST = json.loads((DIR / "manifest.json").read_text())
AHORA = np.datetime64("2026-10-19T05:07:00.000", "ns")
CRED = BinanceTestnetCredentials("K" * 64, "S" * 64)


def frames(nombre: str) -> list[dict[str, Any]]:
    cuerpo = (DIR / nombre).read_bytes()
    assert hashlib.sha256(cuerpo).hexdigest() == MANIFEST["files"][nombre]["sha256"]
    return [json.loads(linea)["frame"] for linea in cuerpo.decode().splitlines()]


FASE_A = frames("fase_a_fill_normal.jsonl")
FASE_C = frames("fase_c_firma_invalida_y_baja.jsonl")
RESPUESTA_OK = FASE_A[0]
EVENTOS_A = [f for f in FASE_A if "event" in f]
TRADES_A = [f for f in EVENTOS_A if f["event"].get("x") == "TRADE"]
MIS_TRADES = json.loads((REST / "my_trades.json").read_bytes())[:1]


class SocketFalso:
    """Socket de guion: ``items`` se sirven en orden; ``None`` = silencio."""

    def __init__(
        self, items: list[Any], respuesta: dict[str, Any] | None = RESPUESTA_OK
    ) -> None:
        self.items: deque[Any] = deque(items)
        self.respuesta = respuesta
        self.enviados: list[dict[str, Any]] = []
        self.cerrado = False

    def send(self, texto: str) -> None:
        self.enviados.append(json.loads(texto))

    def recv(self, timeout: float) -> str | None:
        if self.respuesta is not None and self.enviados:
            r, self.respuesta = {**self.respuesta, "id": self.enviados[-1]["id"]}, None
            return json.dumps(r)
        if not self.items:
            return None
        item = self.items.popleft()
        if isinstance(item, Exception):
            raise item
        return None if item is None else json.dumps(item)

    def close(self) -> None:
        self.cerrado = True


class Conector:
    def __init__(self, *sockets: SocketFalso | Exception) -> None:
        self.cola = deque(sockets)
        self.urls: list[str] = []

    def __call__(self, url: str) -> SocketFalso:
        self.urls.append(url)
        s = self.cola.popleft()
        if isinstance(s, Exception):
            raise s
        return s


class Rest:
    def __init__(self, *respuestas: Any) -> None:
        self.respuestas = deque(respuestas)
        self.pedidos: list[HttpRequest] = []

    def __call__(self, pedido: HttpRequest) -> HttpResponse:
        self.pedidos.append(pedido)
        return HttpResponse(200, json.dumps(self.respuestas.popleft()).encode())


def stream(
    tmp_path: Path, conector: Conector, rest: Rest, clock: SimulatedClock | None = None
) -> TestnetFillStream:
    clock = clock or SimulatedClock(AHORA)
    return TestnetFillStream(
        connect=conector,
        client=BinanceTestnetClient(CRED, clock, transport=rest),
        credentials=CRED,
        clock=clock,
        store=FillStore(
            tmp_path,
            venue_header(initial_cash=0.0, spec_sha256="x", spec_received_at=AHORA),
        ),
        symbol="BTCUSDT",
    )


def test_un_fill_real_se_parsea_con_su_comision() -> None:
    (r,) = [report_from_event(f["event"]) for f in TRADES_A[:1]]
    assert r == TradeReport(
        trade_id=867780,
        order_id=4541299,
        client_order_id="rec8-a-buy-1792386406",
        symbol="BTCUSDT",
        side="BUY",
        qty=0.001,
        price=82608.0,
        commission=0.0,
        commission_asset="BTC",
        time=np.datetime64(1792386409412, "ms") + np.timedelta64(0, "ns"),
        source="ws",
    )


def test_la_comision_real_se_lee_del_campo_n() -> None:
    evento = copy.deepcopy(TRADES_A[0]["event"])
    evento.update({"n": "0.00000100", "N": "BTC"})
    r = report_from_event(evento)
    assert r is not None
    assert (r.commission, r.commission_asset) == (0.000001, "BTC")


def test_solo_los_trades_son_fills(tmp_path: Path) -> None:
    s = SocketFalso(list(FASE_A[1:]))  # NEW, TRADE, posicion de cuenta, ...
    rest = Rest([])
    obs = stream(tmp_path, Conector(s), rest)
    nuevos = obs.poll()
    assert [r.trade_id for r in nuevos] == [
        f["event"]["t"] for f in TRADES_A
    ]  # los NEW (t=-1) y outboundAccountPosition no cuentan


def test_la_suscripcion_va_firmada_y_sin_el_secreto(tmp_path: Path) -> None:
    s = SocketFalso([])
    stream(tmp_path, Conector(s), Rest([])).poll()
    (pedido,) = s.enviados
    p = pedido["params"]
    assert pedido["method"] == "userDataStream.subscribe.signature"
    assert p["signature"] == sign(
        CRED.api_secret, f"apiKey={CRED.api_key}&timestamp={p['timestamp']}"
    )
    assert CRED.api_secret not in json.dumps(pedido)


def test_nunca_envia_ordenes(tmp_path: Path) -> None:
    rest = Rest([])
    obs = stream(tmp_path, Conector(SocketFalso(list(FASE_A[1:]))), rest)
    obs.poll()
    assert not hasattr(obs, "submit_order")
    assert {
        (p.method, p.url.split("?")[0].rsplit("/", 1)[1]) for p in rest.pedidos
    } == {("GET", "myTrades")}


def test_primer_arranque_recupera_por_rest_sin_fromid(tmp_path: Path) -> None:
    rest = Rest(MIS_TRADES)
    (r,) = stream(tmp_path, Conector(SocketFalso([])), rest).poll()
    assert "fromId" not in rest.pedidos[0].url
    assert (r.trade_id, r.source, r.client_order_id, r.side) == (
        863272,
        "rest",
        None,
        "BUY",
    )


def test_tras_reconectar_reconcilia_desde_el_ultimo_trade(tmp_path: Path) -> None:
    clock = SimulatedClock(AHORA)
    caida = ConnectionError("cierre")
    rest = Rest([], [])
    conector = Conector(SocketFalso([*list(FASE_A[1:3]), caida]), SocketFalso([]))
    obs = stream(tmp_path, conector, rest, clock)
    assert len(obs.poll()) == 1  # el TRADE de compra; luego cae el socket
    clock.advance_to(AHORA + np.timedelta64(2, "s"))
    assert obs.poll() == []
    assert len(conector.urls) == 2
    assert f"fromId={TRADES_A[0]['event']['t'] + 1}" in rest.pedidos[1].url


def test_un_trade_visto_dos_veces_se_entrega_una(tmp_path: Path) -> None:
    clock = SimulatedClock(AHORA)
    solapado = copy.deepcopy(MIS_TRADES)
    solapado[0]["id"] = TRADES_A[0]["event"]["t"]  # el REST repite el del stream
    conector = Conector(
        SocketFalso([TRADES_A[0], ConnectionError("x")]), SocketFalso([TRADES_A[0]])
    )
    obs = stream(tmp_path, conector, Rest([], solapado), clock)
    assert len(obs.poll()) == 1
    clock.advance_to(AHORA + np.timedelta64(2, "s"))
    assert obs.poll() == []


def test_el_trade_del_stream_conserva_el_client_order_id_del_rest(
    tmp_path: Path,
) -> None:
    mismo = copy.deepcopy(MIS_TRADES)
    mismo[0]["id"] = TRADES_A[0]["event"]["t"]
    obs = stream(tmp_path, Conector(SocketFalso([TRADES_A[0]])), Rest(mismo))
    (r,) = obs.poll()  # el REST llega primero; el stream lo enriquece sin repetirlo
    assert r.client_order_id is None
    reinicio = stream(tmp_path, Conector(), Rest())
    assert reinicio.last_trade_id == TRADES_A[0]["event"]["t"]


def test_cada_trade_queda_en_disco_y_sobrevive_a_un_reinicio(tmp_path: Path) -> None:
    obs = stream(tmp_path, Conector(SocketFalso(list(TRADES_A))), Rest([]))
    entregados = obs.poll()
    reinicio = stream(tmp_path, Conector(SocketFalso([])), Rest([]))
    assert reinicio.last_trade_id == max(r.trade_id for r in entregados)
    assert reinicio.poll() == []  # los viejos no se vuelven a entregar


def test_event_stream_terminated_reconecta_con_backoff(tmp_path: Path) -> None:
    terminado = next(
        f for f in FASE_C if f.get("event", {}).get("e") == "eventStreamTerminated"
    )
    clock = SimulatedClock(AHORA)
    s1 = SocketFalso([terminado])
    conector = Conector(s1, ConnectionError("x"), SocketFalso([]))
    obs = stream(tmp_path, conector, Rest([], []), clock)
    assert obs.poll() == []
    assert s1.cerrado
    assert obs.poll() == []
    assert len(conector.urls) == 1  # dentro del backoff
    clock.advance_to(AHORA + np.timedelta64(1, "s"))
    obs.poll()
    assert len(conector.urls) == 2  # reintento fallido: el backoff pasa a 2 s
    clock.advance_to(AHORA + np.timedelta64(2, "s"))
    obs.poll()
    assert len(conector.urls) == 2
    clock.advance_to(AHORA + np.timedelta64(3, "s"))
    obs.poll()
    assert len(conector.urls) == 3


def test_un_fallo_de_conexion_reintenta_sin_lanzar(tmp_path: Path) -> None:
    clock = SimulatedClock(AHORA)
    conector = Conector(ConnectionError("dns"), SocketFalso([]))
    obs = stream(tmp_path, conector, Rest([]), clock)
    assert obs.poll() == []
    clock.advance_to(AHORA + np.timedelta64(1, "s"))
    obs.poll()
    assert len(conector.urls) == 2


def test_suscripcion_rechazada_detiene_en_vez_de_reintentar(tmp_path: Path) -> None:
    rechazo = FASE_C[0]
    obs = stream(tmp_path, Conector(SocketFalso([], respuesta=rechazo)), Rest())
    with pytest.raises(StreamError, match="-1022"):
        obs.poll()


def test_close_no_reconecta(tmp_path: Path) -> None:
    s = SocketFalso([])
    conector = Conector(s)
    obs = stream(tmp_path, conector, Rest([]))
    obs.poll()
    obs.close()
    assert s.cerrado
    assert obs.poll() == []
    assert len(conector.urls) == 1


def test_mainnet_se_rechaza_en_la_construccion(tmp_path: Path) -> None:
    clock = SimulatedClock(AHORA)
    with pytest.raises(MainnetOrderError, match="no es de testnet"):
        TestnetFillStream(
            connect=Conector(),
            client=BinanceTestnetClient(CRED, clock, transport=Rest()),
            credentials=CRED,
            clock=clock,
            store=FillStore(
                tmp_path,
                venue_header(initial_cash=0.0, spec_sha256="x", spec_received_at=AHORA),
            ),
            symbol="BTCUSDT",
            url="wss://stream.binance.com/ws-api/v3",
        )
