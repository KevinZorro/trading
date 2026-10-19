"""El grabador del WebSocket de testnet, sin red.

Los dobles solo ejercitan el cableado del script (orden de las fases, firma,
abortos, que no se escriba un secreto). **No son oraculos de Binance**: la forma
real de los mensajes vive en ``respuestas/testnet_ws`` cuando Kevin las graba, y los
tests del observador (PR 8b) deben leer de ahi, no de aca.
"""

from __future__ import annotations

import importlib.util
import json
import urllib.parse
from collections import deque
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from live.testnet_http import (
    BinanceTestnetClient,
    BinanceTestnetCredentials,
    HttpRequest,
    HttpResponse,
    sign,
)

RAIZ = Path(__file__).resolve().parents[2]
CLAVE, SECRETO = "K" * 64, "S" * 64


def _cargar() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "grabar_ws_testnet", RAIZ / "scripts" / "grabar_ws_testnet.py"
    )
    assert spec is not None
    assert spec.loader is not None
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_firma_de_la_suscripcion_calculada_a_mano() -> None:
    # Oraculo externo: printf 'apiKey=KEY&timestamp=1700000000000' |
    #   openssl dgst -sha256 -hmac SECRET
    # Los params se ordenan por clave (apiKey < timestamp) antes de firmar.
    pedido = json.loads(
        _cargar().pedido_firmado("m.x", "KEY", "SECRET", 1700000000000, "abc")
    )
    assert pedido == {
        "id": "abc",
        "method": "m.x",
        "params": {
            "apiKey": "KEY",
            "timestamp": 1700000000000,
            "signature": (
                "1d36c5f410603c229417e6072d7cee1fa5b65e344efa3e78d414badfb7bca3a7"
            ),
        },
    }


@pytest.mark.parametrize(
    ("frame", "esperado"),
    [
        ({"e": "executionReport", "c": "x", "X": "FILLED"}, True),
        ({"event": {"e": "executionReport", "c": "x", "X": "FILLED"}}, True),
        ({"event": {"e": "executionReport", "c": "x", "X": "NEW"}}, False),
        ({"event": {"e": "executionReport", "c": "otro", "X": "FILLED"}}, False),
        ({"event": {"e": "balanceUpdate", "c": "x", "X": "FILLED"}}, False),
        ({"id": "x", "status": 200}, False),
    ],
)
def test_es_fill_de(frame: dict[str, object], esperado: bool) -> None:
    assert _cargar().es_fill_de(frame, "x") is esperado


class Entorno:
    """REST y sockets falsos con un registro comun del orden de los hechos."""

    def __init__(self, estado_suscripcion: int = 200, filtrar: str = "") -> None:
        self.hechos: list[str] = []
        self.estado = estado_suscripcion
        self.filtrar = filtrar  # texto a colar en un evento (prueba de secretos)
        self.activo: FakeSocket | None = None
        self.sockets: list[FakeSocket] = []
        self.pedidos: list[dict[str, list[str]]] = []

    def transporte(self, pedido: HttpRequest) -> HttpResponse:
        ruta = urllib.parse.urlsplit(pedido.url).path
        if ruta == "/api/v3/time":
            return HttpResponse(200, b'{"serverTime": 5000}')
        if ruta == "/api/v3/account":
            return HttpResponse(200, b'{"balances": []}')
        orden = urllib.parse.parse_qs(pedido.body.decode())
        self.pedidos.append(orden)
        cid = orden["newClientOrderId"][0]
        self.hechos.append(f"rest:{orden['side'][0]}:{cid}")
        if self.activo is not None:
            evento = {
                "e": "executionReport",
                "c": cid,
                "X": "FILLED",
                "x": self.filtrar,
            }
            self.activo.cola.append(json.dumps({"subscriptionId": 0, "event": evento}))
        return HttpResponse(200, b"{}")

    def conectar(self, url: str) -> FakeSocket:
        assert url == "wss://ws-api.testnet.binance.vision/ws-api/v3"
        sock = FakeSocket(self, len(self.sockets))
        self.sockets.append(sock)
        self.hechos.append(f"ws{sock.n}:conectar")
        return sock


class FakeSocket:
    def __init__(self, entorno: Entorno, n: int) -> None:
        self.entorno, self.n = entorno, n
        self.cola: deque[str] = deque()
        self.cerrado = False
        self.enviados: list[dict[str, object]] = []

    def send(self, texto: str) -> None:
        pedido = json.loads(texto)
        self.enviados.append(pedido)
        metodo = pedido["method"]
        self.entorno.hechos.append(f"ws{self.n}:{metodo}")
        if metodo.startswith("userDataStream.subscribe"):
            par = pedido["params"]
            valida = sign(
                SECRETO, f"apiKey={par['apiKey']}&timestamp={par['timestamp']}"
            )
            firma_mala = par["signature"] != valida
            if firma_mala:
                respuesta = {
                    "id": pedido["id"],
                    "status": 400,
                    "error": {"code": -1022},
                }
            else:
                respuesta = {"id": pedido["id"], "status": self.entorno.estado}
                if self.entorno.estado == 200:
                    self.entorno.activo = self
            self.cola.append(json.dumps(respuesta))
        else:
            self.cola.append(json.dumps({"id": pedido["id"], "status": 200}))

    def recv(self, timeout: float) -> str | None:
        return self.cola.popleft() if self.cola else None

    def close(self) -> None:
        self.cerrado = True
        self.entorno.hechos.append(f"ws{self.n}:cerrar")
        if self.entorno.activo is self:
            self.entorno.activo = None


@pytest.fixture
def grabador(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[..., tuple[ModuleType, Entorno, Path]]:
    def armar(**kwargs: object) -> tuple[ModuleType, Entorno, Path]:
        modulo = _cargar()
        entorno = Entorno(**kwargs)  # type: ignore[arg-type]
        destino = tmp_path / "ws"
        monkeypatch.setattr(modulo, "DESTINO", destino)
        monkeypatch.setattr(modulo, "RAIZ", tmp_path)
        monkeypatch.setattr(modulo, "conectar", entorno.conectar)
        monkeypatch.setattr(
            modulo,
            "BinanceTestnetClient",
            lambda c, r: BinanceTestnetClient(c, r, transport=entorno.transporte),
        )
        monkeypatch.setattr(
            modulo.BinanceTestnetCredentials,
            "from_env",
            lambda _env: BinanceTestnetCredentials(CLAVE, SECRETO),
        )
        return modulo, entorno, destino

    return armar


def test_el_orden_de_las_fases_deja_un_fill_sin_socket(
    grabador: Callable[..., tuple[ModuleType, Entorno, Path]],
) -> None:
    modulo, entorno, destino = grabador()

    assert modulo.main([]) == 0

    # El fill de la fase B ocurre con el primer socket cerrado y antes de
    # reconectar: es lo que fija el contrato de recuperacion.
    h = entorno.hechos
    cerrar_1 = h.index("ws0:cerrar")
    gap = next(i for i, x in enumerate(h) if "rec8-b-gap" in x)
    reconectar = h.index("ws1:conectar")
    assert cerrar_1 < gap < reconectar
    lados = [(p["side"][0], p["quantity"][0]) for p in entorno.pedidos]
    assert lados == [("BUY", "0.001"), ("SELL", "0.001")] * 2
    assert all(s.cerrado for s in entorno.sockets)


def test_se_escriben_tres_fases_con_manifest_verificable(
    grabador: Callable[..., tuple[ModuleType, Entorno, Path]],
) -> None:
    import hashlib

    modulo, _, destino = grabador()
    modulo.main([])

    manifest = json.loads((destino / "manifest.json").read_text())
    assert set(manifest["files"]) == {
        "fase_a_fill_normal.jsonl",
        "fase_b_fill_con_socket_caido.jsonl",
        "fase_c_firma_invalida_y_baja.jsonl",
    }
    for nombre, meta in manifest["files"].items():
        datos = (destino / nombre).read_bytes()
        assert hashlib.sha256(datos).hexdigest() == meta["sha256"]
        assert len(datos.splitlines()) == meta["frames"]
    fase_c = [
        json.loads(x)["frame"]
        for x in (destino / "fase_c_firma_invalida_y_baja.jsonl")
        .read_text()
        .splitlines()
    ]
    assert fase_c[0]["status"] == 400  # firma invalida
    assert fase_c[1]["status"] == 200  # baja de la suscripcion


def test_nada_con_forma_de_clave_llega_a_disco(
    grabador: Callable[..., tuple[ModuleType, Entorno, Path]],
) -> None:
    modulo, _, destino = grabador()
    modulo.main([])
    todo = "".join(p.read_text() for p in destino.iterdir())
    assert CLAVE not in todo
    assert SECRETO not in todo
    assert "signature" not in todo.lower()


def test_un_evento_con_forma_de_clave_aborta_sin_escribir(
    grabador: Callable[..., tuple[ModuleType, Entorno, Path]],
) -> None:
    modulo, _, destino = grabador(filtrar="Z" * 64)
    with pytest.raises(SystemExit, match="forma de clave o firma"):
        modulo.main([])
    assert not destino.exists()


def test_suscripcion_rechazada_aborta_y_muestra_el_error(
    grabador: Callable[..., tuple[ModuleType, Entorno, Path]],
) -> None:
    modulo, entorno, destino = grabador(estado_suscripcion=410)
    with pytest.raises(SystemExit, match=r"no acepto la suscripcion \(status 410"):
        modulo.main([])
    assert not destino.exists()
    assert entorno.pedidos == []  # no opero: sin suscripcion no hay ordenes


def test_escuchar_se_detiene_tras_el_silencio_pedido() -> None:
    modulo = _cargar()
    sock = FakeSocket(Entorno(), 0)
    sock.cola.extend(['{"a": 1}', '{"a": 2}'])
    fase = modulo.Fase("f", modulo.SystemClock())

    assert modulo.escuchar(sock, fase, silencio_s=3) is False
    assert [x["frame"] for x in fase.lineas] == [{"a": 1}, {"a": 2}]
