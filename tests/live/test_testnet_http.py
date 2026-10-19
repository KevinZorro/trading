"""Capa HTTP firmada de testnet: firma, guard, lista blanca y secretos.

Sin red: el transporte es un doble que captura lo que saldria. El oraculo de la
firma es el vector de la documentacion de Binance, no la implementacion.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
from pathlib import Path

import numpy as np
import pytest

from live.config import ConfigError
from live.guard import MainnetOrderError
from live.testnet_http import (
    ALLOWED_REQUESTS,
    BinanceApiError,
    BinanceHttpError,
    BinanceTestnetClient,
    BinanceTestnetCredentials,
    HttpRequest,
    HttpResponse,
    TransportError,
    find_secret_shapes,
    sign,
)
from sim.clock import SimulatedClock

# Vector de la documentacion de Binance (clave de ejemplo publica, no un secreto).
DOC_SECRET = (
    "NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZjInClVN65XAbvqqM6A7H5fATj0j"  # gitleaks:allow
)
DOC_PAYLOAD = (
    "symbol=LTCBTC&side=BUY&type=LIMIT&timeInForce=GTC&quantity=1&price=0.1"
    "&recvWindow=5000&timestamp=1499827319559"
)
DOC_FIRMA = "c8db56825ae71d6d79447849e617115f4a920fa2acdcab2b053c4b2838bd6b71"
CLAVE = "K" * 64
SECRETO = "S" * 64
AHORA = np.datetime64("2026-10-18T12:00:00.123", "ns")
AHORA_MS = 1_792_324_800_123  # 2026-10-18T12:00:00.123Z


class Doble:
    def __init__(self, *respuestas: HttpResponse | Exception) -> None:
        self.respuestas = list(respuestas)
        self.pedidos: list[HttpRequest] = []

    def __call__(self, pedido: HttpRequest) -> HttpResponse:
        self.pedidos.append(pedido)
        r = self.respuestas.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def cliente(doble: Doble, base_url: str = "https://testnet.binance.vision"):
    return BinanceTestnetClient(
        BinanceTestnetCredentials(CLAVE, SECRETO),
        SimulatedClock(AHORA),
        base_url=base_url,
        transport=doble,
    )


def ok(cuerpo: object = None, status: int = 200) -> HttpResponse:
    return HttpResponse(status, json.dumps(cuerpo or {}).encode())


def test_la_firma_coincide_con_el_vector_de_la_documentacion() -> None:
    assert sign(DOC_SECRET, DOC_PAYLOAD) == DOC_FIRMA


def test_post_firmado_va_en_el_cuerpo_con_firma_al_final() -> None:
    doble = Doble(ok())
    cliente(doble).call("POST", "/api/v3/order", {"symbol": "BTCUSDT", "side": "BUY"})
    (pedido,) = doble.pedidos
    assert pedido.url == "https://testnet.binance.vision/api/v3/order"  # sin query
    assert pedido.headers["X-MBX-APIKEY"] == CLAVE
    cuerpo = pedido.body.decode()
    firmado, firma = cuerpo.rsplit("&signature=", 1)
    assert firmado == (f"symbol=BTCUSDT&side=BUY&recvWindow=5000&timestamp={AHORA_MS}")
    assert firma == sign(SECRETO, firmado)


def test_get_firmado_va_en_la_url_y_publico_sin_firma() -> None:
    doble = Doble(ok(), ok())
    c = cliente(doble)
    c.call("GET", "/api/v3/account")
    c.call("GET", "/api/v3/time")
    firmado, publico = doble.pedidos
    consulta = urllib.parse.parse_qs(urllib.parse.urlsplit(firmado.url).query)
    assert set(consulta) == {"recvWindow", "timestamp", "signature"}
    assert publico.url == "https://testnet.binance.vision/api/v3/time"
    assert "X-MBX-APIKEY" not in publico.headers
    assert not publico.body


def test_sync_time_corrige_el_timestamp_de_las_firmas() -> None:
    doble = Doble(ok({"serverTime": AHORA_MS + 2500}), ok())
    c = cliente(doble)
    assert c.sync_time() == 2500
    c.call("GET", "/api/v3/account")
    consulta = urllib.parse.parse_qs(urllib.parse.urlsplit(doble.pedidos[1].url).query)
    assert consulta["timestamp"] == [str(AHORA_MS + 2500)]


@pytest.mark.parametrize(
    "base",
    [
        "https://api.binance.com",
        "https://data-api.binance.vision",
        "https://testnet.binance.vision.evil.com",
        "https://testnet.binance.vision@api.binance.com",
        "http://testnet.binance.vision",
    ],
)
def test_el_cliente_no_se_construye_contra_mainnet(base: str) -> None:
    with pytest.raises(MainnetOrderError, match="testnet|https|credenciales"):
        cliente(Doble(), base)


def test_base_url_con_path_se_rechaza() -> None:
    with pytest.raises(BinanceHttpError, match="sin path"):
        cliente(Doble(), "https://testnet.binance.vision/api/v3")


@pytest.mark.parametrize(
    ("metodo", "path"),
    [
        ("DELETE", "/api/v3/openOrders"),  # cancelar todo
        ("POST", "/sapi/v1/capital/withdraw/apply"),
        ("GET", "/api/v3/depth"),
        ("PUT", "/api/v3/order"),
    ],
)
def test_lo_que_no_esta_en_la_lista_blanca_no_sale(metodo: str, path: str) -> None:
    doble = Doble()
    with pytest.raises(BinanceHttpError, match="lista blanca"):
        cliente(doble).call(metodo, path)
    assert doble.pedidos == []


def test_la_lista_blanca_no_incluye_retiros_ni_cancelar_todo() -> None:
    assert not [
        p
        for _, p in ALLOWED_REQUESTS
        if "sapi" in p or (p.endswith("openOrders") and _ != "GET")
    ]


def test_un_error_del_venue_se_levanta_con_su_codigo_sin_la_url() -> None:
    cuerpo = {"code": -2013, "msg": "Order does not exist."}
    c = cliente(Doble(ok(cuerpo, 400)))
    with pytest.raises(BinanceApiError, match="code=-2013") as exc:
        c.call("GET", "/api/v3/order", {"symbol": "BTCUSDT", "origClientOrderId": "x"})
    assert (exc.value.code, exc.value.status) == (-2013, 400)
    assert "signature" not in str(exc.value)
    assert CLAVE not in str(exc.value)


def test_send_devuelve_el_error_crudo_sin_lanzar() -> None:
    r = cliente(Doble(ok({"code": -1022, "msg": "x"}, 401))).send(
        "GET", "/api/v3/account"
    )
    assert r.status == 401


def test_respuesta_que_no_es_json_y_http_sin_codigo_son_errores_claros() -> None:
    with pytest.raises(BinanceHttpError, match="no es JSON"):
        cliente(Doble(HttpResponse(502, b"<html>"))).call("GET", "/api/v3/account")
    with pytest.raises(BinanceHttpError, match="HTTP 503"):
        cliente(Doble(ok({"x": 1}, 503))).call("GET", "/api/v3/account")


def test_el_timeout_se_propaga_como_transport_error() -> None:
    with pytest.raises(TransportError, match="sin respuesta"):
        cliente(Doble(TransportError("sin respuesta del venue"))).call(
            "POST", "/api/v3/order"
        )


def test_las_credenciales_no_aparecen_en_repr_ni_en_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    doble = Doble(ok())
    c = cliente(doble)
    with caplog.at_level(logging.DEBUG, logger="live"):
        c.call("POST", "/api/v3/order", {"symbol": "BTCUSDT"})
    firma = doble.pedidos[0].body.decode().rsplit("=", 1)[1]
    visible = repr(c) + repr(c._credentials) + repr(doble.pedidos[0]) + caplog.text
    for secreto in (CLAVE, SECRETO, firma):
        assert secreto not in visible


def test_credenciales_desde_el_entorno() -> None:
    creds = BinanceTestnetCredentials.from_env(
        {"TESTNET_API_KEY": " a ", "TESTNET_API_SECRET": "b"}
    )
    assert (creds.api_key, creds.api_secret) == ("a", "b")
    with pytest.raises(ConfigError, match="TESTNET_API_KEY"):
        BinanceTestnetCredentials.from_env({"TESTNET_API_KEY": "a"})


# -- recorrido de las grabaciones: nada con forma de clave o firma ----------

RESPUESTAS = Path(__file__).parent / "respuestas"


def test_ninguna_grabacion_contiene_algo_con_forma_de_clave_o_firma() -> None:
    archivos = [p for p in RESPUESTAS.rglob("*") if p.is_file()]
    assert archivos, "no hay grabaciones que revisar"
    for ruta in archivos:
        assert find_secret_shapes(ruta.read_text()) == [], ruta.name


@pytest.mark.parametrize(
    "texto",
    [
        "K" * 64,
        "a1" * 32,
        '{"x": "X-MBX-APIKEY: abc"}',
        "https://testnet.binance.vision/api/v3/account?timestamp=1&signature=ab",
        '{"apiKey": "abc"}',
    ],
)
def test_el_detector_encuentra_lo_que_tiene_forma_de_secreto(texto: str) -> None:
    assert find_secret_shapes(texto)


def test_el_detector_ignora_los_sha256_del_manifest_y_los_numeros() -> None:
    manifest = '{"sha256": "' + "ab" * 32 + '"}'
    assert find_secret_shapes(manifest) == []
    assert find_secret_shapes("[[67123.45000000, 0.01500000]]") == []
    assert find_secret_shapes("a" * 65) == []  # demasiado largo para ser una clave
