"""El guard que hace a mainnet de solo lectura por construccion (ADR 0007).

Cada caso de rechazo es una forma real de que una URL "contenga" el host de
testnet y apunte a otro lado. Si alguno pasara, una orden podria llegar a
mainnet con un error de configuracion.
"""

from __future__ import annotations

import pytest

from live.guard import TESTNET_ORDER_HOSTS, MainnetOrderError, assert_testnet_order_url

VALIDA = "https://testnet.binance.vision/api/v3/order"


def test_la_url_de_testnet_pasa() -> None:
    assert assert_testnet_order_url(VALIDA) == VALIDA
    assert assert_testnet_order_url("https://TESTNET.binance.vision/api/v3/order")


@pytest.mark.parametrize(
    "url",
    [
        "https://api.binance.com/api/v3/order",  # mainnet
        "https://api1.binance.com/api/v3/order",
        "https://data-api.binance.vision/api/v3/order",  # datos publicos de mainnet
        "https://testnet.binance.vision.evil.com/api/v3/order",  # sufijo
        "https://evil-testnet.binance.vision/api/v3/order",
        "https://api.binance.com/testnet.binance.vision/api/v3/order",  # en el path
        "https://api.binance.com/api/v3/order?h=testnet.binance.vision",  # en la query
    ],
)
def test_cualquier_otro_host_se_rechaza(url: str) -> None:
    with pytest.raises(MainnetOrderError, match="no es de testnet"):
        assert_testnet_order_url(url)


def test_credenciales_embebidas_disfrazan_el_host() -> None:
    """``testnet.binance.vision@api.binance.com`` va a api.binance.com."""
    with pytest.raises(MainnetOrderError, match="credenciales embebidas"):
        assert_testnet_order_url("https://testnet.binance.vision@api.binance.com/order")


def test_solo_https() -> None:
    with pytest.raises(MainnetOrderError, match="solo por https"):
        assert_testnet_order_url("http://testnet.binance.vision/api/v3/order")


def test_puerto_explicito_se_rechaza() -> None:
    with pytest.raises(MainnetOrderError, match="puerto explicito"):
        assert_testnet_order_url("https://testnet.binance.vision:8443/api/v3/order")


def test_la_lista_de_hosts_permitidos_es_la_declarada() -> None:
    """Ampliarla es una decision de seguridad: este test obliga a verla en el diff."""
    assert frozenset({"testnet.binance.vision"}) == TESTNET_ORDER_HOSTS
