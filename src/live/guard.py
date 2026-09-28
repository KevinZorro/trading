"""Guard de URLs de envio de ordenes: solo testnet. No es una convencion.

Un error de configuracion -una variable de entorno copiada del lugar
equivocado- no puede terminar en una orden sobre mainnet. Todo codigo que envie
ordenes construye la URL y la pasa por aca antes de abrir la conexion.

La comparacion es por **host exacto**, despues de parsear la URL. Nada de
``in`` ni ``endswith`` sobre el string: ``https://testnet.binance.vision.evil.com``,
``https://api.binance.com/testnet.binance.vision`` y
``https://testnet.binance.vision@api.binance.com`` contienen el texto correcto y
apuntan a otro lado.
"""

from __future__ import annotations

from urllib.parse import urlsplit

# Hosts de la API REST de testnet de Binance spot. Cualquier otro host, mainnet
# incluido, es un error. Ampliar esta lista es una decision de seguridad y se
# revisa como tal (ADR 0007).
TESTNET_ORDER_HOSTS: frozenset[str] = frozenset({"testnet.binance.vision"})


class MainnetOrderError(RuntimeError):
    """Se intento enviar una orden a una URL que no es de testnet."""


def assert_testnet_order_url(url: str) -> str:
    """Devuelve la URL si es de testnet por HTTPS; si no, lanza.

    Se rechazan tambien las URLs con credenciales embebidas (``user@host``) y
    con puerto explicito: ninguna de las dos es necesaria para hablar con
    testnet, y las dos son formas de disfrazar el host real.
    """
    partes = urlsplit(url)
    host = (partes.hostname or "").lower()
    if partes.scheme != "https":
        raise MainnetOrderError(
            f"envio de ordenes solo por https, no {partes.scheme!r}"
        )
    if partes.username is not None or partes.password is not None:
        raise MainnetOrderError("URL de ordenes con credenciales embebidas: rechazada")
    if partes.port is not None:
        raise MainnetOrderError("URL de ordenes con puerto explicito: rechazada")
    if host not in TESTNET_ORDER_HOSTS:
        raise MainnetOrderError(
            f"host {host!r} no es de testnet ({sorted(TESTNET_ORDER_HOSTS)}). "
            "En la Etapa 3.5 mainnet es de solo lectura por construccion."
        )
    return url
