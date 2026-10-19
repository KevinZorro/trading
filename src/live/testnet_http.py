"""Capa HTTP firmada contra testnet de Binance spot (ADR 0007, PR 7a).

Todo lo que sale hacia el venue de ordenes pasa por ``BinanceTestnetClient``, y cada
peticion cumple tres cosas **antes** de abrir la conexion:

1. la URL pasa por ``live.guard.assert_testnet_order_url`` (host exacto de
   testnet: mainnet es de solo lectura por construccion);
2. el par ``(metodo, path)`` esta en ``ALLOWED_REQUESTS``. ``DELETE
   /api/v3/openOrders`` (cancelar todo) y cualquier ``/sapi`` quedan fuera;
3. la firma HMAC-SHA256 se calcula sobre la cadena exacta que viaja.

Las credenciales viven en variables de entorno y nunca en ``repr``, mensajes de
excepcion ni logs: una URL firmada contiene la firma, asi que ningun mensaje
incluye la URL ni el cuerpo de la peticion. El tiempo entra por el ``Clock``
inyectado, mas el desvio medido contra ``/api/v3/time``.

El HTTP se inyecta (``Transport``). Lo que sale de aca no sabe de ``Fill`` ni de
``ExecutionVenue``: eso es el PR 7b, que se testea contra respuestas grabadas.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from live.config import ConfigError
from live.guard import assert_testnet_order_url
from sim.clock import Clock

TESTNET_BASE_URL = "https://testnet.binance.vision"
ENV_API_KEY = "TESTNET_API_KEY"
ENV_API_SECRET = "TESTNET_API_SECRET"
RECV_WINDOW_MS = 5000
TIMEOUT_S = 10.0
# Solo lo que el venue de testnet necesita. Ampliarlo es una decision de
# seguridad y se revisa como tal.
ALLOWED_REQUESTS: frozenset[tuple[str, str]] = frozenset(
    {
        ("GET", "/api/v3/time"),
        ("GET", "/api/v3/exchangeInfo"),
        ("GET", "/api/v3/account"),
        ("GET", "/api/v3/openOrders"),
        ("GET", "/api/v3/myTrades"),
        ("GET", "/api/v3/order"),
        ("POST", "/api/v3/order"),
        ("DELETE", "/api/v3/order"),
    }
)
_PUBLICOS = frozenset({"/api/v3/time", "/api/v3/exchangeInfo"})
# Forma de una API key o de una firma de Binance: 64 caracteres alfanumericos.
_FORMA_SECRETO = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9]{64}(?![A-Za-z0-9])")
# Los SHA256 del manifest tambien miden 64 caracteres y no son secretos.
_SHA256_MANIFEST = re.compile(r'"sha256":\s*"[0-9a-f]{64}"')
_MARCAS = ("x-mbx-apikey", "signature=", "apikey", "api_key")

log = logging.getLogger(__name__)


class BinanceHttpError(RuntimeError):
    """Base de los errores de esta capa. Nunca lleva la URL ni el cuerpo."""


class TransportError(BinanceHttpError):
    """Sin respuesta (timeout o caida): **el estado del envio es desconocido**."""


class BinanceApiError(BinanceHttpError):
    """El venue respondio con un error: ``{"code": -2013, "msg": "..."}``."""

    def __init__(self, status: int, code: int, msg: str) -> None:
        super().__init__(f"binance {status} code={code}: {msg}")
        self.status = status
        self.code = code
        self.msg = msg


@dataclass(frozen=True)
class HttpRequest:
    method: str
    url: str
    headers: Mapping[str, str] = field(repr=False)  # lleva X-MBX-APIKEY
    body: bytes = field(default=b"", repr=False)  # lleva la firma en POST/DELETE


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes


Transport = Callable[[HttpRequest], HttpResponse]


def urllib_transport(request: HttpRequest) -> HttpResponse:  # pragma: no cover - red
    """Transporte real. Un 4xx es una respuesta (trae ``code``), no una excepcion."""
    pedido = urllib.request.Request(
        request.url,
        data=request.body or None,
        headers=dict(request.headers),
        method=request.method,
    )
    try:
        with urllib.request.urlopen(pedido, timeout=TIMEOUT_S) as r:
            return HttpResponse(r.status, r.read())
    except urllib.error.HTTPError as exc:
        return HttpResponse(exc.code, exc.read())
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise TransportError(f"sin respuesta del venue: {type(exc).__name__}") from None


@dataclass(frozen=True)
class BinanceTestnetCredentials:
    api_key: str = field(repr=False)
    api_secret: str = field(repr=False)

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> BinanceTestnetCredentials:
        clave = environ.get(ENV_API_KEY, "").strip()
        secreto = environ.get(ENV_API_SECRET, "").strip()
        if not clave or not secreto:
            raise ConfigError(f"faltan {ENV_API_KEY} y {ENV_API_SECRET} en el entorno")
        return cls(api_key=clave, api_secret=secreto)


def sign(secret: str, payload: str) -> str:
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def find_secret_shapes(text: str) -> list[str]:
    """Fragmentos con forma de clave o firma, o marcas de cabecera/query firmada.

    Lo usan el script de grabacion (antes de escribir) y un test que recorre
    ``tests/live/respuestas/``: gitleaks no siempre reconoce una clave de Binance.
    """
    text = _SHA256_MANIFEST.sub('"sha256": ""', text)
    hallazgos = [m.group(0) for m in _FORMA_SECRETO.finditer(text)]
    bajo = text.lower()
    hallazgos += [marca for marca in _MARCAS if marca in bajo]
    return hallazgos


class BinanceTestnetClient:
    def __init__(
        self,
        credentials: BinanceTestnetCredentials,
        clock: Clock,
        *,
        base_url: str = TESTNET_BASE_URL,
        transport: Transport = urllib_transport,
    ) -> None:
        assert_testnet_order_url(base_url)
        partes = urllib.parse.urlsplit(base_url)
        if partes.path.strip("/") or partes.query:
            raise BinanceHttpError("base_url debe ser https://host sin path ni query")
        self._credentials = credentials
        self._clock = clock
        self._base_url = base_url.rstrip("/")
        self._transport = transport
        self._offset_ms = 0

    def __repr__(self) -> str:
        return f"BinanceTestnetClient(base_url={self._base_url!r})"

    def _now_ms(self) -> int:
        local = int(self._clock.now().astype("datetime64[ms]").astype(np.int64))
        return local + self._offset_ms

    def sync_time(self) -> int:
        """Mide ``serverTime - reloj local`` y lo aplica a las firmas. Devuelve ms."""
        antes = self._now_ms()
        servidor = int(self.call("GET", "/api/v3/time")["serverTime"])
        self._offset_ms += servidor - antes
        return self._offset_ms

    def send(
        self, method: str, path: str, params: Mapping[str, str | int] | None = None
    ) -> HttpResponse:
        """Una peticion; devuelve la respuesta cruda, errores del venue incluidos."""
        if (method, path) not in ALLOWED_REQUESTS:
            raise BinanceHttpError(f"{method} {path} fuera de la lista blanca")
        pares = list((params or {}).items())
        cabeceras = {"Accept": "application/json"}
        if path not in _PUBLICOS:
            pares += [("recvWindow", RECV_WINDOW_MS), ("timestamp", self._now_ms())]
            cabeceras["X-MBX-APIKEY"] = self._credentials.api_key
        cadena = urllib.parse.urlencode(pares)
        if path not in _PUBLICOS:
            cadena += "&signature=" + sign(self._credentials.api_secret, cadena)
        url = assert_testnet_order_url(f"{self._base_url}{path}")
        cuerpo = b""
        if method == "GET":
            url = f"{url}?{cadena}" if cadena else url
        else:
            cuerpo = cadena.encode()
            cabeceras["Content-Type"] = "application/x-www-form-urlencoded"
        log.info("peticion a testnet", extra={"method": method, "path": path})
        return self._transport(HttpRequest(method, url, cabeceras, cuerpo))

    def call(
        self, method: str, path: str, params: Mapping[str, str | int] | None = None
    ) -> Any:
        """JSON de la respuesta; lanza ``BinanceApiError`` si el venue dio error."""
        respuesta = self.send(method, path, params)
        try:
            datos = json.loads(respuesta.body)
        except ValueError:
            raise BinanceHttpError(
                f"{method} {path}: respuesta {respuesta.status} que no es JSON"
            ) from None
        if respuesta.status >= 400:
            if isinstance(datos, dict) and "code" in datos:
                raise BinanceApiError(
                    respuesta.status, int(datos["code"]), str(datos.get("msg", ""))
                )
            raise BinanceHttpError(f"{method} {path}: HTTP {respuesta.status}")
        return datos
