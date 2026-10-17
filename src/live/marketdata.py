"""Datos de mercado de mainnet por REST, de solo lectura. Ver ``docs/adr/0007``.

``MarketData`` es lo que la sombra necesita del venue: una foto del libro, las
ultimas barras **cerradas** y las reglas del instrumento. ``BinanceRestMarketData``
lo implementa contra el host de datos publicos (``data-api.binance.vision``),
sin API key.

Solo lectura por construccion, y no solo por el host: toda URL pasa por
:func:`assert_market_data_url`, que admite **unicamente** tres paths GET.
``/api/v3/order`` o ``/api/v3/account`` lanzan antes de abrir la conexion,
aunque alguien apunte ``SHADOW_MARKET_DATA_URL`` a ``api.binance.com``.

El HTTP se inyecta (``Fetch``): en los tests es un doble que sirve respuestas
reales grabadas por ``scripts/grabar_respuestas_binance.py``.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from data.binance import MAX_LIMIT, klines_to_frame
from data.errors import DataError
from data.instruments import CommissionSchema, InstrumentSpec
from data.loaders import bars_from_frame
from data.schema import BarSeries, FloatArray
from sim.clock import Clock

ALLOWED_PATHS: frozenset[str] = frozenset(
    {"/api/v3/depth", "/api/v3/klines", "/api/v3/exchangeInfo"}
)
# Rango que acepta GET /api/v3/depth.
MAX_DEPTH_LIMIT = 5000
TIMEOUT_S = 10.0
# Frecuencia de BarSeries por intervalo de data.binance.INTERVALS; la misma de
# scripts/validate_binance.py.
FREQ = {"1d": "1D", "1h": "1h"}
# ASSUMPTION: exchangeInfo es publico y no trae la comision, que depende de la
# cuenta. Se usa la del tarifario (10 bps taker, sin descuentos), la misma de
# binance_spot_spec y del ADR 0006. La sombra no la mide (ADR 0007, seccion 3).
DEFAULT_COMMISSION = CommissionSchema(kind="percent", value=0.001)

Fetch = Callable[[str], bytes]

log = logging.getLogger(__name__)


class MarketDataError(RuntimeError):
    """El venue respondio algo que no se puede usar, o la URL no es admisible."""


class InvalidBookError(MarketDataError):
    """El libro de ordenes no cumple los invariantes de un libro sano."""


@dataclass(frozen=True)
class BookSnapshot:
    """Foto del libro L2. No existe una instancia que no haya pasado la validacion.

    Un libro cruzado o desordenado no se corrige: se rechaza con el motivo.
    Reordenarlo en silencio haria que ``walk_book`` midiera un costo sobre un
    libro que el venue nunca publico.
    """

    received_at: np.datetime64  # del Clock inyectado; el REST de depth no trae hora
    last_update_id: int
    bids: FloatArray  # (n, 2) precio, cantidad; mejor primero
    asks: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "received_at", np.datetime64(self.received_at, "ns"))
        for lado in ("bids", "asks"):
            niveles = np.array(getattr(self, lado), dtype=np.float64)
            if niveles.ndim != 2 or niveles.shape[1] != 2 or len(niveles) == 0:
                raise InvalidBookError(
                    f"{lado} debe ser (n>=1, 2) precio y cantidad, es "
                    f"{niveles.shape}"
                )
            if not np.isfinite(niveles).all():
                raise InvalidBookError(f"{lado} tiene valores no finitos")
            if (niveles <= 0).any():
                fila = int(np.argwhere(niveles <= 0)[0][0])
                raise InvalidBookError(
                    f"{lado} nivel {fila} con precio o cantidad <= 0: "
                    f"{niveles[fila].tolist()}"
                )
            niveles.flags.writeable = False
            object.__setattr__(self, lado, niveles)
        if not (np.diff(self.bids[:, 0]) < 0).all():
            raise InvalidBookError("bids no estrictamente descendentes por precio")
        if not (np.diff(self.asks[:, 0]) > 0).all():
            raise InvalidBookError("asks no estrictamente ascendentes por precio")
        if self.best_bid >= self.best_ask:
            raise InvalidBookError(
                f"libro cruzado: best bid {self.best_bid} >= best ask {self.best_ask}"
            )

    @property
    def best_bid(self) -> float:
        return float(self.bids[0, 0])

    @property
    def best_ask(self) -> float:
        return float(self.asks[0, 0])


class MarketData(Protocol):
    def depth(self, symbol: str, limit: int) -> BookSnapshot: ...

    def closed_klines(self, symbol: str, interval: str, n: int) -> BarSeries: ...

    def instrument(self, symbol: str) -> InstrumentSpec: ...


def assert_market_data_url(url: str, base_url: str) -> str:
    """Devuelve la URL si es del host de datos y de un path de la lista blanca.

    Comparacion por host y puerto exactos tras parsear, como en
    ``live.guard``: un ``startswith`` dejaria pasar ``https://host.evil.com``.
    """
    partes = urllib.parse.urlsplit(url)
    base = urllib.parse.urlsplit(base_url)
    if partes.scheme != "https":
        raise MarketDataError(f"datos de mercado solo por https, no {partes.scheme!r}")
    if partes.username is not None or partes.password is not None:
        raise MarketDataError("URL de datos con credenciales embebidas: rechazada")
    if (partes.hostname, partes.port) != (base.hostname, base.port):
        raise MarketDataError(
            f"host {partes.netloc!r} distinto del configurado {base.netloc!r}"
        )
    if partes.path not in ALLOWED_PATHS:
        raise MarketDataError(
            f"path {partes.path!r} fuera de la lista blanca {sorted(ALLOWED_PATHS)}: "
            "MarketData es de solo lectura"
        )
    return url


def urllib_fetch(url: str) -> bytes:  # pragma: no cover - red real
    pedido = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(pedido, timeout=TIMEOUT_S) as respuesta:
        cuerpo: bytes = respuesta.read()
    return cuerpo


class BinanceRestMarketData:
    """``MarketData`` sobre la API REST publica de Binance spot."""

    def __init__(
        self,
        base_url: str,
        clock: Clock,
        *,
        fetch: Fetch = urllib_fetch,
        commission: CommissionSchema = DEFAULT_COMMISSION,
    ) -> None:
        base = urllib.parse.urlsplit(base_url)
        if base.scheme != "https" or base.path.strip("/") or base.query:
            raise MarketDataError(
                f"base_url debe ser https://host sin path ni query: {base_url!r}"
            )
        self._base_url = base_url.rstrip("/")
        self._clock = clock
        self._fetch = fetch
        self._commission = commission

    def _get(self, path: str, params: dict[str, str | int]) -> Any:
        url = f"{self._base_url}{path}?{urllib.parse.urlencode(params)}"
        assert_market_data_url(url, self._base_url)
        try:
            return json.loads(self._fetch(url))
        except (OSError, ValueError) as exc:
            raise MarketDataError(f"GET {path} fallo: {exc}") from exc

    def depth(self, symbol: str, limit: int) -> BookSnapshot:
        if not 1 <= limit <= MAX_DEPTH_LIMIT:
            raise MarketDataError(f"limit={limit} fuera de [1, {MAX_DEPTH_LIMIT}]")
        datos = self._get("/api/v3/depth", {"symbol": symbol, "limit": limit})
        recibido = self._clock.now()
        try:
            return BookSnapshot(
                received_at=recibido,
                last_update_id=int(datos["lastUpdateId"]),
                bids=np.array(datos["bids"], dtype=np.float64),
                asks=np.array(datos["asks"], dtype=np.float64),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MarketDataError(f"respuesta de depth malformada: {exc}") from exc

    def closed_klines(self, symbol: str, interval: str, n: int) -> BarSeries:
        """Las ``n`` ultimas barras cerradas, con ``timestamp`` = cierre.

        La hora de corte se toma **antes** de pedir: si la barra cierra
        mientras la respuesta viaja, el venue la serializo todavia abierta,
        con valores provisorios, y tomar la hora al recibir la aceptaria.

        ASSUMPTION: el reloj local no adelanta al del venue. Si adelantara, una
        vela todavia abierta se tomaria por cerrada. La sombra pide a las
        00:00:30 (ADR 0007, seccion 4): tolera hasta 30 s de desvio.
        """
        if interval not in FREQ:
            raise MarketDataError(f"intervalo no soportado: {interval!r}")
        if not 1 <= n < MAX_LIMIT:
            raise MarketDataError(f"n={n} fuera de [1, {MAX_LIMIT - 1}]")
        corte = self._clock.now()
        # n + 1: la ultima kline de la respuesta suele ser la vela en curso.
        filas = self._get(
            "/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": n + 1}
        )
        if not isinstance(filas, list):
            raise MarketDataError(f"respuesta de klines inesperada: {str(filas)[:200]}")
        try:
            marco = klines_to_frame(filas, symbol=symbol, interval=interval, now=corte)
        except DataError as exc:
            raise MarketDataError(f"klines de {symbol} {interval}: {exc}") from exc
        if len(marco) < n:
            raise MarketDataError(
                f"se pidieron {n} klines cerradas y el venue devolvio {len(marco)}"
            )
        marco = marco.tail(n).reset_index(drop=True)
        anomalas = marco.loc[marco["close_time_desvio_ms"] != 0, "timestamp"]
        if not anomalas.empty:
            # Se conservan sin reparar, como en el dataset versionado (data.binance).
            log.warning(
                "klines cerradas antes de tiempo por el venue",
                extra={"symbol": symbol, "timestamps": [str(t) for t in anomalas]},
            )
        return bars_from_frame(
            marco,
            instrument=self.instrument(symbol),
            freq=FREQ[interval],
            enforce_tick_size=True,
            source=f"binance-rest:{symbol}:{interval}",
        )

    def instrument(self, symbol: str) -> InstrumentSpec:
        """Filtros vigentes de ``exchangeInfo`` como ``InstrumentSpec``.

        ASSUMPTION: se ignoran ``MARKET_LOT_SIZE`` (su ``stepSize`` 0 delega en
        ``LOT_SIZE``; su ``maxQty`` es de cientos de BTC) y que ``NOTIONAL`` se
        evalua contra el precio medio de 5 minutos y no el del fill. A 10-1000
        USDT ninguno de los dos muerde.
        """
        datos = self._get("/api/v3/exchangeInfo", {"symbol": symbol})
        try:
            simbolos = [s for s in datos["symbols"] if s["symbol"] == symbol]
            if len(simbolos) != 1:
                raise MarketDataError(
                    f"exchangeInfo trae {len(simbolos)} entradas para {symbol}"
                )
            info = simbolos[0]
            if info["status"] != "TRADING":
                raise MarketDataError(f"{symbol} no opera: status={info['status']}")
            filtros = {f["filterType"]: f for f in info["filters"]}
            nocionales = [
                float(filtros[t]["minNotional"])
                for t in ("NOTIONAL", "MIN_NOTIONAL")
                if t in filtros
            ]
            if len(nocionales) != 1:
                raise MarketDataError(
                    f"{symbol}: se esperaba exactamente uno de NOTIONAL o "
                    f"MIN_NOTIONAL, hay {len(nocionales)}"
                )
            paso = float(filtros["LOT_SIZE"]["stepSize"])
            return InstrumentSpec(
                symbol=symbol,
                venue="BINANCE",
                tick_size=float(filtros["PRICE_FILTER"]["tickSize"]),
                lot_size=paso,
                allow_fractional=True,
                qty_precision=int(info["baseAssetPrecision"]),
                min_order_qty=float(filtros["LOT_SIZE"]["minQty"]),
                min_notional=nocionales[0],
                commission_schema=self._commission,
                asset_class="crypto",
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MarketDataError(
                f"exchangeInfo de {symbol} malformado: {exc}"
            ) from exc
