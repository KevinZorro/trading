"""Respuestas reales grabadas de Binance, verificadas contra su manifest.

Las graba ``scripts/grabar_respuestas_binance.py``. Ningun test inventa una
respuesta: los casos de error se derivan modificando una grabacion real.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from live.config import MARKET_DATA_URL
from live.marketdata import BinanceRestMarketData
from sim.clock import SimulatedClock

RESPUESTAS = Path(__file__).parent / "respuestas"
MANIFEST: dict[str, Any] = json.loads((RESPUESTAS / "manifest.json").read_text())
ARCHIVO_POR_PATH = {
    "/api/v3/depth": "depth.json",
    "/api/v3/klines": "klines_1d.json",
    "/api/v3/exchangeInfo": "exchange_info.json",
}
DIA_MS = 86_400_000


def grabacion(nombre: str) -> Any:
    """JSON de una grabacion, solo si su SHA256 coincide con el manifest."""
    cuerpo = (RESPUESTAS / nombre).read_bytes()
    assert hashlib.sha256(cuerpo).hexdigest() == MANIFEST["files"][nombre]["sha256"]
    return json.loads(cuerpo)


def recibido(nombre: str) -> np.datetime64:
    return np.datetime64(MANIFEST["files"][nombre]["received_at_utc"].rstrip("Z"), "ns")


class Servidor:
    """Doble del HTTP: sirve las grabaciones, con modificaciones por path."""

    def __init__(self, **modificadas: Any) -> None:
        self.cuerpos = {
            path: modificadas.get(nombre.split(".")[0], grabacion(nombre))
            for path, nombre in ARCHIVO_POR_PATH.items()
        }
        self.urls: list[str] = []
        self.al_pedir: Callable[[], None] = lambda: None

    def __call__(self, url: str) -> bytes:
        self.urls.append(url)
        self.al_pedir()
        return json.dumps(self.cuerpos[urlsplit(url).path]).encode()


def cliente(
    servidor: Servidor | None = None, ahora: np.datetime64 | None = None
) -> BinanceRestMarketData:
    reloj = SimulatedClock(ahora if ahora is not None else recibido("depth.json"))
    return BinanceRestMarketData(MARKET_DATA_URL, reloj, fetch=servidor or Servidor())
