"""Graba una respuesta real de cada endpoint de mercado que usa ``live.marketdata``.

Uso (desde la raiz del repo, con red)::

    PYTHONPATH=src uv run python scripts/grabar_respuestas_binance.py

Escribe en ``tests/live/respuestas/`` el cuerpo **byte a byte** de:

- ``GET /api/v3/depth?symbol=BTCUSDT&limit=1000``
- ``GET /api/v3/klines?symbol=BTCUSDT&interval=1d&limit=30``
- ``GET /api/v3/exchangeInfo?symbol=BTCUSDT``

y un ``manifest.json`` con la URL, la hora UTC de recepcion y el SHA256 de
cada archivo. Los tests de ``live`` corren contra estos archivos y verifican el
SHA256 antes de usarlos: una grabacion editada a mano no llega a un test.

Solo datos publicos (``data-api.binance.vision``): sin API key y sin ninguna
ruta que envie ordenes. Re-correrlo reemplaza las grabaciones; los tests que
dependen de valores concretos del libro se recalculan, no se ajustan a mano.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from sim.clock import SystemClock

RAIZ = Path(__file__).resolve().parent.parent
DESTINO = RAIZ / "tests" / "live" / "respuestas"
BASE_URL = "https://data-api.binance.vision"
TIMEOUT_S = 30


def _endpoints(symbol: str) -> dict[str, tuple[str, dict[str, str | int]]]:
    return {
        "depth.json": ("/api/v3/depth", {"symbol": symbol, "limit": 1000}),
        "klines_1d.json": (
            "/api/v3/klines",
            {"symbol": symbol, "interval": "1d", "limit": 30},
        ),
        "exchange_info.json": ("/api/v3/exchangeInfo", {"symbol": symbol}),
    }


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as respuesta:
        cuerpo: bytes = respuesta.read()
    json.loads(cuerpo)  # falla aca si el venue devolvio algo que no es JSON
    return cuerpo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbol", default="BTCUSDT")
    args = parser.parse_args(argv)

    DESTINO.mkdir(parents=True, exist_ok=True)
    reloj = SystemClock()
    archivos: dict[str, dict[str, str]] = {}
    for nombre, (path, params) in _endpoints(args.symbol).items():
        url = f"{BASE_URL}{path}?{urllib.parse.urlencode(params)}"
        cuerpo = _get(url)
        recibido = reloj.now()
        (DESTINO / nombre).write_bytes(cuerpo)
        archivos[nombre] = {
            "url": url,
            "received_at_utc": np.datetime_as_string(recibido, unit="ms") + "Z",
            "sha256": hashlib.sha256(cuerpo).hexdigest(),
        }
    manifest = {
        "script": "scripts/grabar_respuestas_binance.py",
        "symbol": args.symbol,
        "files": archivos,
    }
    texto = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    (DESTINO / "manifest.json").write_text(texto, encoding="utf-8")
    print(texto)
    return 0


if __name__ == "__main__":
    sys.exit(main())
