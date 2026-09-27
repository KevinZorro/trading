"""Reproduce los datasets de BTCUSDT versionados en ``datasets/binance/``.

Uso (desde la raiz del repo, con red)::

    PYTHONPATH=src uv run python scripts/download_binance.py --interval 1d
    PYTHONPATH=src uv run python scripts/download_binance.py --interval 1h

Solo datos publicos de mercado (``data-api.binance.vision``): sin API key y
sin ninguna ruta que envie ordenes. El resultado es un CSV y su manifest con
el SHA256. Re-correrlo otro dia agrega barras nuevas al final y por lo tanto
cambia la huella: el dataset del estudio es el commiteado, no el que produzca
este script manana.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from data.binance import build_dataset
from sim.clock import SystemClock

RAIZ = Path(__file__).resolve().parent.parent
DESTINO = RAIZ / "datasets" / "binance"
# Primer dia de BTCUSDT en Binance spot. Pedir desde antes no trae nada.
INICIO = pd.Timestamp("2017-08-17", tz="UTC")
EXTENSION = {"1d": ".csv", "1h": ".csv.gz"}


def _get_json(url: str, params: dict[str, str | int]) -> Any:
    consulta = urllib.parse.urlencode(params)
    with urllib.request.urlopen(f"{url}?{consulta}", timeout=30) as respuesta:
        return json.load(respuesta)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", choices=sorted(EXTENSION), required=True)
    args = parser.parse_args(argv)

    DESTINO.mkdir(parents=True, exist_ok=True)
    salida = DESTINO / f"{args.symbol}-{args.interval}{EXTENSION[args.interval]}"
    manifest = build_dataset(
        _get_json,
        symbol=args.symbol,
        interval=args.interval,
        start=INICIO,
        now=SystemClock().now(),
        out_path=salida,
        script="scripts/download_binance.py --interval " + args.interval,
    )
    print(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
