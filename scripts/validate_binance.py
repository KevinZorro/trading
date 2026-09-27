"""Reporte de validacion de los datasets de Binance versionados.

Uso (desde la raiz del repo)::

    PYTHONPATH=src uv run python scripts/validate_binance.py \\
        > datasets/binance/VALIDACION.md

Verifica el SHA256 de cada dataset contra su manifest y corre la bateria de
``data.validation`` completa, sin parar en el primer fallo. No repara nada: si
hay huecos, el reporte los cuenta y el dataset queda como el venue lo publico.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from data import AlwaysOpen, binance_spot_spec
from data.manifest import MANIFEST_SUFFIX, read_manifest, verify_manifest
from data.validation import validation_report

RAIZ = Path(__file__).resolve().parent.parent
DATASETS = RAIZ / "datasets" / "binance"
FRECUENCIA = {"1d": "1D", "1h": "1h"}


def main() -> int:
    manifests = sorted(DATASETS.glob(f"*{MANIFEST_SUFFIX}"))
    if not manifests:
        print(f"no hay manifests en {DATASETS}", file=sys.stderr)
        return 1
    print("# Validacion de los datasets de Binance\n")
    print("Generado por `scripts/validate_binance.py`. No editar a mano.\n")
    for ruta in manifests:
        manifest = read_manifest(ruta)
        datos = verify_manifest(ruta)
        frame = pd.read_csv(datos)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame["split_factor"] = 1.0
        frame["cash_dividend"] = 0.0
        reporte = validation_report(
            frame,
            instrument=binance_spot_spec(manifest.symbol),
            calendar=AlwaysOpen(),
            freq=FRECUENCIA[manifest.interval],
        )
        print(f"## `{manifest.file}`\n")
        print(f"- SHA256 verificado: `{manifest.sha256}`")
        print(f"- rango: {manifest.first_timestamp} -> {manifest.last_timestamp}")
        print(f"- descargado: {manifest.downloaded_at}\n")
        print("```")
        print(reporte.render())
        print("```\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
