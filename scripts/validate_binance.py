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
        frame["open_time"] = pd.to_datetime(frame["open_time"])
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
        _anomalias(frame, pd.Timedelta(FRECUENCIA[manifest.interval]))
    return 0


def _anomalias(frame: pd.DataFrame, paso: pd.Timedelta) -> None:
    """Klines cerradas antes de tiempo por el venue, con el hueco que sigue.

    No se reparan. Se listan para que cualquier ejecucion que dependa de ellas
    se pueda identificar como outlier.
    """
    marcadas = frame.index[frame["close_time_desvio_ms"] != 0]
    print(f"### Anomalias del venue: {len(marcadas)}\n")
    if len(marcadas) == 0:
        return
    print(
        "Klines con `close_time` distinto del teorico. Conservan el cierre "
        "teorico como `timestamp` (sigue siendo point-in-time) y quedan marcadas "
        "en `close_time_desvio_ms`. **Una decision tomada en el cierre de una de "
        "estas barras se ejecuta en el open de la primera barra tras reanudar**, "
        "con un gap que puede ser grande: esas ejecuciones se reportan como "
        "outliers, identificadas por la marca de la barra de decision.\n"
    )
    print(
        "| timestamp | open_time | cierre real | desvio | volumen | trades "
        "| barras faltantes despues | gap al reanudar |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for i in marcadas:
        fila = frame.loc[i]
        abre = pd.Timestamp(fila["open_time"])
        real = (
            abre + paso + pd.Timedelta(milliseconds=int(fila["close_time_desvio_ms"]))
        )
        real -= pd.Timedelta(milliseconds=1)
        if i + 1 < len(frame):
            siguiente = frame.loc[i + 1]
            faltan = int((siguiente["timestamp"] - fila["timestamp"]) / paso) - 1
            gap = f"{siguiente['open'] / fila['close'] - 1:+.2%}"
        else:
            faltan, gap = 0, "n/a"
        cierre_real = real.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(
            f"| {fila['timestamp']} | {abre} | {cierre_real} "
            f"| {int(fila['close_time_desvio_ms'])} ms | {fila['volume']:.2f} "
            f"| {int(fila['n_trades'])} | {faltan} | {gap} |"
        )
    print()


if __name__ == "__main__":
    sys.exit(main())
