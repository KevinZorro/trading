"""Klines de Binance spot: descarga paginada, parseo y dataset versionado.

Solo datos publicos de mercado; este modulo no conoce credenciales ni envia
ordenes. La funcion que habla HTTP se inyecta (``GetJson``): en los tests es un
doble que devuelve respuestas grabadas, y la suite sigue corriendo sin red.

ASSUMPTION: el ``timestamp`` de cada barra es su **cierre** (``open_time`` +
intervalo), no su apertura. Es la convencion del proyecto (``WeekdayCalendar``
fecha las sesiones por su hora de cierre) y es la que hace point-in-time a la
fila completa: todo lo que contiene se conoce en ``timestamp``. Binance rotula
por la apertura; esa hora queda en la columna ``open_time`` para trazabilidad.
La Etapa 4 alinea noticias contra ``timestamp`` y hereda la garantia.

Klines cerradas antes de tiempo (caidas y mantenimientos del venue) conservan
el cierre teorico como ``timestamp`` y quedan marcadas en
``close_time_desvio_ms``. Consecuencia para el backtest: una decision tomada en
el cierre teorico de esa barra se ejecuta en el open de la primera barra tras
reanudar, con un gap que puede ser grande. Lo mismo pasa en todo hueco del
calendario, este o no precedido por una barra marcada: en BTCUSDT 1h, 17 de los
28 huecos siguen a una barra cerrada en hora. Una ejecucion es outlier si su
barra de decision esta marcada **o** si entre la barra de decision y la de
ejecucion falta al menos una barra; la marca sola no alcanza.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.errors import SchemaError
from data.manifest import DatasetManifest, manifest_path_for, sha256_of, write_manifest

PUBLIC_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"
MAX_LIMIT = 1000

# Orden de los campos de GET /api/v3/klines. El ultimo esta documentado como
# "unused field" y no se conserva.
KLINE_FIELDS: tuple[str, ...] = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "n_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
)

# Un epoch en milisegundos entre 2010 y 2100. En microsegundos o nanosegundos
# el valor es al menos mil veces mayor y cae fuera: el cambio de unidad se
# detecta por magnitud, sin contar filas.
EPOCH_MS_MIN = 1_262_304_000_000  # 2010-01-01
EPOCH_MS_MAX = 4_102_444_800_000  # 2100-01-01

# Umbral entre "caidas aisladas del venue" y "error estructural". Hacen falta
# las dos condiciones: con solo la fraccion, una descarga corta que contenga
# una caida fallaria por accidente; con solo el conteo, una historia larga con
# un intervalo equivocado pasaria. Observado en BTCUSDT: 1 de ~3.300 barras
# diarias y 5 de ~78.000 horarias.
MAX_ANOMALY_ROWS = 3
MAX_ANOMALY_FRAC = 0.01

INTERVALS: dict[str, pd.Timedelta] = {
    "1h": pd.Timedelta(hours=1),
    "1d": pd.Timedelta(days=1),
}

# Columnas del CSV versionado, en orden. ``symbol`` va al final porque el
# validador la exige y no es una medicion.
CSV_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "open_time",
    "close_time_desvio_ms",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "n_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "symbol",
)

GetJson = Callable[[str, dict[str, str | int]], Any]


def _intervalo(interval: str) -> pd.Timedelta:
    if interval not in INTERVALS:
        raise SchemaError(f"intervalo no soportado: {interval!r} ({sorted(INTERVALS)})")
    return INTERVALS[interval]


def _ms(ts: pd.Timestamp) -> int:
    return int(ts.value // 1_000_000)


def fetch_klines(
    get_json: GetJson,
    *,
    symbol: str,
    interval: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    url: str = PUBLIC_KLINES_URL,
) -> list[list[Any]]:
    """Klines con ``open_time`` en ``[start, end)``, paginando por ``startTime``.

    Pagina hasta que el venue devuelve una pagina vacia o corta. No confia en
    que cada pagina traiga ``MAX_LIMIT`` filas: si la API cambia su maximo, se
    sigue paginando igual.
    """
    _intervalo(interval)
    filas: list[list[Any]] = []
    desde = _ms(start)
    hasta = _ms(end) - 1
    while desde <= hasta:
        pagina = get_json(
            url,
            {
                "symbol": symbol,
                "interval": interval,
                "startTime": desde,
                "endTime": hasta,
                "limit": MAX_LIMIT,
            },
        )
        if not isinstance(pagina, list):
            raise SchemaError(f"respuesta inesperada del venue: {str(pagina)[:200]}")
        if not pagina:
            break
        ultimo = int(pagina[-1][0])
        if filas and ultimo <= int(filas[-1][0]):
            raise SchemaError(f"la paginacion no avanza: open_time {ultimo} repetido")
        filas.extend(pagina)
        desde = ultimo + 1
    return filas


def klines_to_frame(
    rows: Sequence[Sequence[Any]],
    *,
    symbol: str,
    interval: str,
    now: np.datetime64,
) -> pd.DataFrame:
    """Convierte klines crudas al esquema del CSV versionado.

    Descarta la barra en curso: una kline que no llego a su cierre teorico
    tiene high, low, close y volumen provisorios, y versionarla congelaria un
    valor que el venue va a cambiar.

    Separa dos fallos que se ven igual fila por fila:

    - **Cambio de unidad** (los volcados de data.binance.vision pasaron a
      microsegundos en 2025): se detecta por la magnitud del epoch, no contando
      filas. Falla siempre.
    - **Anomalias del venue**: klines cerradas antes de tiempo por una caida o
      un mantenimiento. Se conservan, se marcan en ``close_time_desvio_ms`` y se
      reportan; no se reparan. Si son mas de ``MAX_ANOMALY_ROWS`` filas **y** mas
      de ``MAX_ANOMALY_FRAC`` del total, ya no son aisladas sino estructurales
      (p. ej. un intervalo distinto del pedido) y falla.
    """
    paso = _intervalo(interval)
    if not rows:
        raise SchemaError("no hay klines que convertir")
    anchos = {len(r) for r in rows}
    if anchos != {len(KLINE_FIELDS)}:
        raise SchemaError(
            f"klines de ancho {sorted(anchos)}, se esperaba {len(KLINE_FIELDS)}"
        )

    crudo = pd.DataFrame([list(r) for r in rows], columns=list(KLINE_FIELDS))
    apertura = crudo["open_time"].astype("int64")
    cierre = crudo["close_time"].astype("int64")
    for nombre, valores in (("open_time", apertura), ("close_time", cierre)):
        fuera = valores[(valores < EPOCH_MS_MIN) | (valores > EPOCH_MS_MAX)]
        if not fuera.empty:
            raise SchemaError(
                f"{nombre} fuera del rango de un epoch en milisegundos "
                f"({fuera.head(3).tolist()}): unidad de tiempo distinta de la "
                "esperada"
            )

    paso_ms = _ms(pd.Timestamp(0, tz=UTC) + paso)
    cierre_teorico = apertura + paso_ms - 1
    desvio = cierre - cierre_teorico
    anomalas = int((desvio != 0).sum())
    if anomalas > MAX_ANOMALY_ROWS and anomalas > MAX_ANOMALY_FRAC * len(crudo):
        raise SchemaError(
            f"{anomalas} de {len(crudo)} klines con close_time != open_time + "
            f"{interval} - 1ms (primeras: "
            f"{crudo.loc[desvio != 0, 'open_time'].head(5).tolist()}). Son "
            "demasiadas para ser caidas aisladas del venue: intervalo o formato "
            "distintos de los esperados"
        )

    # ASSUMPTION: timestamp = max(cierre teorico, close_time real + 1 ms). Es el
    # primer instante en que la fila entera se conoce, redondeado hacia arriba a
    # la rejilla. Si el venue cerro antes (caida), el teorico sigue siendo
    # point-in-time y no saca la barra de la rejilla. Si cerrara despues, el
    # teorico seria lookahead y manda el real.
    sello_ms = np.maximum(apertura + paso_ms, cierre + 1)
    ahora_ms = int(pd.Timestamp(now).tz_localize(UTC).value // 1_000_000)
    cerradas = sello_ms <= ahora_ms
    crudo = crudo.loc[cerradas].reset_index(drop=True)
    if crudo.empty:
        raise SchemaError("ninguna kline cerrada antes de `now`")

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                sello_ms[cerradas].to_numpy(), unit="ms", utc=True
            ),
            "open_time": pd.to_datetime(
                crudo["open_time"].astype("int64"), unit="ms", utc=True
            ),
            "close_time_desvio_ms": desvio[cerradas].to_numpy(dtype="int64"),
        }
    )
    for campo in (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ):
        frame[campo] = crudo[campo].astype("float64")
    frame["n_trades"] = crudo["n_trades"].astype("int64")
    frame["symbol"] = symbol
    return frame.loc[:, list(CSV_COLUMNS)]


def write_klines_csv(frame: pd.DataFrame, path: Path) -> None:
    """CSV determinista: el mismo frame produce los mismos bytes y el mismo SHA256.

    ``mtime=0`` porque gzip graba la hora de escritura en el encabezado; sin eso,
    re-generar un dataset identico cambiaria su huella.
    """
    compresion: dict[str, Any] | None = (
        {"method": "gzip", "mtime": 0} if path.suffix == ".gz" else None
    )
    frame.to_csv(path, index=False, compression=compresion, lineterminator="\n")


def build_dataset(
    get_json: GetJson,
    *,
    symbol: str,
    interval: str,
    start: pd.Timestamp,
    now: np.datetime64,
    out_path: Path,
    script: str,
    url: str = PUBLIC_KLINES_URL,
) -> DatasetManifest:
    """Descarga, escribe el CSV y su manifest. Devuelve el manifest escrito."""
    fin = pd.Timestamp(now).tz_localize(UTC)
    filas = fetch_klines(
        get_json, symbol=symbol, interval=interval, start=start, end=fin, url=url
    )
    frame = klines_to_frame(filas, symbol=symbol, interval=interval, now=now)
    write_klines_csv(frame, out_path)
    manifest = DatasetManifest(
        file=out_path.name,
        sha256=sha256_of(out_path),
        rows=len(frame),
        first_timestamp=frame["timestamp"].iloc[0].isoformat(),
        last_timestamp=frame["timestamp"].iloc[-1].isoformat(),
        symbol=symbol,
        interval=interval,
        source=f"{url}?symbol={symbol}&interval={interval}",
        downloaded_at=fin.isoformat(),
        script=script,
        notes=(
            "timestamp = cierre de la barra: max(open_time + intervalo, "
            "close_time + 1ms)",
            "close_time_desvio_ms != 0 marca klines cerradas antes de tiempo "
            "por el venue (caidas, mantenimientos); se conservan sin reparar",
            "open_time se conserva como columna para trazabilidad",
            "precios sin ajustar, tal como los publica el venue",
            "barra en curso descartada: solo klines cerradas antes de downloaded_at",
        ),
    )
    write_manifest(manifest, manifest_path_for(out_path))
    return manifest
