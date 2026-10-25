"""Agregados por lote de 15 min del GKG (Etapa 4, PR N3): lo que B necesita, sin texto.

El nivel articulo pesa ~1,1 GB y no se versiona en el repo. Las siete features del
ADR 0012 solo necesitan, por lote, cuatro conteos y una suma de tonos:

- ``rows_total`` y ``rows_theme``  -> ``news_share``
- ``tone_sum`` / ``rows_theme``    -> ``tone_mean``
- ``n_titled``                     -> ``no_title`` (y el alcance de FinBERT)

Agregar aqui es contar y sumar: no deduplica (ADR 0012, 2.4) ni mira el reloj.
``last_modified_offset_s`` conserva la latencia observada de cada lote para poder
auditarla despues sin volver al crudo.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from collections.abc import Iterator, Mapping
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from data.errors import DataValidationError, SchemaError
from data.gdelt import LOTE_MINUTOS
from data.manifest import verify_manifest
from data.news import NewsBatch, _validar_lote, parse_batch

AGGREGATE_COLUMNS: tuple[str, ...] = (
    "label",
    "status",
    "rows_total",
    "rows_malformed",
    "rows_theme",
    "tone_sum",
    "n_titled",
    "last_modified_offset_s",
)
# ADR 0012, 2.5: un lote sano llega al 25 % de la mediana de los 30 dias previos.
HEALTHY_FRACTION = 0.25
HEALTHY_WINDOW = 30 * 24 * 60 // LOTE_MINUTOS
BATCHES_PER_DAY = 24 * 60 // LOTE_MINUTOS
COVERAGE_MIN_BATCHES = 87  # ADR 0012, 2.5: 87 de 96 lotes sanos (90 %)
_PASO_NS = LOTE_MINUTOS * 60 * 10**9


def _offset_s(batch: NewsBatch) -> int | None:
    """``last_modified - rotulo`` en segundos; negativo = publicado antes del rotulo."""
    if batch.last_modified is None:
        return None
    try:
        publicado = parsedate_to_datetime(batch.last_modified)
    except (TypeError, ValueError) as exc:
        raise SchemaError(
            f"last_modified ilegible en {batch.label}: {batch.last_modified!r}"
        ) from exc
    rotulo = batch.label.astype("datetime64[s]").astype("int64")
    return int(publicado.timestamp()) - int(rotulo)


def aggregate_batch(batch: NewsBatch) -> dict[str, object]:
    """Una fila del dataset. Un lote ``missing`` trae solo rotulo y estado."""
    fila: dict[str, object] = dict.fromkeys(AGGREGATE_COLUMNS)
    fila["label"] = str(np.datetime_as_string(batch.label, unit="s")) + "Z"
    fila["status"] = batch.status
    if batch.status == "ok":
        fila["rows_total"] = batch.rows_total
        fila["rows_malformed"] = batch.rows_malformed
        fila["rows_theme"] = len(batch.events)
        fila["tone_sum"] = repr(float(sum(e.tone for e in batch.events)))
        fila["n_titled"] = sum(1 for e in batch.events if e.title)
        fila["last_modified_offset_s"] = _offset_s(batch)
    return fila


def iter_raw_batches(
    directory: Path, expected_sha256: Mapping[str, str]
) -> Iterator[NewsBatch]:
    """Lotes del crudo en orden de rotulo, verificando el SHA256 de cada archivo.

    ``expected_sha256`` es el ``files`` del ``MANIFEST.json`` de la descarga, con
    rutas relativas a ``directory``. Un archivo ausente del manifest, alterado o
    con una linea truncada detiene la lectura: un agregado sobre datos que no son los
    sellados no sirve de nada.
    """
    archivos = sorted(
        (p for p in directory.glob("p*/news-*.jsonl")), key=lambda p: p.name
    )
    if not archivos:
        raise SchemaError(f"no hay p*/news-*.jsonl en {directory}")
    vistos: set[str] = set()
    anterior: np.datetime64 | None = None
    for ruta in archivos:
        clave = f"{ruta.parent.name}/{ruta.name}"
        esperado = expected_sha256.get(clave)
        if esperado is None:
            raise SchemaError(f"{clave} no figura en el manifest de la descarga")
        cuerpo = ruta.read_bytes()
        real = hashlib.sha256(cuerpo).hexdigest()
        if real != esperado:
            raise DataValidationError(f"{clave}: SHA256 {real} != manifest {esperado}")
        vistos.add(clave)
        for linea in cuerpo.decode("utf-8").splitlines():
            try:
                dato = json.loads(linea)
            except ValueError as exc:
                raise DataValidationError(f"linea truncada en {clave}") from exc
            if dato.get("type") != "batch":
                continue
            lote = parse_batch(dato)
            _validar_lote(lote)
            if anterior is not None and lote.label <= anterior:
                raise DataValidationError(
                    f"rotulo repetido o desordenado en {clave}: {lote.label}"
                )
            anterior = lote.label
            yield lote
    sobran = sorted(set(expected_sha256) - vistos)
    if sobran:
        raise SchemaError(
            f"el manifest lista {len(sobran)} archivos ausentes: {sobran[:3]}"
        )


def write_aggregates(batches: Iterator[NewsBatch], path: Path) -> int:
    """CSV gzip determinista (sin nombre ni mtime en el gzip). Devuelve las filas."""
    n = 0
    with (
        path.open("wb") as crudo,
        gzip.GzipFile(fileobj=crudo, mode="wb", filename="", mtime=0) as gz,
    ):
        texto = io.TextIOWrapper(gz, encoding="utf-8", newline="")
        escritor = csv.DictWriter(texto, AGGREGATE_COLUMNS, lineterminator="\n")
        escritor.writeheader()
        for lote in batches:
            escritor.writerow(aggregate_batch(lote))
            n += 1
        texto.flush()
    return n


def read_aggregates(manifest_path: Path) -> pd.DataFrame:
    """Dataset verificado contra su manifest, con ``label`` como ``datetime64[ns]``."""
    frame = pd.read_csv(verify_manifest(manifest_path))
    frame["label"] = pd.to_datetime(frame["label"], utc=True).dt.tz_localize(None)
    frame["label"] = frame["label"].astype("datetime64[ns]")
    return frame


def healthy_batches(frame: pd.DataFrame) -> npt.NDArray[np.bool_]:
    """Lote sano (ADR 0012, 2.5): ``ok`` y ``rows_total`` >= 25 % de la mediana causal.

    La mediana es la de los lotes ``ok`` de los 30 dias previos, sin el lote mismo. La
    rejilla debe ser completa (un lote por rotulo, ``missing`` incluido), porque la
    ventana se mide en lotes.

    ASSUMPTION: un ``ok`` sin ningun ``ok`` previo (el primer lote del dataset) cuenta
    como sano; no hay con que compararlo.
    """
    pasos = np.diff(frame["label"].to_numpy().astype("int64"))
    if len(pasos) and not (pasos == _PASO_NS).all():
        raise DataValidationError("la rejilla de 15 min tiene huecos o repetidos")
    ok = (frame["status"] == "ok").to_numpy()
    total = frame["rows_total"].where(frame["status"] == "ok")
    mediana = total.shift(1).rolling(HEALTHY_WINDOW, min_periods=1).median()
    umbral = (HEALTHY_FRACTION * mediana).to_numpy()
    con_historia = ~np.isnan(umbral)
    suficiente = ~con_historia | (total.to_numpy() >= np.where(con_historia, umbral, 0))
    return np.asarray(ok & suficiente, dtype=np.bool_)


def bar_healthy_counts(
    frame: pd.DataFrame,
    healthy: npt.NDArray[np.bool_],
    latency: np.timedelta64,
) -> pd.Series:
    """Lotes sanos por barra diaria, con la regla de pertenencia del ADR 0012, 2.3.

    Un lote es de la barra ``t`` si ``close[t-1] < rotulo + latencia <= close[t]``; con
    ``timestamp`` = cierre, la barra se identifica por su cierre. Devuelve el conteo
    indexado por cierre (00:00 UTC). Las barras de los bordes salen incompletas.
    """
    disponible = frame["label"] + pd.Timedelta(latency)
    cierre = (disponible - pd.Timedelta(1, "ns")).dt.floor("D") + pd.Timedelta(1, "D")
    return pd.Series(healthy, index=frame.index).groupby(cierre).sum().astype(int)
