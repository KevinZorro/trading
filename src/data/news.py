"""Esquema, validacion y lectura de las noticias grabadas (Etapa 4, PR N1).

Lee los ``news-AAAA-MM-DD.jsonl`` que escribe ``live.news.NewsRecorder`` y que, mas
adelante, escribira la descarga historica con el mismo formato: historico y vivo se
leen con el mismo codigo. Este modulo no parsea el GKG (eso es ``data.gdelt``) ni
calcula features (eso es la alineacion de N4): da el esquema tipado, rechaza lo
inconsistente y concentra en **un solo lugar** la regla anti-lookahead.

Regla de la Etapa 4 (``CLAUDE.md``, convencion temporal): un lote es usable en la
decision de la barra ``t`` si y solo si ``label + latencia <= close_time[t]``. La
latencia se suma al rotulo, nunca se resta a ``close_time``, y no puede ser negativa
(una latencia negativa seria usar el lote antes de que exista).

ASSUMPTION: ``pub_time`` de un articulo es el rotulo de su lote. GDELT no da la hora
de publicacion original sino la de deteccion, que es posterior: sesgo conservador,
documentado como limitacion del principio 2. Medido el 2026-10-20 (L1), el archivo
aparece hasta ~11 min *antes* de su rotulo, por eso la latencia por defecto es de
30 min y no de 15.
ASSUMPTION: la identidad de un articulo es ``(record_id, url)``, no ``record_id``
solo. Medido sobre el historico 2016-08 a 2026-09: GDELT repite el ``record_id`` dentro
de un mismo lote para articulos distintos (329 de 2,74 millones, todos con URL
diferente). Dos filas con el mismo par si son un duplicado y se rechazan.
ASSUMPTION: de los siete campos de ``V2Tone`` solo se conserva el tono neto (el
primero). Los demas se pueden releer del registro crudo si una version futura los
necesita; agregarlos despues de ver resultados sera una decision registrada.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from data.errors import DataValidationError, SchemaError
from data.gdelt import LOTE_MINUTOS
from data.schema import TimeArray

NEWS_SCHEMA = 1
DEFAULT_LATENCY = np.timedelta64(30, "m")
STATUSES: tuple[str, ...] = ("ok", "missing")
_PASO = np.timedelta64(LOTE_MINUTOS, "m")
_MD5 = re.compile(r"[0-9a-f]{32}")


@dataclass(frozen=True)
class NewsEvent:
    """Un articulo del tema. Sin texto del articulo (derechos de autor)."""

    record_id: str
    label: np.datetime64  # rotulo del lote; ver ASSUMPTION sobre pub_time
    domain: str
    url: str
    tone: float  # V2Tone[0]: positivo menos negativo, segun el diccionario de GDELT
    title: str | None


@dataclass(frozen=True)
class NewsBatch:
    """Un lote de 15 min. ``missing`` es un hueco explicito: sin filas, sin conteos."""

    label: np.datetime64
    status: str
    rows_total: int | None  # filas del GKG completo; el denominador de news_share
    rows_malformed: int | None
    events: tuple[NewsEvent, ...]
    md5: str | None
    received_at: np.datetime64 | None  # hora de recepcion propia (solo grabado en vivo)
    last_modified: str | None  # cabecera HTTP tal cual, para medir la latencia real

    def available_at(self, latency: np.timedelta64 = DEFAULT_LATENCY) -> np.datetime64:
        """Instante desde el cual el lote es usable: ``label + latencia``."""
        return self.label + _check_latency(latency)


def _check_latency(latency: np.timedelta64) -> np.timedelta64:
    if latency < np.timedelta64(0, "s"):
        raise DataValidationError(
            f"la latencia no puede ser negativa ({latency}): usaria el lote antes de "
            "que exista"
        )
    return latency


@dataclass(frozen=True)
class NewsDataset:
    """Lotes ordenados por rotulo, estrictamente crecientes y alineados a 15 min."""

    batches: tuple[NewsBatch, ...]
    torn_lines: int = 0  # lineas truncadas por una caida del grabador, contadas
    _labels: TimeArray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        labels = np.array([b.label for b in self.batches], dtype="datetime64[ns]")
        for b in self.batches:
            _validar_lote(b)
        if len(labels) > 1 and not (np.diff(labels) > np.timedelta64(0, "ns")).all():
            malo = int(np.argmax(np.diff(labels) <= np.timedelta64(0, "ns")))
            raise DataValidationError(
                f"rotulos duplicados o desordenados en {labels[malo + 1]}"
            )
        claves = [(e.record_id, e.url) for b in self.batches for e in b.events]
        if len(set(claves)) != len(claves):
            raise DataValidationError("record_id y URL repetidos entre lotes")
        labels.flags.writeable = False
        object.__setattr__(self, "_labels", labels)

    def usable(
        self, close_time: np.datetime64, latency: np.timedelta64 = DEFAULT_LATENCY
    ) -> tuple[NewsBatch, ...]:
        """Lotes con ``label + latencia <= close_time``: lo que la barra puede ver."""
        disponibles = self._labels + _check_latency(latency)
        k = int(np.searchsorted(disponibles, np.datetime64(close_time, "ns"), "right"))
        return self.batches[:k]


def _validar_lote(b: NewsBatch) -> None:
    if b.status not in STATUSES:
        raise SchemaError(f"status desconocido {b.status!r} en el lote {b.label}")
    if np.datetime64(b.label, "ns") != np.datetime64(b.label, "m") or (
        np.datetime64(b.label, "m").astype("int64") % LOTE_MINUTOS != 0
    ):
        raise DataValidationError(f"el rotulo {b.label} no es multiplo de 15 min UTC")
    if b.status == "missing":
        if b.events or b.rows_total is not None or b.md5 is not None:
            raise DataValidationError(f"el lote missing {b.label} trae datos")
        return
    if b.rows_total is None or b.rows_malformed is None or b.md5 is None:
        raise SchemaError(f"al lote ok {b.label} le faltan conteos o md5")
    if not _MD5.fullmatch(b.md5):
        raise DataValidationError(f"md5 invalido en el lote {b.label}: {b.md5!r}")
    if b.rows_malformed < 0 or b.rows_malformed + len(b.events) > b.rows_total:
        raise DataValidationError(
            f"conteos incoherentes en {b.label}: total={b.rows_total}, "
            f"malformadas={b.rows_malformed}, del tema={len(b.events)}"
        )
    for e in b.events:
        if e.label != b.label:
            raise DataValidationError(f"{e.record_id} no es del lote {b.label}")
        if not np.isfinite(e.tone):
            raise DataValidationError(f"tono no finito en {e.record_id}")


def _ts(texto: object) -> np.datetime64:
    if not isinstance(texto, str):
        raise SchemaError(f"se esperaba una fecha ISO, llego {texto!r}")
    try:
        return np.datetime64(texto.rstrip("Z"), "ns")
    except ValueError as exc:
        raise SchemaError(f"fecha ISO invalida {texto!r}") from exc


def _campo(r: Mapping[str, Any], nombre: str) -> Any:
    if nombre not in r:
        raise SchemaError(f"al registro del lote {r.get('label')} le falta {nombre!r}")
    return r[nombre]


def parse_batch(registro: Mapping[str, Any]) -> NewsBatch:
    """Un registro ``type: batch`` del grabador -> :class:`NewsBatch` (sin validar)."""
    if _campo(registro, "schema") != NEWS_SCHEMA:
        raise SchemaError(
            f"schema {registro['schema']!r} no soportado (se espera {NEWS_SCHEMA})"
        )
    label = _ts(_campo(registro, "label"))
    estado = _campo(registro, "status")
    if estado == "missing":
        return NewsBatch(label, "missing", None, None, (), None, None, None)
    eventos = tuple(
        NewsEvent(
            str(_campo(a, "id")),
            label,
            str(_campo(a, "domain")),
            str(_campo(a, "url")),
            float(_campo(a, "tone")[0]),
            _campo(a, "title"),
        )
        for a in _campo(registro, "articles")
    )
    if _campo(registro, "rows_theme") != len(eventos):
        raise DataValidationError(f"rows_theme no coincide con articles en {label}")
    recibido = registro.get("received_at")
    return NewsBatch(
        label,
        estado,
        _campo(registro, "rows_total"),
        _campo(registro, "rows_malformed"),
        eventos,
        _campo(registro, "md5"),
        None if recibido is None else _ts(recibido),
        registro.get("last_modified"),
    )


def load_news(directory: Path) -> NewsDataset:
    """Lee todos los ``news-*.jsonl`` de ``directory`` y valida el conjunto.

    Una linea que no es JSON es la escritura truncada por una caida (el grabador
    reintenta ese rotulo en una linea nueva): se cuenta en ``torn_lines``, no se
    descarta en silencio. Un registro que si es JSON pero viola el esquema falla.
    """
    archivos = sorted(directory.glob("news-*.jsonl"))
    if not archivos:
        raise SchemaError(f"no hay news-*.jsonl en {directory}")
    lotes: list[NewsBatch] = []
    rotas = 0
    for ruta in archivos:
        for linea in ruta.read_text(encoding="utf-8").splitlines():
            try:
                dato = json.loads(linea)
            except ValueError:
                rotas += 1
                continue
            if not isinstance(dato, dict) or dato.get("type") not in (
                "header",
                "batch",
            ):
                raise SchemaError(
                    f"registro desconocido en {ruta.name}: {linea[:80]!r}"
                )
            if dato["type"] == "batch":
                lotes.append(parse_batch(dato))
    lotes.sort(key=lambda b: int(b.label.astype("int64")))
    return NewsDataset(tuple(lotes), rotas)
