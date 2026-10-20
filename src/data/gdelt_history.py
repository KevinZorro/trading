"""Descarga historica del GKG 2.0 de GDELT (Etapa 4, PR N2). Sin red propia.

Produce los mismos ``news-AAAA-MM-DD.jsonl`` que ``live.news.NewsRecorder``
(un registro ``type: batch`` por lote de 15 min, mismo esquema), para que
``data.news.load_news`` lea historico y vivo con el mismo codigo. Los lotes se
filtran con :func:`data.gdelt.parse_gkg`, el mismo parser que usa el grabador.

Flujo: el ``masterfilelist.txt`` de GDELT (unas 128 MB, se lee en streaming) lista
cada archivo con su tamano y MD5. Cada lote se baja, se verifica contra ese MD5,
se filtra y se descarta; solo se guarda lo filtrado. Es reanudable: al arrancar se
lee que rotulos ya estan escritos y se saltan.

- **Hueco explicito.** Un rotulo del rango que el masterfilelist no lista, o que
  esta listado y da 404, se escribe ``status: "missing"``. Nunca se inventa un lote
  ni se interpola.
- **Fallo transitorio no se escribe.** Un error de red, un MD5 que no coincide o un
  zip roto agotan los reintentos y el rotulo queda sin registro, para que la
  siguiente corrida lo reintente. Escribirlo como ``missing`` taparia un lote que
  existe.
- Escritura incremental con ``fsync`` (append): una caida a mitad de camino no
  pierde lo ya bajado.

ASSUMPTION: el MD5 del masterfilelist es el del archivo tal como se sirve hoy. Los
archivos no se reprocesan (``Last-Modified`` coincide con el rotulo, medido el
2026-10-20), asi que un MD5 distinto es corrupcion de la descarga y se reintenta.
ASSUMPTION: la deduplicacion de sindicacion (:func:`dedupe_first_seen`) no se aplica
al escribir. El archivo conserva todos los articulos, auditable; quien cuenta
articulos dedupica al leer.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import unicodedata
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from data.errors import DataValidationError
from data.gdelt import (
    LOTE_MINUTOS,
    THEME_BITCOIN,
    GkgBatch,
    floor_label,
    parse_gkg,
)
from data.news import NEWS_SCHEMA, NewsEvent

MASTERFILELIST_URL = "https://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
PASO = np.timedelta64(LOTE_MINUTOS, "m")
DEDUPE_WINDOW = np.timedelta64(24, "h")
_GKG = re.compile(r"/(\d{14})\.gkg\.csv\.zip$")
_NO_ALFANUM = re.compile(r"[^0-9a-z]+")


@dataclass(frozen=True)
class MasterEntry:
    label: np.datetime64
    url: str
    md5: str
    size: int


@dataclass(frozen=True)
class HttpResult:
    status: int
    body: bytes = b""
    last_modified: str | None = None


Fetch = Callable[[str], HttpResult]


def _label(texto: str) -> np.datetime64:
    return np.datetime64(
        f"{texto[:4]}-{texto[4:6]}-{texto[6:8]}T{texto[8:10]}:{texto[10:12]}:{texto[12:]}",
        "ns",
    )


def iso(label: np.datetime64) -> str:
    return f"{np.datetime64(label, 'ms')}Z"


def parse_master(
    lineas: Iterable[str], start: np.datetime64, end: np.datetime64
) -> list[MasterEntry]:
    """Entradas GKG con ``start <= rotulo < end``, en el orden del archivo.

    Las lineas export y mentions se ignoran. Una linea GKG que no tiene tres
    campos o cuyo rotulo no es multiplo de 15 min es un archivo corrupto: falla.
    """
    desde, hasta = np.datetime64(start, "ns"), np.datetime64(end, "ns")
    entradas: list[MasterEntry] = []
    for linea in lineas:
        m = _GKG.search(linea.strip())
        if m is None:
            continue
        campos = linea.split()
        if len(campos) != 3 or not campos[0].isdigit() or len(campos[1]) != 32:
            raise DataValidationError(f"linea del masterfilelist ilegible: {linea!r}")
        label = _label(m.group(1))
        if np.datetime64(label, "m").astype("int64") % LOTE_MINUTOS != 0:
            raise DataValidationError(
                f"rotulo fuera de la rejilla de 15 min: {linea!r}"
            )
        if desde <= label < hasta:
            # HTTPS siempre: el masterfilelist lista http://, GDELT sirve ambos.
            url = campos[2].replace("http://", "https://", 1)
            entradas.append(MasterEntry(label, url, campos[1].lower(), int(campos[0])))
    return entradas


def expected_labels(start: np.datetime64, end: np.datetime64) -> list[np.datetime64]:
    primero = floor_label(start)
    if primero < np.datetime64(start, "ns"):
        primero = primero + PASO
    return list(np.arange(primero, np.datetime64(end, "ns"), PASO))


def read_batch(cuerpo: bytes, theme: str = THEME_BITCOIN) -> GkgBatch:
    with zipfile.ZipFile(io.BytesIO(cuerpo)) as z:
        miembros = [n for n in z.namelist() if n.endswith(".gkg.csv")]
        if len(miembros) != 1:
            raise zipfile.BadZipFile(f"se esperaba un .gkg.csv, hay {z.namelist()}")
        with z.open(miembros[0]) as crudo:
            return parse_gkg(
                io.TextIOWrapper(crudo, encoding="utf-8", errors="replace"), theme
            )


def _registro(label: np.datetime64, estado: str, **campos: Any) -> dict[str, Any]:
    return {
        "type": "batch",
        "schema": NEWS_SCHEMA,
        "label": iso(label),
        "status": estado,
        "source": "masterfilelist",
        **campos,
    }


def missing_record(label: np.datetime64, motivo: str) -> dict[str, Any]:
    return _registro(label, "missing", last_error=motivo)


def ok_record(
    entry: MasterEntry, lote: GkgBatch, last_modified: str | None
) -> dict[str, Any]:
    """Mismos campos que el registro ``ok`` de ``live.news``, sin los de recepcion."""
    return _registro(
        entry.label,
        "ok",
        url=entry.url,
        bytes=entry.size,
        md5=entry.md5,
        last_modified=last_modified,
        received_at=None,  # el historico no tiene hora de recepcion propia
        rows_total=lote.rows_total,
        rows_malformed=lote.rows_malformed,
        rows_theme=len(lote.articles),
        articles=[
            {
                "id": a.record_id,
                "domain": a.domain,
                "url": a.url,
                "tone": list(a.tone),
                "title": a.title,
            }
            for a in lote.articles
        ],
    )


def fetch_batch(
    entry: MasterEntry,
    fetch: Fetch,
    *,
    retries: int,
    sleep: Callable[[float], object],
    theme: str = THEME_BITCOIN,
) -> dict[str, Any] | None:
    """Registro del lote, o ``None`` si agoto los reintentos (no se escribe)."""
    for intento in range(retries + 1):
        if intento:
            sleep(2.0**intento)  # 2, 4, 8 s
        try:
            r = fetch(entry.url)
            if r.status == 404:
                return missing_record(entry.label, "http 404")
            if r.status != 200:
                continue
            if hashlib.md5(r.body, usedforsecurity=False).hexdigest() != entry.md5:
                continue
            return ok_record(entry, read_batch(r.body, theme), r.last_modified)
        except (OSError, zipfile.BadZipFile, EOFError):
            continue
    return None


def http_fetch(url: str, timeout_s: float = 120.0) -> HttpResult:  # pragma: no cover
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as r:
            return HttpResult(r.status, r.read(), r.headers.get("Last-Modified"))
    except urllib.error.HTTPError as exc:
        return HttpResult(exc.code)


class HistoryStore:
    """Un ``news-AAAA-MM-DD.jsonl`` por dia UTC del rotulo, con append y ``fsync``."""

    def __init__(self, directory: Path, header_extra: dict[str, Any]) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._dir = directory
        self._header = {
            "type": "header",
            "schema": NEWS_SCHEMA,
            "source": "gdelt-gkg-2.0",
            "origin": "history",
            **header_extra,
        }

    def done(self) -> set[str]:
        """Rotulos ya escritos. Una ultima linea truncada por una caida se ignora."""
        hechos: set[str] = set()
        for ruta in self._dir.glob("news-*.jsonl"):
            for linea in ruta.read_text(encoding="utf-8").splitlines():
                try:
                    dato = json.loads(linea)
                except ValueError:
                    continue
                if isinstance(dato, dict) and dato.get("type") == "batch":
                    hechos.add(str(dato["label"]))
        return hechos

    def append(self, registro: dict[str, Any]) -> None:
        ruta = self._dir / f"news-{str(registro['label'])[:10]}.jsonl"
        nuevo = not ruta.exists()
        with open(ruta, "ab") as f:
            if not nuevo and not _termina_en_salto(ruta):
                f.write(b"\n")  # aisla una linea rota por una caida
            if nuevo:
                f.write(_linea(self._header))
            f.write(_linea(registro))
            f.flush()
            os.fsync(f.fileno())


def _linea(dato: dict[str, Any]) -> bytes:
    return (json.dumps(dato, sort_keys=True, ensure_ascii=False) + "\n").encode()


def _termina_en_salto(ruta: Path) -> bool:
    with open(ruta, "rb") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() == 0:
            return True
        f.seek(-1, os.SEEK_END)
        return f.read(1) == b"\n"


@dataclass(frozen=True)
class DownloadSummary:
    written_ok: int
    written_missing: int
    skipped_done: int
    failed: tuple[str, ...]  # rotulos sin registro tras agotar reintentos


def download(
    entries: Sequence[MasterEntry],
    start: np.datetime64,
    end: np.datetime64,
    store: HistoryStore,
    fetch: Fetch,
    *,
    workers: int = 8,
    retries: int = 3,
    sleep: Callable[[float], object],
    theme: str = THEME_BITCOIN,
) -> DownloadSummary:
    """Baja los lotes pendientes de ``[start, end)`` y escribe los huecos.

    Los rotulos del rango que no estan en ``entries`` se escriben ``missing`` sin
    pedir nada. La escritura es de un solo hilo, en el orden de los rotulos.
    """
    hechos = store.done()
    por_rotulo = {e.label: e for e in entries}
    ok = faltan = saltados = 0
    fallidos: list[str] = []
    pendientes: list[MasterEntry | np.datetime64] = []
    for label in expected_labels(start, end):
        if iso(label) in hechos:
            saltados += 1
        else:
            pendientes.append(por_rotulo.get(label, label))

    def trabajar(item: MasterEntry | np.datetime64) -> dict[str, Any] | None:
        if isinstance(item, MasterEntry):
            return fetch_batch(item, fetch, retries=retries, sleep=sleep, theme=theme)
        return missing_record(item, "ausente del masterfilelist")

    paso = max(workers * 8, 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i in range(0, len(pendientes), paso):
            lote = pendientes[i : i + paso]
            for item, registro in zip(lote, pool.map(trabajar, lote), strict=True):
                if registro is None:
                    assert isinstance(item, MasterEntry)
                    fallidos.append(iso(item.label))
                    continue
                store.append(registro)
                if registro["status"] == "ok":
                    ok += 1
                else:
                    faltan += 1
    return DownloadSummary(ok, faltan, saltados, tuple(fallidos))


def title_key(titulo: str) -> str:
    """Titulo normalizado: NFKC, minusculas, alfanumericos separados por espacio."""
    plano = unicodedata.normalize("NFKC", titulo).casefold()
    return _NO_ALFANUM.sub(" ", plano).strip()


def dedupe_first_seen(
    eventos: Iterable[NewsEvent], window: np.timedelta64 = DEDUPE_WINDOW
) -> Iterator[NewsEvent]:
    """Quita la sindicacion: un titulo normalizado ya visto hace menos de ``window``.

    Se queda con la **primera** aparicion (en el orden recibido, que es el del
    archivo: rotulo creciente). Quedarse con otra seria usar una copia posterior en
    lugar de la que el mercado vio primero, es decir, mirar hacia adelante. La
    ventana se mide desde la ultima aparicion *conservada*, no desde la ultima
    descartada: una nota republicada cada 20 h no se encadena para siempre. Sin
    titulo (o con titulo vacio tras normalizar) no hay clave y el evento se conserva.
    """
    conservado: dict[str, np.datetime64] = {}
    previo: np.datetime64 | None = None
    for e in eventos:
        if previo is not None and e.label < previo:
            raise DataValidationError("dedupe_first_seen exige rotulos no decrecientes")
        previo = e.label
        clave = title_key(e.title) if e.title else ""
        if clave:
            visto = conservado.get(clave)
            if visto is not None and e.label - visto < window:
                continue
            conservado[clave] = e.label
        yield e
