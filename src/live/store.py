"""Persistencia incremental de la sombra (ADR 0007, seccion 2).

Append con ``fsync`` en cada registro, un archivo por dia UTC. Un proceso que
muere a las tres semanas no se lleva las tres semanas: lo unico que puede
quedar roto es la ultima linea, y el siguiente append la separa con un salto.

La cabecera (primera linea de cada archivo de snapshots) lleva el SHA256 del
ADR 0007: deja verificable que el pre-registro es anterior a los datos.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class StoreError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise StoreError(
            f"no se puede leer {path} para fijar su SHA256: {exc}"
        ) from exc


def _fsync_dir(directorio: Path) -> None:
    fd = os.open(directorio, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _linea(registro: Mapping[str, Any]) -> bytes:
    # allow_nan=False: un NaN en un registro es un bug, no un dato.
    return (json.dumps(registro, ensure_ascii=False, allow_nan=False) + "\n").encode()


class SnapshotStore:
    """Archivos ``snapshots-*.jsonl``, ``book-*.jsonl.gz`` y ``heartbeat.json``."""

    def __init__(self, data_dir: Path, header: Mapping[str, Any]) -> None:
        self._dir = data_dir
        self._header = dict(header)
        data_dir.mkdir(parents=True, exist_ok=True)

    def _append(
        self, ruta: Path, datos: bytes, *, cabecera: bytes = b"", lineas: bool = True
    ) -> None:
        nuevo = not ruta.exists()
        separar = False
        if lineas and not nuevo and ruta.stat().st_size > 0:
            with open(ruta, "rb") as lectura:
                lectura.seek(-1, os.SEEK_END)
                separar = lectura.read(1) != b"\n"  # linea rota por una caida
        with open(ruta, "ab") as f:
            f.write((b"\n" if separar else b"") + cabecera + datos)
            f.flush()
            os.fsync(f.fileno())
        if nuevo:
            _fsync_dir(self._dir)

    def append_snapshot(self, day: str, registro: Mapping[str, Any]) -> None:
        ruta = self._dir / f"snapshots-{day}.jsonl"
        cabecera = _linea(self._header) if not ruta.exists() else b""
        self._append(ruta, _linea(registro), cabecera=cabecera)

    def append_book(self, day: str, registro: Mapping[str, Any]) -> None:
        """Cada registro es un miembro gzip: la concatenacion sigue siendo valida."""
        ruta = self._dir / f"book-{day}.jsonl.gz"
        self._append(ruta, gzip.compress(_linea(registro), mtime=0), lineas=False)

    def write_heartbeat(self, estado: Mapping[str, Any]) -> None:
        """Atomico: el watchdog nunca lee un archivo a medio escribir."""
        destino = self._dir / "heartbeat.json"
        tmp = self._dir / "heartbeat.json.tmp"
        with open(tmp, "wb") as f:
            f.write(_linea(estado))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, destino)
        _fsync_dir(self._dir)
