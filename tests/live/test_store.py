"""SnapshotStore: append con fsync, cabecera con el SHA del ADR, heartbeat atomico."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import pytest

from live import store as store_mod
from live.store import SnapshotStore, StoreError, file_sha256

CABECERA = {"type": "header", "adr_sha256": "abc"}


def lineas(ruta: Path) -> list[str]:
    return ruta.read_text().splitlines()


def test_cabecera_solo_en_la_primera_linea(tmp_path: Path) -> None:
    s = SnapshotStore(tmp_path, CABECERA)
    s.append_snapshot("2026-10-17", {"n": 1})
    s.append_snapshot("2026-10-17", {"n": 2})
    s.append_snapshot("2026-10-18", {"n": 3})
    uno = lineas(tmp_path / "snapshots-2026-10-17.jsonl")
    assert [json.loads(x) for x in uno] == [CABECERA, {"n": 1}, {"n": 2}]
    # un archivo por dia UTC, cada uno con su cabecera
    assert json.loads(lineas(tmp_path / "snapshots-2026-10-18.jsonl")[0]) == CABECERA


def test_fsync_en_cada_registro(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llamadas: list[int] = []
    real = os.fsync
    monkeypatch.setattr(
        store_mod.os, "fsync", lambda fd: (llamadas.append(fd), real(fd))[1]
    )
    s = SnapshotStore(tmp_path, CABECERA)
    s.append_snapshot("2026-10-17", {"n": 1})
    primero = len(llamadas)
    s.append_snapshot("2026-10-17", {"n": 2})
    assert primero >= 2  # archivo y directorio (archivo nuevo)
    assert len(llamadas) - primero == 1  # solo el archivo


def test_linea_rota_por_una_caida_se_separa(tmp_path: Path) -> None:
    s = SnapshotStore(tmp_path, CABECERA)
    s.append_snapshot("2026-10-17", {"n": 1})
    ruta = tmp_path / "snapshots-2026-10-17.jsonl"
    with open(ruta, "ab") as f:
        f.write(b'{"n": 2, "rot')  # SIGKILL a mitad de linea
    s.append_snapshot("2026-10-17", {"n": 3})
    ultimas = lineas(ruta)[-2:]
    assert ultimas[0] == '{"n": 2, "rot'
    assert json.loads(ultimas[1]) == {"n": 3}


def test_libro_miembros_gzip_concatenados(tmp_path: Path) -> None:
    s = SnapshotStore(tmp_path, CABECERA)
    s.append_book("2026-10-17", {"id": 1})
    s.append_book("2026-10-17", {"id": 2})
    texto = gzip.decompress((tmp_path / "book-2026-10-17.jsonl.gz").read_bytes())
    assert [json.loads(x)["id"] for x in texto.decode().splitlines()] == [1, 2]


def test_nan_se_rechaza(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Out of range float"):
        SnapshotStore(tmp_path, CABECERA).append_snapshot(
            "2026-10-17", {"x": float("nan")}
        )


def test_heartbeat_atomico(tmp_path: Path) -> None:
    s = SnapshotStore(tmp_path, CABECERA)
    s.write_heartbeat({"at": "a"})
    s.write_heartbeat({"at": "b"})
    assert json.loads((tmp_path / "heartbeat.json").read_text()) == {"at": "b"}
    assert not (tmp_path / "heartbeat.json.tmp").exists()


def test_sha256_del_archivo(tmp_path: Path) -> None:
    archivo = tmp_path / "adr.md"
    archivo.write_bytes(b"abc")
    # SHA256("abc"), vector estandar de FIPS 180-2
    assert file_sha256(archivo) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    with pytest.raises(StoreError, match="no se puede leer"):
        file_sha256(tmp_path / "no_existe.md")
