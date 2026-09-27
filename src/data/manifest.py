"""Manifest de un dataset versionado: que archivo es, de donde salio y su huella.

Un dataset commiteado sin manifest es un archivo que nadie puede auditar: no se
sabe de donde vino, cuando se bajo ni si alguien lo toco despues. El manifest
fija las tres cosas, y :func:`verify_manifest` se niega a devolver la ruta si
el SHA256 no coincide. La carga pasa por ahi, asi que un CSV editado a mano no
llega nunca a una ``BarSeries``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from data.errors import DataError

MANIFEST_SUFFIX = ".manifest.json"
_BLOQUE = 1 << 20


class ManifestError(DataError):
    """El dataset no corresponde a su manifest."""


@dataclass(frozen=True)
class DatasetManifest:
    """Procedencia de un archivo de datos versionado.

    ``file`` es relativo al directorio del manifest, para que el par se pueda
    mover junto sin reescribirlo.
    """

    file: str
    sha256: str
    rows: int
    first_timestamp: str
    last_timestamp: str
    symbol: str
    interval: str
    source: str
    downloaded_at: str
    script: str
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        datos = asdict(self)
        datos["notes"] = list(self.notes)
        return datos

    @classmethod
    def from_dict(cls, datos: dict[str, object]) -> DatasetManifest:
        completo = dict(datos)
        notas = completo.get("notes", ())
        if not isinstance(notas, list | tuple):
            raise ManifestError(f"notes debe ser una lista, es {type(notas).__name__}")
        completo["notes"] = tuple(str(n) for n in notas)
        return cls(**completo)  # type: ignore[arg-type]


def sha256_of(path: Path) -> str:
    """SHA256 del contenido del archivo, leido por bloques."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for bloque in iter(lambda: handle.read(_BLOQUE), b""):
            digest.update(bloque)
    return digest.hexdigest()


def manifest_path_for(data_path: Path) -> Path:
    return data_path.with_name(data_path.name + MANIFEST_SUFFIX)


def write_manifest(manifest: DatasetManifest, path: Path) -> None:
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_manifest(path: Path) -> DatasetManifest:
    return DatasetManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


def verify_manifest(manifest_path: Path) -> Path:
    """Devuelve la ruta del dataset solo si su SHA256 coincide con el manifest."""
    manifest = read_manifest(manifest_path)
    data_path = manifest_path.parent / manifest.file
    if not data_path.exists():
        raise ManifestError(f"el manifest apunta a un archivo inexistente: {data_path}")
    real = sha256_of(data_path)
    if real != manifest.sha256:
        raise ManifestError(
            f"{data_path.name}: SHA256 {real} no coincide con el del manifest "
            f"{manifest.sha256}. El archivo cambio despues de versionarse."
        )
    return data_path
