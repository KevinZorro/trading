"""Agrega la descarga cruda de GDELT a un lote por fila y escribe su manifest.

PowerShell:  $env:PYTHONPATH = "src"; python scripts/agregar_gdelt.py `
                 --raw RUTA/gdelt-raw --out datasets/gdelt

``--raw`` es la carpeta con ``MANIFEST.json`` y ``p1`` a ``p4``. Cada archivo crudo se
verifica contra su SHA256 antes de leerse. Ver ``data.news_aggregate``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from data.manifest import DatasetManifest, sha256_of, write_manifest
from data.news_aggregate import iter_raw_batches, write_aggregates

NOMBRE = "gdelt-batches.csv.gz"


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    ruta_manifest = a.raw / "MANIFEST.json"
    crudo = json.loads(ruta_manifest.read_text(encoding="utf-8"))
    a.out.mkdir(parents=True, exist_ok=True)
    destino = a.out / NOMBRE
    filas = write_aggregates(iter_raw_batches(a.raw, crudo["files"]), destino)
    desde, hasta = crudo["range"]
    # "range" del crudo es [inicio, fin): el ultimo rotulo es el fin menos un lote.
    ultimo = (
        str(np.datetime64(f"{hasta}T00:00:00") - np.timedelta64(15, "m")) + "+00:00"
    )
    manifest = DatasetManifest(
        file=NOMBRE,
        sha256=sha256_of(destino),
        rows=filas,
        first_timestamp=f"{desde}T00:00:00+00:00",
        last_timestamp=ultimo,
        symbol=crudo["theme"],
        interval="15m",
        source=crudo["source"],
        downloaded_at=crudo["downloaded_utc"],
        script="scripts/agregar_gdelt.py",
        notes=(
            "un lote de 15 min por fila, rotulado por el nombre del archivo GKG "
            "(label); usable solo desde label + latencia (ADR 0012, 2.2)",
            f"crudo: {len(crudo['files'])} archivos, MANIFEST.json "
            f"SHA256 {sha256_of(ruta_manifest)}; cada archivo se verifico contra el al "
            "agregar",
            "sin deduplicar: rows_theme y tone_sum cuentan filas (ADR 0012, 2.4)",
            "tone_sum = suma de V2Tone[0] de las filas del tema; "
            "n_titled = filas del tema con PAGE_TITLE",
            "last_modified_offset_s = Last-Modified menos label; negativo = publicado "
            "antes del rotulo",
            "los lotes missing se conservan como fila, sin conteos",
            "el nivel articulo (titulos) no se versiona aqui: lo necesita N2b",
        ),
    )
    write_manifest(manifest, a.out / (NOMBRE + ".manifest.json"))
    print(f"{filas} lotes -> {destino} ({manifest.sha256})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
