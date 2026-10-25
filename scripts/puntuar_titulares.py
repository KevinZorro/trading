"""Puntua con FinBERT los titulares del crudo de GDELT y agrega por lote (PR N2b).

PowerShell:  $env:PYTHONPATH = "src"; python scripts/puntuar_titulares.py `
                 --raw RUTA/gdelt-raw --model RUTA/finbert --out RUTA/finbert-salida `
                 --agregados datasets/gdelt

Reanudable: si ``--out`` ya tiene ``finbert-articles.jsonl`` sigue desde el ultimo
rotulo completo, con la deduplicacion restaurada desde lo ya escrito. ``--descargar``
baja el modelo a ``--model`` en la revision pinneada y verifica el SHA256 de cada
archivo.
Ver ``features.finbert`` y ``features.finbert_dataset``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from importlib import metadata
from pathlib import Path

import numpy as np

from data.manifest import DatasetManifest, sha256_of, write_manifest
from data.news_aggregate import iter_raw_batches
from features.finbert import (
    DEDUP_WINDOW,
    MAX_LENGTH,
    MODEL_FILES,
    MODEL_ID,
    MODEL_REVISION,
    DedupWindow,
    FinBertScorer,
    score_articles,
)
from features.finbert_dataset import (
    aggregate_by_batch,
    append_articles,
    read_articles,
    repair_tail,
    write_aggregates,
)

ARTICULOS = "finbert-articles.jsonl"
AGREGADOS = "finbert-batches.csv.gz"
CHUNK = 2000


def descargar(destino: Path) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    for nombre in MODEL_FILES:
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{nombre}"
        with urllib.request.urlopen(url) as r, (destino / nombre).open("wb") as f:
            while bloque := r.read(1 << 20):
                f.write(bloque)


def puntuar(raw: Path, modelo: Path, salida: Path, crudo_manifest: dict) -> None:  # type: ignore[type-arg]
    ruta = salida / ARTICULOS
    ultimo = repair_tail(ruta)
    dedup = DedupWindow()
    if ultimo is not None:
        desde = ultimo - DEDUP_WINDOW
        dedup.restore(
            (np.datetime64(str(a["label"]).removesuffix("Z"), "ns"), str(a["key"]))
            for a in read_articles(ruta)
            if np.datetime64(str(a["label"]).removesuffix("Z"), "ns") >= desde
        )
    scorer = FinBertScorer(modelo)
    cache: dict[str, tuple[float, float]] = {}
    pendientes: list[tuple[np.datetime64, str, str]] = []
    t0, hechos = time.time(), 0

    def volcar() -> None:
        nonlocal hechos
        puntuados = score_articles(pendientes, scorer, dedup, cache)
        append_articles(ruta, puntuados)
        hechos += len(puntuados)
        print(
            f"{pendientes[-1][0]}  {hechos} articulos  "
            f"{hechos / (time.time() - t0):.1f}/s",
            file=sys.stderr,
            flush=True,
        )
        pendientes.clear()

    for lote in iter_raw_batches(raw, crudo_manifest["files"]):
        if ultimo is not None and lote.label <= ultimo:
            continue
        pendientes += [
            (lote.label, e.record_id, e.title) for e in lote.events if e.title
        ]
        if len(pendientes) >= CHUNK:
            volcar()
    if pendientes:
        volcar()


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--agregados", type=Path, required=True)
    p.add_argument("--descargar", action="store_true")
    a = p.parse_args(argv)
    if a.descargar:
        descargar(a.model)
    ruta_raw = a.raw / "MANIFEST.json"
    crudo = json.loads(ruta_raw.read_text(encoding="utf-8"))
    a.out.mkdir(parents=True, exist_ok=True)
    puntuar(a.raw, a.model, a.out, crudo)

    frame = aggregate_by_batch(a.out / ARTICULOS)
    a.agregados.mkdir(parents=True, exist_ok=True)
    destino = a.agregados / AGREGADOS
    filas = write_aggregates(frame, destino)
    articulos = a.out / ARTICULOS
    manifest = DatasetManifest(
        file=AGREGADOS,
        sha256=sha256_of(destino),
        rows=filas,
        first_timestamp=str(frame["label"].iloc[0]),
        last_timestamp=str(frame["label"].iloc[-1]),
        symbol="ECON_BITCOIN",
        interval="15m",
        source=f"https://huggingface.co/{MODEL_ID}",
        downloaded_at=time.strftime("%Y-%m-%d", time.gmtime()),
        script="scripts/puntuar_titulares.py",
        notes=(
            f"modelo {MODEL_ID} revision {MODEL_REVISION}; SHA256 de cada archivo "
            f"en features.finbert.MODEL_FILES (model.safetensors "
            f"{MODEL_FILES['model.safetensors']})",
            f"crudo: MANIFEST.json SHA256 {sha256_of(ruta_raw)}",
            f"nivel articulo: {ARTICULOS} SHA256 {sha256_of(articulos)}, "
            f"{sum(1 for _ in read_articles(articulos))} articulos (va al Release)",
            f"dedup por titulo normalizado, primera aparicion, ventana {DEDUP_WINDOW}",
            f"fp32 en CPU, modo evaluacion, max_length {MAX_LENGTH}, lote fijo",
            f"torch {metadata.version('torch')}, "
            f"transformers {metadata.version('transformers')}",
            "n_scored = articulos conservados del lote; score_sum = suma de "
            "P(positivo) - P(negativo); solo se listan los lotes con articulos",
        ),
    )
    write_manifest(manifest, a.agregados / (AGREGADOS + ".manifest.json"))
    print(f"{filas} lotes -> {destino} ({manifest.sha256})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
