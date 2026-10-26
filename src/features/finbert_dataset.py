"""Lectura, escritura y agregado por lote de los puntajes de FinBERT (PR N2b).

Dos niveles, como el resto de las noticias:

- **Articulo** (``finbert-articles.jsonl``, ~150 MB, va al Release): una linea por
  titular conservado tras la deduplicacion, con su rotulo, ``record_id``, la huella del
  titulo y P(positivo), P(negativo). Se escribe en *append* y se puede reanudar.
- **Lote** (``datasets/gdelt/finbert-batches.csv.gz``, al repo con su manifest): por
  rotulo, cuantos articulos se puntuaron y la suma de P(pos) - P(neg). Es lo unico que
  necesita ``features.news``, que asigna cada lote a una barra con la misma regla de
  latencia que el resto (ADR 0012, 2.3).

El texto del titular no se guarda en ninguno de los dos.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from data.errors import DataValidationError
from data.manifest import verify_manifest
from features.finbert import ScoredArticle

AGGREGATE_COLUMNS: tuple[str, ...] = ("label", "n_scored", "score_sum")


def _rotulo(label: np.datetime64) -> str:
    return str(np.datetime_as_string(label, unit="s")) + "Z"


def append_articles(path: Path, scored: list[ScoredArticle]) -> None:
    """Agrega y fuerza a disco: un proceso que muere no se lleva lo ya puntuado."""
    if not scored:
        return
    with path.open("a", encoding="utf-8", newline="") as f:
        for a in scored:
            f.write(
                json.dumps(
                    {
                        "label": _rotulo(a.label),
                        "id": a.record_id,
                        "key": a.key,
                        "pos": a.pos,
                        "neg": a.neg,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        f.flush()
        os.fsync(f.fileno())


def read_articles(path: Path) -> Iterator[dict[str, object]]:
    """Lineas del nivel articulo. Una linea truncada es un error, no se salta."""
    with path.open(encoding="utf-8") as f:
        for n, linea in enumerate(f, start=1):
            try:
                yield json.loads(linea)
            except ValueError as exc:
                raise DataValidationError(f"{path.name}:{n}: linea truncada") from exc


def repair_tail(path: Path) -> np.datetime64 | None:
    """Deja el archivo en un punto de reanudacion y devuelve su ultimo rotulo.

    Una caida deja la ultima linea a medio escribir. En ese caso se descartan tambien
    las lineas completas de ese rotulo, que pudieron quedar a medias, y el lote se
    vuelve a puntuar. Una linea rota en el medio no se toca: ``read_articles`` la
    rechaza.
    """
    if not path.exists():
        return None
    datos = path.read_bytes()
    if datos and not datos.endswith(b"\n"):
        lineas = datos[: datos.rfind(b"\n") + 1].splitlines(keepends=True)
        if lineas:
            roto = json.loads(lineas[-1])["label"]
            while lineas and json.loads(lineas[-1])["label"] == roto:
                lineas.pop()
        datos = b"".join(lineas)
        path.write_bytes(datos)
    lineas = datos.splitlines()
    if not lineas:
        return None
    return np.datetime64(json.loads(lineas[-1])["label"].removesuffix("Z"), "ns")


def aggregate_by_batch(path: Path) -> pd.DataFrame:
    """Un renglon por rotulo con articulos: ``n_scored`` y ``score_sum``."""
    n: dict[str, int] = {}
    suma: dict[str, float] = {}
    anterior = ""
    for a in read_articles(path):
        rotulo = str(a["label"])
        if rotulo < anterior:
            raise DataValidationError(f"rotulos desordenados en {path.name}: {rotulo}")
        anterior = rotulo
        n[rotulo] = n.get(rotulo, 0) + 1
        suma[rotulo] = suma.get(rotulo, 0.0) + (float(a["pos"]) - float(a["neg"]))  # type: ignore[arg-type]
    return pd.DataFrame(
        {
            "label": list(n),
            "n_scored": list(n.values()),
            "score_sum": list(suma.values()),
        }
    )


def write_aggregates(frame: pd.DataFrame, path: Path) -> int:
    """CSV gzip determinista (sin nombre ni mtime), como ``data.news_aggregate``."""
    with (
        path.open("wb") as crudo,
        gzip.GzipFile(fileobj=crudo, mode="wb", filename="", mtime=0) as gz,
    ):
        texto = io.TextIOWrapper(gz, encoding="utf-8", newline="")
        escritor = csv.writer(texto, lineterminator="\n")
        escritor.writerow(AGGREGATE_COLUMNS)
        for r in frame.itertuples(index=False):
            escritor.writerow((r.label, r.n_scored, repr(float(r.score_sum))))
        texto.flush()
    return len(frame)


def read_finbert_aggregates(manifest_path: Path) -> pd.DataFrame:
    """Agregados verificados contra su manifest (SHA256), como ``read_aggregates``."""
    return pd.read_csv(verify_manifest(manifest_path))


def attach_finbert(frame: pd.DataFrame, finbert: pd.DataFrame) -> pd.DataFrame:
    """Suma ``finbert_n`` y ``finbert_sum`` a los agregados por lote (0 donde no hay).

    Un rotulo de FinBERT que no esta en la rejilla de GDELT es un dataset que no
    corresponde al otro: falla en vez de descartarse.
    """
    fin = finbert.copy()
    fin["label"] = pd.to_datetime(fin["label"], utc=True).dt.tz_localize(None)
    fin["label"] = fin["label"].astype("datetime64[ns]")
    if not fin["label"].isin(frame["label"]).all():
        raise DataValidationError("hay rotulos de FinBERT fuera de la rejilla de GDELT")
    fin = fin.rename(columns={"n_scored": "finbert_n", "score_sum": "finbert_sum"})
    unido = frame.merge(fin, on="label", how="left")
    unido[["finbert_n", "finbert_sum"]] = unido[["finbert_n", "finbert_sum"]].fillna(0)
    return unido
