"""Nivel articulo y agregado por lote de FinBERT: oraculos a mano."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.errors import DataValidationError
from data.manifest import DatasetManifest, ManifestError, sha256_of, write_manifest
from features.finbert import ScoredArticle
from features.finbert_dataset import (
    aggregate_by_batch,
    append_articles,
    attach_finbert,
    read_articles,
    read_finbert_aggregates,
    repair_tail,
    write_aggregates,
)
from tests.features.news_helpers import grilla

T = np.datetime64("2020-03-01T00:00:00", "ns")
M = np.timedelta64(15, "m")


def art(k: int, rid: str, pos: float, neg: float) -> ScoredArticle:
    return ScoredArticle(T + k * M, rid, f"key{rid}", pos, neg)


def test_agregado_por_lote_a_mano(tmp_path: Path) -> None:
    ruta = tmp_path / "a.jsonl"
    append_articles(
        ruta,
        [art(0, "a", 0.75, 0.125), art(0, "b", 0.25, 0.5), art(2, "c", 0.5, 0.25)],
    )
    frame = aggregate_by_batch(ruta)
    assert frame["label"].tolist() == ["2020-03-01T00:00:00Z", "2020-03-01T00:30:00Z"]
    assert frame["n_scored"].tolist() == [2, 1]
    # (0.75 - 0.125) + (0.25 - 0.5) = 0.375 ; 0.5 - 0.25 = 0.25 (exactos en binario)
    assert frame["score_sum"].tolist() == [0.375, 0.25]


def test_el_agregado_rechaza_rotulos_desordenados(tmp_path: Path) -> None:
    ruta = tmp_path / "a.jsonl"
    append_articles(ruta, [art(2, "a", 0.5, 0.25), art(0, "b", 0.5, 0.25)])
    with pytest.raises(DataValidationError, match="desordenados"):
        aggregate_by_batch(ruta)


def test_repair_tail_descarta_la_linea_rota_y_el_lote_a_medias(tmp_path: Path) -> None:
    ruta = tmp_path / "a.jsonl"
    append_articles(ruta, [art(0, "a", 0.5, 0.25), art(1, "b", 0.5, 0.25)])
    append_articles(ruta, [art(1, "c", 0.5, 0.25)])
    with ruta.open("a") as f:
        f.write('{"label": "2020-03-01T00:15:0')  # caida a mitad de la linea
    ultimo = repair_tail(ruta)
    # El lote 1 pudo quedar a medias: se descarta entero y se vuelve a puntuar.
    assert ultimo == T
    assert [a["id"] for a in read_articles(ruta)] == ["a"]


def test_repair_tail_sin_archivo_o_intacto(tmp_path: Path) -> None:
    ruta = tmp_path / "a.jsonl"
    assert repair_tail(ruta) is None
    append_articles(ruta, [art(3, "a", 0.5, 0.25)])
    assert repair_tail(ruta) == T + 3 * M
    assert len(list(read_articles(ruta))) == 1


def test_una_linea_rota_en_el_medio_se_rechaza(tmp_path: Path) -> None:
    ruta = tmp_path / "a.jsonl"
    ruta.write_text('{"label": "x"}\n{roto\n{"label": "y"}\n', encoding="utf-8")
    with pytest.raises(DataValidationError, match="a.jsonl:2: linea truncada"):
        list(read_articles(ruta))


def test_el_csv_es_determinista_y_legible(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {"label": ["2020-01-01T00:15:00Z"], "n_scored": [3], "score_sum": [0.1]}
    )
    a, b = tmp_path / "a.csv.gz", tmp_path / "b.csv.gz"
    assert write_aggregates(frame, a) == write_aggregates(frame, b) == 1
    assert a.read_bytes() == b.read_bytes()
    texto = gzip.decompress(a.read_bytes()).decode()
    assert texto == "label,n_scored,score_sum\n2020-01-01T00:15:00Z,3,0.1\n"


def test_attach_finbert_rellena_con_cero_y_rechaza_rotulos_ajenos() -> None:
    frame = grilla(1)
    fin = pd.DataFrame(
        {"label": ["2020-01-01T00:15:00Z"], "n_scored": [2], "score_sum": [0.5]}
    )
    unido = attach_finbert(frame, fin)
    assert unido["finbert_n"].sum() == 2
    assert unido.loc[1, "finbert_sum"] == 0.5
    assert unido.loc[0, "finbert_n"] == 0
    assert len(unido) == len(frame)
    ajeno = fin.assign(label=["2021-01-01T00:15:00Z"])
    with pytest.raises(DataValidationError, match="fuera de la rejilla"):
        attach_finbert(frame, ajeno)


def test_read_finbert_aggregates_verifica_el_sha256(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {"label": ["2020-01-01T00:15:00Z"], "n_scored": [3], "score_sum": [0.1]}
    )
    csv = tmp_path / "f.csv.gz"
    write_aggregates(frame, csv)
    manifest = DatasetManifest(
        "f.csv.gz", sha256_of(csv), 1, "a", "b", "X", "15m", "s", "2026-10-26", "x"
    )
    ruta = tmp_path / "f.csv.gz.manifest.json"
    write_manifest(manifest, ruta)
    assert read_finbert_aggregates(ruta)["n_scored"].tolist() == [3]
    csv.write_bytes(csv.read_bytes() + b"x")
    with pytest.raises(ManifestError, match="SHA256"):
        read_finbert_aggregates(ruta)
