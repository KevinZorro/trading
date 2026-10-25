"""Agregados por lote: oraculos a mano, cadena de SHA256 y lote sano."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.errors import DataValidationError, SchemaError
from data.manifest import DatasetManifest, ManifestError, sha256_of, write_manifest
from data.news import parse_batch
from data.news_aggregate import (
    aggregate_batch,
    healthy_batches,
    iter_raw_batches,
    read_aggregates,
    write_aggregates,
)

FIXTURE = Path(__file__).parent / "news_fixture" / "news-2026-10-20.jsonl"


def crudo(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Un directorio con la forma de la descarga: p1/news-*.jsonl y sus SHA256."""
    destino = tmp_path / "raw" / "p1"
    destino.mkdir(parents=True)
    shutil.copy(FIXTURE, destino / FIXTURE.name)
    sha = hashlib.sha256((destino / FIXTURE.name).read_bytes()).hexdigest()
    return tmp_path / "raw", {f"p1/{FIXTURE.name}": sha}


def lotes_fixture() -> list[dict[str, object]]:
    return [json.loads(x) for x in FIXTURE.read_text().splitlines()]


def test_el_lote_ok_se_agrega_a_mano() -> None:
    fila = aggregate_batch(parse_batch(lotes_fixture()[1]))
    assert fila["label"] == "2026-10-20T01:15:00Z"
    assert (fila["rows_total"], fila["rows_theme"], fila["n_titled"]) == (5, 3, 3)
    # 4.25531914893617 - 2.11502782931354 - 5.12333965844402
    assert float(str(fila["tone_sum"])) == pytest.approx(-2.98304833882139)
    # Last-Modified 01:03:41 contra rotulo 01:15:00: 11 min 19 s antes.
    assert fila["last_modified_offset_s"] == -679


def test_un_lote_missing_solo_trae_rotulo_y_estado() -> None:
    fila = aggregate_batch(parse_batch(lotes_fixture()[2]))
    assert fila["status"] == "missing"
    assert all(v is None for k, v in fila.items() if k not in ("label", "status"))


def test_el_agregado_es_determinista_y_verifica_contra_su_manifest(
    tmp_path: Path,
) -> None:
    raw, shas = crudo(tmp_path)
    a, b = tmp_path / "a.csv.gz", tmp_path / "b.csv.gz"
    assert write_aggregates(iter_raw_batches(raw, shas), a) == 2
    write_aggregates(iter_raw_batches(raw, shas), b)
    assert a.read_bytes() == b.read_bytes()
    manifest = DatasetManifest(
        "a.csv.gz", sha256_of(a), 2, "x", "y", "ECON_BITCOIN", "15m", "s", "d", "s"
    )
    write_manifest(manifest, tmp_path / "a.csv.gz.manifest.json")
    frame = read_aggregates(tmp_path / "a.csv.gz.manifest.json")
    assert list(frame["status"]) == ["ok", "missing"]
    assert frame["label"].iloc[0] == pd.Timestamp("2026-10-20T01:15:00")
    a.write_bytes(gzip.compress(b"label\n"))
    with pytest.raises(ManifestError, match="no coincide"):
        read_aggregates(tmp_path / "a.csv.gz.manifest.json")


def test_un_archivo_alterado_detiene_la_lectura(tmp_path: Path) -> None:
    raw, shas = crudo(tmp_path)
    ruta = raw / "p1" / FIXTURE.name
    ruta.write_text(ruta.read_text() + "\n")
    with pytest.raises(DataValidationError, match="SHA256"):
        list(iter_raw_batches(raw, shas))


def test_un_archivo_fuera_del_manifest_detiene_la_lectura(tmp_path: Path) -> None:
    raw, _ = crudo(tmp_path)
    with pytest.raises(SchemaError, match="no figura en el manifest"):
        list(iter_raw_batches(raw, {}))


def test_un_archivo_del_manifest_ausente_se_nota(tmp_path: Path) -> None:
    raw, shas = crudo(tmp_path)
    with pytest.raises(SchemaError, match="ausentes"):
        list(iter_raw_batches(raw, {**shas, "p2/news-2026-10-21.jsonl": "0" * 64}))


def test_una_linea_truncada_no_se_descarta_en_silencio(tmp_path: Path) -> None:
    raw, _ = crudo(tmp_path)
    ruta = raw / "p1" / FIXTURE.name
    ruta.write_text(ruta.read_text() + '{"type": "batch", "lab\n')
    sha = hashlib.sha256(ruta.read_bytes()).hexdigest()
    with pytest.raises(DataValidationError, match="truncada"):
        list(iter_raw_batches(raw, {f"p1/{FIXTURE.name}": sha}))


def rejilla(total: list[int | None]) -> pd.DataFrame:
    """Una fila por 15 min; ``None`` es un lote missing."""
    n = len(total)
    return pd.DataFrame(
        {
            "label": pd.date_range("2020-01-01", periods=n, freq="15min"),
            "status": ["missing" if t is None else "ok" for t in total],
            "rows_total": [np.nan if t is None else float(t) for t in total],
        }
    )


def test_lote_sano_contra_el_25_por_ciento_de_la_mediana_previa() -> None:
    # Mediana previa 100: umbral 25. 25 pasa (>=), 24 no, el missing no.
    sano = healthy_batches(rejilla([100, 100, 100, 25, 24, None, 100]))
    assert list(sano) == [True, True, True, True, False, False, True]


def test_la_mediana_no_incluye_al_lote_mismo() -> None:
    # Si incluyera al propio 40: mediana de [100, 40] = 70, umbral 17,5 -> 40 pasa
    # igual. Con 20: mediana [100, 20] = 60, umbral 15 -> pasaria; sin el lote, el
    # umbral es 25 y 20 no pasa.
    assert list(healthy_batches(rejilla([100, 20]))) == [True, False]


def test_el_primer_lote_ok_sin_historia_cuenta_como_sano() -> None:
    assert list(healthy_batches(rejilla([None, 3, 100]))) == [False, True, True]


def test_la_rejilla_con_huecos_se_rechaza() -> None:
    f = rejilla([100, 100, 100]).drop(index=1)
    with pytest.raises(DataValidationError, match="rejilla"):
        healthy_batches(f)
