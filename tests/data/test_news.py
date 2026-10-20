"""Esquema y carga de noticias contra la salida real del grabador (L1)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from data.errors import DataValidationError, SchemaError
from data.news import (
    DEFAULT_LATENCY,
    NEWS_SCHEMA,
    NewsBatch,
    NewsDataset,
    load_news,
    parse_batch,
)

FIXTURE = Path(__file__).parent / "news_fixture"
L0115 = np.datetime64("2026-10-20T01:15:00", "ns")
L0130 = np.datetime64("2026-10-20T01:30:00", "ns")
MD5 = "64deb4972af935ec7c9519474d976749"


def registros() -> list[dict[str, object]]:
    ruta = FIXTURE / "news-2026-10-20.jsonl"
    return [json.loads(x) for x in ruta.read_text().splitlines()]


def lote_ok(label: np.datetime64 = L0115, **cambios: object) -> NewsBatch:
    base = parse_batch(registros()[1])
    campos = {**base.__dict__, "label": label, **cambios}
    return NewsBatch(**campos)  # type: ignore[arg-type]


def test_el_esquema_coincide_con_el_del_grabador() -> None:
    from live.news import SCHEMA

    assert NEWS_SCHEMA == SCHEMA


def test_carga_la_salida_real_del_grabador() -> None:
    ds = load_news(FIXTURE)
    assert [b.label for b in ds.batches] == [L0115, L0130]
    assert [b.status for b in ds.batches] == ["ok", "missing"]
    assert ds.torn_lines == 0
    ok, hueco = ds.batches
    # Oraculo a mano, del lote GDELT 20261020011500: 5 filas, 3 con ECON_BITCOIN.
    assert (ok.rows_total, ok.rows_malformed, len(ok.events)) == (5, 0, 3)
    assert ok.md5 == MD5
    assert ok.received_at == np.datetime64("2026-10-20T01:20:00", "ns")
    assert ok.last_modified == "Tue, 20 Oct 2026 01:03:41 GMT"
    assert [e.record_id for e in ok.events] == [
        "20261020011500-345",
        "20261020011500-443",
        "20261020011500-973",
    ]
    assert ok.events[0].tone == pytest.approx(4.25531914893617)
    assert ok.events[0].domain == "econotimes.com"
    assert ok.events[2].tone == pytest.approx(-5.12333965844402)
    assert all(e.label == L0115 for e in ok.events)
    # Un hueco es un hueco: nada de conteos en cero que parezcan "sin noticias".
    assert (hueco.rows_total, hueco.md5, hueco.events) == (None, None, ())


def test_lineas_truncadas_se_cuentan_y_no_se_descartan_en_silencio(
    tmp_path: Path,
) -> None:
    ruta = tmp_path / "news-2026-10-20.jsonl"
    lineas = (FIXTURE / ruta.name).read_text().splitlines()
    ruta.write_text("\n".join([lineas[0], lineas[1][:60], *lineas[1:]]) + "\n")
    ds = load_news(tmp_path)
    assert ds.torn_lines == 1
    assert len(ds.batches) == 2


def test_une_varios_dias_y_ordena_por_rotulo(tmp_path: Path) -> None:
    ayer = registros()[1] | {"label": "2026-10-19T23:45:00.000Z", "articles": []}
    ayer |= {"rows_theme": 0}
    (tmp_path / "news-2026-10-19.jsonl").write_text(json.dumps(ayer) + "\n")
    shutil.copy(FIXTURE / "news-2026-10-20.jsonl", tmp_path)
    ds = load_news(tmp_path)
    assert [str(b.label)[:16] for b in ds.batches] == [
        "2026-10-19T23:45",
        "2026-10-20T01:15",
        "2026-10-20T01:30",
    ]


def test_directorio_sin_archivos_falla(tmp_path: Path) -> None:
    with pytest.raises(SchemaError, match="no hay news"):
        load_news(tmp_path)


@pytest.mark.parametrize(
    ("cambio", "error", "texto"),
    [
        ({"schema": 2}, SchemaError, "no soportado"),
        ({"rows_theme": 2}, DataValidationError, "rows_theme"),
        ({"label": "no-es-fecha"}, SchemaError, "fecha ISO"),
        ({"label": None}, SchemaError, "fecha ISO"),
    ],
)
def test_registro_invalido_falla(
    cambio: dict[str, object], error: type[Exception], texto: str
) -> None:
    with pytest.raises(error, match=texto):
        parse_batch(registros()[1] | cambio)


def test_registro_sin_campo_obligatorio_falla() -> None:
    r = registros()[1]
    del r["md5"]
    with pytest.raises(SchemaError, match="falta 'md5'"):
        parse_batch(r)


def test_registro_de_tipo_desconocido_falla(tmp_path: Path) -> None:
    (tmp_path / "news-2026-10-20.jsonl").write_text('{"type": "otro"}\n')
    with pytest.raises(SchemaError, match="registro desconocido"):
        load_news(tmp_path)


@pytest.mark.parametrize(
    ("cambios", "texto"),
    [
        ({"label": np.datetime64("2026-10-20T01:20:00", "ns")}, "multiplo de 15"),
        ({"label": np.datetime64("2026-10-20T01:15:30", "ns")}, "multiplo de 15"),
        ({"md5": "xyz"}, "md5 invalido"),
        ({"rows_total": 2}, "conteos incoherentes"),
        ({"rows_malformed": -1}, "conteos incoherentes"),
        ({"status": "ok", "md5": None}, "faltan conteos o md5"),
        ({"status": "raro"}, "status desconocido"),
        ({"status": "missing"}, "trae datos"),
    ],
)
def test_lote_incoherente_falla(cambios: dict[str, object], texto: str) -> None:
    with pytest.raises((DataValidationError, SchemaError), match=texto):
        NewsDataset((lote_ok(**cambios),))


def test_rotulos_duplicados_o_desordenados_fallan() -> None:
    b0 = parse_batch(registros()[1])
    otro = lote_ok(L0130, events=())
    with pytest.raises(DataValidationError, match="duplicados o desordenados"):
        NewsDataset((b0, b0))
    with pytest.raises(DataValidationError, match="duplicados o desordenados"):
        NewsDataset((otro, b0))


def test_record_id_repetido_entre_lotes_falla() -> None:
    b0 = parse_batch(registros()[1])
    con_mismos_ids = NewsBatch(**{**b0.__dict__, "label": L0130, "events": b0.events})
    with pytest.raises(DataValidationError, match="no es del lote|record_id repetido"):
        NewsDataset((b0, con_mismos_ids))


def test_tono_no_finito_falla() -> None:
    b0 = parse_batch(registros()[1])
    malo = NewsBatch(
        **{
            **b0.__dict__,
            "events": (
                b0.events[0].__class__("x", L0115, "d", "u", float("nan"), None),
            )
            + b0.events[1:],
        }
    )
    with pytest.raises(DataValidationError, match="tono no finito"):
        NewsDataset((malo,))


def test_latencia_por_defecto_es_de_30_minutos() -> None:
    assert np.timedelta64(30, "m") == DEFAULT_LATENCY
    assert lote_ok().available_at() == np.datetime64("2026-10-20T01:45:00", "ns")
