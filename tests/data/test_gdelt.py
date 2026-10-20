"""Parser del GKG contra un lote real recortado (tests/live/gdelt)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from data.gdelt import (
    N_COLUMNAS,
    batch_label_str,
    floor_label,
    gkg_url,
    parse_gkg,
)

LOTE = Path(__file__).resolve().parents[1] / "live/gdelt/20261020011500.gkg.csv.zip"


def fila(
    themes: str, tone: str = "1,2,3,4,5,6,7", extras: str = "", n: int = N_COLUMNAS
) -> str:
    c = [""] * n
    c[0], c[3], c[4] = "id-1", "ejemplo.com", "https://ejemplo.com/a"
    if n == N_COLUMNAS:
        c[8], c[15], c[26] = themes, tone, extras
    return "\t".join(c)


def lineas_reales() -> list[str]:
    with zipfile.ZipFile(LOTE) as z:
        return z.read(z.namelist()[0]).decode().splitlines()


def test_lote_real_trae_cinco_filas_y_tres_articulos_del_tema() -> None:
    lote = parse_gkg(lineas_reales())
    # Oraculo a mano, leido del archivo original: 5 filas, 3 con ECON_BITCOIN.
    assert (lote.rows_total, lote.rows_malformed) == (5, 0)
    assert [a.record_id for a in lote.articles] == [
        "20261020011500-345",
        "20261020011500-443",
        "20261020011500-973",
    ]


def test_articulo_real_conserva_dominio_url_tono_y_titulo() -> None:
    a = parse_gkg(lineas_reales()).articles[0]
    assert a.domain == "econotimes.com"
    assert a.url == (
        "https://econotimes.com/Siebert-Analyst-Sets-Bitcoin-Target-at-175-000-"
        "Within-12-Months-1757150"
    )
    assert a.tone == (
        4.25531914893617,
        4.9645390070922,
        0.709219858156028,
        5.67375886524823,
        20.5673758865248,
        0.0,
        116.0,
    )
    assert a.title == "Siebert Analyst Sets Bitcoin Target at $175,000 Within 12 Months"


def test_tema_se_busca_por_token_exacto() -> None:
    lote = parse_gkg(
        [
            fila("TAX,1;ECON_BITCOIN,51;MEDIA,3"),
            fila("ECON_BITCOIN_FUTURES,5"),
            fila("NOT_ECON_BITCOIN,5"),
            fila("ECON_BITCOINX,5;OTRO,2"),
        ]
    )
    assert lote.rows_total == 4
    assert len(lote.articles) == 1


def test_otro_tema_se_elige_por_parametro() -> None:
    lote = parse_gkg(
        [fila("ECON_BITCOIN,1"), fila("ECON_INFLATION,1")], "ECON_INFLATION"
    )
    assert len(lote.articles) == 1


def test_fila_malformada_cuenta_en_el_total_y_no_aborta() -> None:
    lote = parse_gkg([fila("ECON_BITCOIN,1", n=5), fila("ECON_BITCOIN,1"), "", "\n"])
    assert (lote.rows_total, lote.rows_malformed, len(lote.articles)) == (2, 1, 1)


def test_tono_ilegible_en_fila_con_el_tema_no_inventa_un_tono() -> None:
    lote = parse_gkg(
        [fila("ECON_BITCOIN,1", tone="1,2"), fila("ECON_BITCOIN,1", tone="a,b")]
    )
    assert (lote.rows_malformed, len(lote.articles)) == (2, 0)


def test_sin_titulo_queda_none_y_con_titulo_multilinea_se_conserva() -> None:
    sin, con = parse_gkg(
        [
            fila("ECON_BITCOIN,1", extras="<AUTHORS>x</AUTHORS>"),
            fila("ECON_BITCOIN,1", extras="<PAGE_TITLE>a\nb &amp; c</PAGE_TITLE>"),
        ]
    ).articles
    assert sin.title is None
    assert con.title == "a\nb &amp; c"  # sin limpiar: es decision de quien puntua


def test_archivo_vacio_da_lote_vacio() -> None:
    buf = io.StringIO("")
    lote = parse_gkg(buf)
    assert (lote.rows_total, lote.rows_malformed, lote.articles) == (0, 0, ())


def test_rotulo_y_url() -> None:
    r = np.datetime64("2026-10-20T01:15:00")
    assert batch_label_str(r) == "20261020011500"
    assert (
        gkg_url(r) == "https://data.gdeltproject.org/gdeltv2/20261020011500.gkg.csv.zip"
    )


@pytest.mark.parametrize(
    ("instante", "esperado"),
    [
        ("2026-10-20T01:40:43", "2026-10-20T01:30:00"),
        ("2026-10-20T01:45:00", "2026-10-20T01:45:00"),  # el rotulo exacto es su lote
        ("2026-10-20T01:44:59.999", "2026-10-20T01:30:00"),
        ("2026-10-20T00:00:00", "2026-10-20T00:00:00"),
    ],
)
def test_floor_label_es_multiplo_de_15_minutos_menor_o_igual(
    instante: str, esperado: str
) -> None:
    assert floor_label(np.datetime64(instante)) == np.datetime64(esperado, "ns")
