"""Features de noticias: oraculos calculados a mano sobre agregados sinteticos."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from data.errors import DataValidationError
from features.news import (
    NEWS_FEATURE_NAMES,
    PLACEBO_LAG,
    build_news_features,
)
from tests.features.news_helpers import cierre, grilla

COL = {n: i for i, n in enumerate(NEWS_FEATURE_NAMES)}


def fila(frame: pd.DataFrame, dia: str) -> np.ndarray:  # type: ignore[type-arg]
    return build_news_features(frame).at(np.array([cierre(dia)]))[0]


def test_news_share_a_mano() -> None:
    # Una barra de 96 lotes: tema 2 de 100 -> 0.02; log(1 + 1e4 * 0.02) = log(201).
    f = fila(grilla(3), "2020-01-02")
    assert f[COL["news_share"]] == pytest.approx(math.log(201))


def test_tone_mean_y_no_title_a_mano() -> None:
    f = fila(grilla(3), "2020-01-02")
    assert f[COL["tone_mean"]] == pytest.approx(-2.0)  # (96 * -4) / (96 * 2)
    assert f[COL["no_title"]] == 0.0
    f0 = fila(grilla(3, n_titled=0), "2020-01-02")
    assert f0[COL["no_title"]] == 1.0


def test_finbert_queda_enmascarada_hasta_n2b() -> None:
    assert np.isnan(fila(grilla(3), "2020-01-02")[COL["finbert_mean"]])


def test_coverage_ok_exige_87_de_96_lotes_sanos() -> None:
    for caidos, esperado in [(9, 1.0), (10, 0.0)]:
        frame = grilla(3)
        # Rotulos 01-01 23:15 a 01-02 23:00 (+60 min): barra que cierra el 01-03.
        dia = (frame["label"] > "2020-01-01 23:00") & (
            frame["label"] <= "2020-01-02 23:00"
        )
        idx = frame.index[dia][:caidos]
        frame.loc[idx, ["status", "rows_total", "rows_malformed", "rows_theme"]] = [
            "missing",
            np.nan,
            np.nan,
            np.nan,
        ]
        frame.loc[idx, ["tone_sum", "n_titled", "last_modified_offset_s"]] = np.nan
        assert fila(frame, "2020-01-03")[COL["coverage_ok"]] == esperado


def test_un_lote_degradado_no_cuenta_en_news_share_pero_si_en_el_tono() -> None:
    frame = grilla(3)
    # Lote degradado (6 filas contra 100): bajo el 25 % de la mediana, no es sano.
    # Su unico articulo del tema tiene tono +50 (los demas lotes, -2 por articulo).
    i = frame.index[frame["label"] == "2020-01-02 12:00"][0]
    frame.loc[i, ["rows_total", "rows_theme", "tone_sum"]] = [6, 1, 50.0]
    f = fila(frame, "2020-01-03")
    assert f[COL["news_share"]] == pytest.approx(math.log(201))  # sin el lote malo
    assert f[COL["tone_mean"]] == pytest.approx((95 * -4 + 50) / (95 * 2 + 1))
    assert f[COL["coverage_ok"]] == 1.0  # 95 sanos de 96 >= 87


def test_barra_sin_lotes_sanos_enmascara_las_features_de_volumen() -> None:
    frame = grilla(4)
    dia = (frame["label"] > "2020-01-02 23:00") & (frame["label"] <= "2020-01-03 23:00")
    frame.loc[dia, ["status", "rows_total", "rows_malformed", "rows_theme"]] = [
        "missing",
        np.nan,
        np.nan,
        np.nan,
    ]
    frame.loc[dia, ["tone_sum", "n_titled", "last_modified_offset_s"]] = np.nan
    f = fila(frame, "2020-01-04")
    assert np.isnan(
        f[[COL["news_share"], COL["news_share_z30"], COL["tone_mean"]]]
    ).all()
    assert (f[COL["no_title"]], f[COL["coverage_ok"]]) == (1.0, 0.0)


def test_z30_y_delta7_a_mano() -> None:
    dias = 40
    frame = grilla(dias)
    tema = np.repeat(np.arange(dias) % 5 + 1, 96)  # 1..5 repetido por dia calendario
    desplazado = np.roll(
        tema, -4
    )  # la barra k agrupa rotulos 23:00 de k-1 a 22:45 de k
    frame["rows_theme"] = desplazado
    frame["tone_sum"] = desplazado * (-1.5)
    tabla = build_news_features(frame)
    # La primera barra incompleta es la del 01-01 + 1; tomo la 35.a barra completa.
    k = 35
    share = tabla.values[:, COL["news_share"]]
    previas = share[k - 30 : k]
    z = (share[k] - previas.mean()) / previas.std()
    assert tabla.values[k, COL["news_share_z30"]] == pytest.approx(z)
    tono = tabla.values[:, COL["tone_mean"]]
    assert tabla.values[k, COL["tone_delta7"]] == pytest.approx(
        tono[k] - tono[k - 7 : k].mean()
    )


def test_z30_se_enmascara_con_menos_de_20_previas_validas() -> None:
    tabla = build_news_features(grilla(40))
    z = tabla.values[:, COL["news_share_z30"]]
    # Desvio de una serie constante = 0 -> enmascarada tambien con historia completa.
    assert np.isnan(z).all()
    tema = np.repeat(np.arange(40) % 3 + 1, 96)
    tabla = build_news_features(grilla(40, rows_theme=tema))
    z = tabla.values[:, COL["news_share_z30"]]
    primeras_validas = int(np.argmax(np.isfinite(z)))
    # Con 20 previas validas ya hay valor; con 19, no. La barra 0 no tiene lotes sanos
    # completos pero si algunos: cuenta como valida.
    assert primeras_validas == 20
    assert np.isnan(z[:20]).all()


def test_el_placebo_es_la_fila_de_364_dias_antes() -> None:
    frame = grilla(400, rows_theme=np.repeat(np.arange(400) % 7 + 1, 96))
    tabla = build_news_features(frame)
    cierres = np.array([cierre("2020-12-31"), cierre("2021-01-01")])
    placebo = tabla.at(cierres, PLACEBO_LAG)
    pos = np.searchsorted(tabla.closes, cierres)
    np.testing.assert_array_equal(placebo, tabla.values[pos - 364])
    # 364 = 52 semanas: el dia de la semana se conserva.
    dias = (cierres - np.timedelta64(364, "D")).astype("datetime64[D]").astype("int64")
    assert ((dias - cierres.astype("datetime64[D]").astype("int64")) % 7 == 0).all()


def test_at_rechaza_lag_negativo_y_cierres_sin_fila() -> None:
    tabla = build_news_features(grilla(5))
    with pytest.raises(DataValidationError, match="lag no puede ser negativo"):
        tabla.at(np.array([cierre("2020-01-03")]), np.timedelta64(-1, "D"))
    with pytest.raises(DataValidationError, match="no hay features de noticias"):
        tabla.at(np.array([cierre("2019-12-01")]))
    with pytest.raises(DataValidationError, match="no hay features de noticias"):
        tabla.at(np.array([np.datetime64("2020-01-03T12:00", "ns")]))


def test_latencia_negativa_falla() -> None:
    with pytest.raises(DataValidationError, match="latencia no puede ser negativa"):
        build_news_features(grilla(3), np.timedelta64(-15, "m"))
