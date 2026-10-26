"""Anti-leakage de las features de noticias: la fila ``t`` no ve nada posterior."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.news import NEWS_LATENCY, build_news_features
from tests.features.news_helpers import cierre, grilla

pytestmark = pytest.mark.leakage

LAT = pd.Timedelta(NEWS_LATENCY)


def variada(dias: int) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = dias * 96
    frame = grilla(dias)
    frame["rows_theme"] = rng.integers(0, 6, n)
    frame["tone_sum"] = rng.normal(-2.0, 3.0, n)
    frame["n_titled"] = np.minimum(frame["rows_theme"], rng.integers(0, 3, n))
    return frame


def test_corromper_lotes_posteriores_al_cierre_no_cambia_esa_fila_ni_las_previas() -> (
    None
):
    frame = variada(60)
    base = build_news_features(frame)
    c = cierre("2020-02-15")
    # Lotes cuyo rotulo + 60 min es > c: ninguno de ellos puede mover filas <= c.
    despues = (frame["label"] + LAT) > pd.Timestamp(c)
    sucio = frame.copy()
    sucio.loc[despues, "rows_theme"] = 99
    sucio.loc[despues, "tone_sum"] = 1e6
    sucio.loc[despues, "n_titled"] = 0
    alterado = build_news_features(sucio)
    k = int(np.searchsorted(base.closes, c, "right"))
    np.testing.assert_array_equal(base.values[:k], alterado.values[:k])
    assert not np.array_equal(base.values[k:], alterado.values[k:], equal_nan=True)


def test_la_fila_t_sale_igual_con_los_datos_truncados_en_close_t() -> None:
    frame = variada(60)
    base = build_news_features(frame)
    for dia in ("2020-01-20", "2020-02-10", "2020-02-25"):
        c = cierre(dia)
        corte = frame[(frame["label"] + LAT) <= pd.Timestamp(c)]
        truncada = build_news_features(corte.reset_index(drop=True))
        np.testing.assert_array_equal(
            truncada.at(np.array([c])), base.at(np.array([c]))
        )


@pytest.mark.parametrize(
    ("rotulo", "barra"),
    [
        ("2020-01-01 23:00", "2020-01-02"),  # 23:00 + 60 = 00:00 <= cierre: entra
        ("2020-01-01 23:15", "2020-01-03"),  # 00:15 > cierre: a la barra siguiente
    ],
)
def test_frontera_de_la_latencia_de_60_min(rotulo: str, barra: str) -> None:
    frame = grilla(4)
    i = frame.index[frame["label"] == rotulo][0]
    frame.loc[i, ["rows_theme", "tone_sum"]] = [1, 500.0]
    tabla = build_news_features(frame)
    tono = dict(
        zip(
            tabla.closes.astype("datetime64[D]").astype(str),
            tabla.values[:, 2],
            strict=True,
        )
    )
    contaminada = [d for d, t in tono.items() if np.isfinite(t) and abs(t + 2.0) > 1e-9]
    assert contaminada == [barra]


def test_el_placebo_de_t_no_depende_de_nada_posterior_a_close_t_menos_364() -> None:
    frame = variada(420)
    base = build_news_features(frame)
    c = cierre("2021-02-05")
    placebo_c = c - np.timedelta64(364, "D")
    despues = (frame["label"] + LAT) > pd.Timestamp(placebo_c)
    sucio = frame.copy()
    sucio.loc[despues, ["rows_theme", "tone_sum"]] = [99, 1e6]
    alterado = build_news_features(sucio)
    np.testing.assert_array_equal(
        base.at(np.array([c]), np.timedelta64(364, "D")),
        alterado.at(np.array([c]), np.timedelta64(364, "D")),
    )


def test_finbert_de_lotes_posteriores_al_cierre_no_cambia_esa_fila() -> None:
    frame = variada(60)
    frame["finbert_n"] = frame["n_titled"]
    frame["finbert_sum"] = 0.1 * frame["n_titled"]
    base = build_news_features(frame)
    c = cierre("2020-02-15")
    despues = (frame["label"] + LAT) > pd.Timestamp(c)
    sucio = frame.copy()
    sucio.loc[despues, "finbert_n"] = 7
    sucio.loc[despues, "finbert_sum"] = 7.0
    alterado = build_news_features(sucio)
    k = int(np.searchsorted(base.closes, c, "right"))
    np.testing.assert_array_equal(base.values[:k], alterado.values[:k])
    assert not np.array_equal(base.values[k:], alterado.values[k:], equal_nan=True)
