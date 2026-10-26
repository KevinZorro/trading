"""Fixtures del nivel 5 (ADR 0012, 5): canal exogeno con optimo conocido.

El 5a pone la senal en ``tone_mean`` y no en el precio; el 5b es el 4b mas siete
columnas de ruido. Los oraculos son aritmetica a mano sobre la definicion, nunca
el generador contra si mismo.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from data.fixtures import (
    DEFAULT_DRIFT,
    DEFAULT_SIGMA,
    EXOG_NAMES,
    ExogChannel,
    FixtureError,
    level_4b_detectable_drift,
    level_5a_exog_signal,
    level_5b_exog_noise,
)
from features.news import NEWS_FEATURE_NAMES

pytestmark = pytest.mark.leakage

N = 6_000


def _col(fx, nombre):  # type: ignore[no-untyped-def]
    return fx.series.exog[:, fx.series.exog_names.index(nombre)]


def _corr(a, b):  # type: ignore[no-untyped-def]
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1])


def test_los_nombres_son_los_de_las_features_de_noticias() -> None:
    assert EXOG_NAMES == NEWS_FEATURE_NAMES


def test_5a_la_columna_predice_el_retorno_siguiente_y_el_precio_no() -> None:
    fx = level_5a_exog_signal(N, seed=7)
    r, z = np.asarray(fx.signal), _col(fx, "tone_mean")
    # corr(z_t, r_{t+1}) = beta = 0.3; error estandar ~ 1/sqrt(6000) = 0.013.
    assert _corr(z[:-1], r[1:]) == pytest.approx(0.3, abs=0.05)
    # El precio casi no contiene nada: r_t y r_{t+1} comparten z por la
    # autocorrelacion de la columna, corr = beta^2 * 0.549 = 0.049 (R^2 = 0.0024).
    assert _corr(r[1:-1], r[2:]) == pytest.approx(0.09 * 0.549, abs=0.03)
    # Marginal del nivel 1: desvio sigma_target.
    assert np.std(r[1:]) == pytest.approx(DEFAULT_SIGMA, rel=0.05)


def test_5a_las_otras_columnas_no_predicen() -> None:
    fx = level_5a_exog_signal(N, seed=7)
    r = np.asarray(fx.signal)
    for nombre in EXOG_NAMES:
        if nombre == "tone_mean":
            continue
        assert abs(_corr(_col(fx, nombre)[:-1], r[1:])) < 0.06, nombre


def test_la_esperanza_condicional_es_la_formula_a_mano() -> None:
    fx = level_5a_exog_signal(N, seed=3)
    z = _col(fx, "tone_mean")
    esperado = DEFAULT_DRIFT + 0.3 * DEFAULT_SIGMA * z
    np.testing.assert_allclose(fx.expected_next_log_return, esperado, rtol=1e-12)
    # Desvio condicional: sigma * sqrt(1 - beta^2).
    assert fx.sigma_by_bar[0] == pytest.approx(DEFAULT_SIGMA * math.sqrt(0.91))


def test_5a_el_techo_informado_supera_a_estar_siempre_invertido() -> None:
    fx = level_5a_exog_signal(N, seed=3)
    techo = fx.ceilings(safety=0.98)
    assert not techo.informed_below_reference
    assert (
        techo.informed_total_return > techo.always_long[-1] / techo.always_long[0] - 1
    )
    assert techo.expected_growth_per_bar is not None
    assert techo.expected_growth_per_bar > DEFAULT_DRIFT - 0.5 * DEFAULT_SIGMA**2


def test_las_columnas_ruido_tienen_la_autocorrelacion_pedida() -> None:
    fx = level_5b_exog_noise(N, seed=5)
    esperado = {"news_share": 0.907, "tone_mean": 0.549, "tone_delta7": 0.315}
    for nombre, acf in esperado.items():
        x = _col(fx, nombre)
        assert _corr(x[:-1], x[1:]) == pytest.approx(acf, abs=0.06), nombre
    assert _col(fx, "coverage_ok").mean() == pytest.approx(0.917, abs=0.08)


def test_finbert_esta_enmascarado_exactamente_donde_no_hay_titulos() -> None:
    fx = level_5b_exog_noise(N, seed=5)
    sin_titulo = _col(fx, "no_title") == 1.0
    assert sin_titulo.any()
    np.testing.assert_array_equal(np.isnan(_col(fx, "finbert_mean")), sin_titulo)
    for nombre in EXOG_NAMES:
        if nombre != "finbert_mean":
            assert np.isfinite(_col(fx, nombre)).all(), nombre


def test_5b_el_precio_es_el_del_4b_y_el_ruido_no_lo_toca() -> None:
    base = level_4b_detectable_drift(N, seed=11)
    fx = level_5b_exog_noise(N, seed=11)
    np.testing.assert_array_equal(fx.series.close, base.series.close)
    assert fx.series.exog.shape == (N, 7)
    r = np.asarray(fx.signal)
    for nombre in EXOG_NAMES:
        assert abs(_corr(_col(fx, nombre)[:-1], r[1:])) < 0.06, nombre


@pytest.mark.parametrize("builder", [level_5a_exog_signal, level_5b_exog_noise])
def test_el_pasado_no_depende_de_cuantas_barras_se_piden(builder) -> None:  # type: ignore[no-untyped-def]
    corto, largo = builder(400, seed=9), builder(700, seed=9)
    np.testing.assert_array_equal(corto.series.exog, largo.series.exog[:400])
    if builder is level_5a_exog_signal:
        # El precio del 5b es el del generador Heston del 4b, cuyo prefijo no
        # esta fijado por sus propios tests; aqui solo se fija lo que agrega el 5.
        np.testing.assert_array_equal(corto.series.close, largo.series.close[:400])
        np.testing.assert_array_equal(corto.series.volume, largo.series.volume[:400])


def test_la_semilla_cambia_el_camino_y_la_misma_lo_repite() -> None:
    a, b, c = (level_5a_exog_signal(300, seed=s) for s in (1, 1, 2))
    np.testing.assert_array_equal(a.series.exog, b.series.exog)
    assert not np.array_equal(a.series.exog, c.series.exog)


def test_split_corta_las_exogenas_con_el_precio() -> None:
    fx = level_5a_exog_signal(1_000, seed=4)
    train, val, test = fx.split()
    assert len(train) == 600
    assert train.series.exog.shape == (600, 7)
    np.testing.assert_array_equal(test.series.exog, fx.series.exog[800:])
    assert test.channel == fx.channel
    # La esperanza condicional del tramo es la del padre, recortada.
    np.testing.assert_allclose(
        test.expected_next_log_return, fx.expected_next_log_return[800:]
    )


def test_without_exog_deja_el_mercado_de_5b_sin_columnas() -> None:
    fx = level_5b_exog_noise(300, seed=2)
    a = fx.without_exog()
    assert a.series.exog is None
    assert a.series.exog_names == ()
    np.testing.assert_array_equal(a.series.close, fx.series.close)


def test_without_exog_rechaza_el_5a() -> None:
    with pytest.raises(FixtureError, match="el techo del canal"):
        level_5a_exog_signal(300, seed=2).without_exog()


def test_el_source_marca_los_fixtures_y_describe_el_canal() -> None:
    fx = level_5a_exog_signal(300, seed=2)
    assert fx.series.source.startswith("fixture:level_5a")
    canal = fx.describe()["channel"]
    assert canal["r_squared"] == pytest.approx(0.09)  # type: ignore[index]
    assert level_5b_exog_noise(300, seed=2).describe()["channel"] is None


def test_validaciones() -> None:
    with pytest.raises(FixtureError, match="beta debe estar en"):
        level_5a_exog_signal(300, seed=1, beta=1.0)
    with pytest.raises(FixtureError, match="beta del canal"):
        ExogChannel("tone_mean", 0.0, 0.01)
    with pytest.raises(FixtureError, match="sigma_marginal"):
        ExogChannel("tone_mean", 0.3, 0.0)
    fx = level_5a_exog_signal(300, seed=1)
    from dataclasses import replace

    with pytest.raises(FixtureError, match="que la serie no tiene"):
        replace(fx, channel=ExogChannel("otra", 0.3, 0.01))
