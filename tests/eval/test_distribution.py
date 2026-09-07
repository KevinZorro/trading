"""La distribucion por semillas y el Deflated Sharpe Ratio.

Lo que se verifica no es que los percentiles esten bien calculados -eso lo hace
numpy- sino que las **reglas del proyecto** sean mecanicas: que reportar menos
de 10 semillas falle, que saltearse la regla deje marca, y que el DSR descuente
de verdad el numero de configuraciones probadas.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from eval.distribution import (
    MIN_SEEDS,
    DistributionError,
    decompose_variance,
    deflated_sharpe,
    drift_t_statistic,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    summarize,
    t_critical_95,
)

SEMILLAS = list(range(10))


def test_menos_de_diez_semillas_falla() -> None:
    """El principio 5 del proyecto, hecho codigo y no nota al pie."""
    with pytest.raises(DistributionError, match="el minimo es 10"):
        summarize("retorno", [1, 2, 3], [0.1, 0.2, 0.3])


def test_saltearse_la_regla_deja_marca() -> None:
    """Se puede explorar con pocas semillas; lo que no se puede es disimularlo."""
    d = summarize("retorno", [1, 2], [0.1, 0.2], allow_fewer_seeds=True)
    assert d.below_minimum_seeds is True
    assert d.describe()["below_minimum_seeds"] is True
    assert "MENOS DE 10 SEMILLAS" in d.render()


def test_con_diez_semillas_no_hace_falta_el_permiso() -> None:
    d = summarize("retorno", SEMILLAS, [float(i) for i in range(10)])
    assert d.below_minimum_seeds is False
    assert len(d.values) == MIN_SEEDS


def test_los_percentiles_son_los_de_numpy_con_interpolacion_lineal() -> None:
    """Oraculo independiente: la mediana de 0..9 es 4.5 y el p25 es 2.25.

    Se fija el metodo porque con 10 observaciones cambiarlo mueve el p25 de
    forma visible, y dos reportes con metodos distintos no son comparables.
    """
    d = summarize("x", SEMILLAS, [float(i) for i in range(10)])
    assert d.median == pytest.approx(4.5)
    assert d.p25 == pytest.approx(2.25)
    assert d.p75 == pytest.approx(6.75)
    assert d.minimum == pytest.approx(0.0)
    assert d.maximum == pytest.approx(9.0)
    assert d.iqr == pytest.approx(4.5)


def test_un_valor_indefinido_no_se_convierte_en_cero() -> None:
    """``None`` es "la metrica no existe", que no es "la metrica vale cero".

    Mismo criterio que ``win_rate`` sin trades cerrados. Convertirlo a 0.0
    arrastraria la mediana hacia abajo inventando observaciones.
    """
    valores: list[float | None] = [1.0] * 9 + [None]
    d = summarize("win_rate", SEMILLAS, valores)
    assert d.n_valid == 9
    assert d.n_missing == 1
    assert d.median == pytest.approx(1.0)


def test_todas_indefinidas_da_estadisticos_nulos() -> None:
    d = summarize("win_rate", SEMILLAS, [None] * 10)
    assert d.median is None
    assert d.n_valid == 0
    assert d.fraction_above(0.0) is None


def test_los_valores_no_finitos_se_rechazan() -> None:
    """Un inf domina cualquier percentil sin significar nada."""
    with pytest.raises(DistributionError, match="no finitos"):
        summarize("sharpe", SEMILLAS, [1.0] * 9 + [float("inf")])


def test_semillas_repetidas_se_rechazan() -> None:
    with pytest.raises(DistributionError, match="semillas repetidas"):
        summarize("x", [1] * 10, [0.5] * 10)


def test_fraction_above_cuenta_semillas_y_no_mira_la_mediana() -> None:
    """Dos distribuciones con la misma mediana no son la misma evidencia."""
    d = summarize("retorno", SEMILLAS, [-1.0] * 4 + [0.5] * 6)
    assert d.median == pytest.approx(0.5)
    assert d.fraction_above(0.0) == pytest.approx(0.6)


def test_longitudes_distintas_se_rechazan() -> None:
    with pytest.raises(DistributionError, match="no se pueden parear"):
        summarize("x", [1, 2], [0.1], allow_fewer_seeds=True)


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio
# ---------------------------------------------------------------------------


def test_el_maximo_esperado_crece_con_el_numero_de_intentos() -> None:
    """Probar mas configuraciones sube la vara. Ese es todo el punto del DSR."""
    valores = [expected_max_sharpe(n, 0.04) for n in (2, 10, 100, 1000)]
    assert valores == sorted(valores)
    assert all(v > 0 for v in valores)


def test_con_un_solo_intento_no_hay_nada_que_deflactar() -> None:
    assert expected_max_sharpe(1, 0.04) == 0.0


def test_sin_dispersion_entre_intentos_no_hay_nada_que_deflactar() -> None:
    """Si todas las configuraciones dan el mismo Sharpe, no hubo seleccion."""
    assert expected_max_sharpe(50, 0.0) == 0.0


def test_el_maximo_esperado_tiene_la_forma_cerrada() -> None:
    """Oraculo a mano de la formula de Bailey y Lopez de Prado."""
    from statistics import NormalDist

    n, var = 10, 0.09
    gamma = 0.5772156649015329
    normal = NormalDist()
    esperado = math.sqrt(var) * (
        (1.0 - gamma) * normal.inv_cdf(1.0 - 1.0 / n)
        + gamma * normal.inv_cdf(1.0 - 1.0 / (n * math.e))
    )
    assert expected_max_sharpe(n, var) == pytest.approx(esperado, rel=1e-12)


def test_el_psr_es_media_cuando_el_sharpe_iguala_al_benchmark() -> None:
    """Si el Sharpe observado es exactamente el benchmark, la probabilidad es 0.5."""
    p = probabilistic_sharpe_ratio(
        0.1, benchmark_sharpe=0.1, n_obs=500, skew=0.0, kurtosis=3.0
    )
    assert p == pytest.approx(0.5)


def test_la_asimetria_negativa_baja_el_psr() -> None:
    """Colas izquierdas hacen el estimador mas ruidoso, no mas significativo."""
    base = probabilistic_sharpe_ratio(
        0.1, benchmark_sharpe=0.0, n_obs=500, skew=0.0, kurtosis=3.0
    )
    sesgada = probabilistic_sharpe_ratio(
        0.1, benchmark_sharpe=0.0, n_obs=500, skew=-1.5, kurtosis=8.0
    )
    assert sesgada < base


def test_el_dsr_castiga_haber_probado_muchas_configuraciones() -> None:
    """El mismo resultado deja de ser significativo si se probaron 200 cosas.

    Es el escenario que el proyecto quiere evitar: barrer hiperparametros,
    quedarse con el mejor y reportar su p-value como si hubiera sido el unico.
    """
    rng = np.random.default_rng(7)
    retornos = rng.normal(0.0012, 0.01, size=1_000)
    pocos = deflated_sharpe(retornos, n_trials=1, sharpe_variance=0.01)
    muchos = deflated_sharpe(retornos, n_trials=200, sharpe_variance=0.01)
    assert pocos.value > muchos.value
    assert pocos.observed_sharpe == pytest.approx(muchos.observed_sharpe)
    assert muchos.benchmark_sharpe > pocos.benchmark_sharpe


def test_el_dsr_no_declara_significativo_al_ruido_puro() -> None:
    """10 semillas de ruido con dispersion: el mejor no puede ser significativo."""
    rng = np.random.default_rng(3)
    retornos = rng.normal(0.0, 0.01, size=800)
    resultado = deflated_sharpe(retornos, n_trials=10, sharpe_variance=0.04)
    assert resultado.significant is False


def test_el_dsr_rechaza_una_serie_sin_dispersion() -> None:
    with pytest.raises(DistributionError, match="dispersion nula"):
        deflated_sharpe(np.full(100, 0.001), n_trials=1, sharpe_variance=0.01)


def test_el_dsr_rechaza_series_demasiado_cortas() -> None:
    with pytest.raises(DistributionError, match="al menos 2 retornos"):
        deflated_sharpe(np.array([0.01]), n_trials=1, sharpe_variance=0.01)


# ---------------------------------------------------------------------------
# Varianza de mercado contra varianza de entrenamiento
# ---------------------------------------------------------------------------


def test_separa_la_varianza_de_mercado_de_la_de_entrenamiento() -> None:
    """Caminos muy distintos entre si, semillas muy parecidas dentro de cada uno.

    Es la forma que tenia el hallazgo del nivel 4: diez semillas de acuerdo entre
    si sobre un camino, y una dispersion enorme al cambiar de camino.
    """
    rng = np.random.default_rng(11)
    medias = [-1.0, -0.6, -0.2, 0.1, 0.3, 0.5, 0.8, 1.1, 1.4, 1.8]
    por_camino = [list(m + 0.01 * rng.standard_normal(10)) for m in medias]
    d = decompose_variance("exceso", por_camino)
    assert d.n_paths == 10
    assert d.n_seeds_per_path == 10
    assert d.between_path_std > 0.8
    assert d.within_path_std is not None
    assert d.within_path_std < 0.05
    ratio = d.variance_ratio
    assert ratio is not None
    assert ratio > 20


def test_un_exceso_chico_frente_a_la_dispersion_entre_caminos_no_se_distingue() -> None:
    """La regresion del nivel 4, en su forma minima.

    Media +0.164 con desvio entre caminos ~0.8 y N=10: el error estandar es ~0.25
    y el estadistico no llega ni a 1. El criterio viejo, que comparaba +0.164
    contra un umbral fijo de 0.05, lo declaraba significativo.
    """
    excesos = [
        0.164 + d for d in (0.9, -1.1, 0.4, -0.7, 1.2, -0.5, 0.2, -0.9, 0.6, -0.1)
    ]
    d = decompose_variance("exceso", [[e] * 10 for e in excesos])
    assert d.mean == pytest.approx(0.164, abs=0.01)
    assert abs(d.t_statistic or 0.0) < d.t_critical
    assert d.distinguishable_from_zero is False
    assert "NO distinguible de cero" in d.render()


def test_un_exceso_consistente_entre_caminos_si_se_distingue() -> None:
    """Contraprueba: sin esto, el test anterior podria pasar por vacuidad."""
    d = decompose_variance("exceso", [[0.30 + 0.01 * i] * 10 for i in range(10)])
    assert d.distinguishable_from_zero is True
    assert "DISTINGUIBLE de cero" in d.render()


def test_el_valor_critico_usa_la_t_y_no_la_normal() -> None:
    """Con N=10 la diferencia entre 2.262 y 1.960 decide un veredicto."""
    assert t_critical_95(9) == pytest.approx(2.262)
    assert t_critical_95(1) == pytest.approx(12.706)
    assert t_critical_95(500) == pytest.approx(1.960)
    with pytest.raises(DistributionError, match="al menos 2 observaciones"):
        t_critical_95(0)


def test_un_solo_camino_no_permite_separar_nada() -> None:
    """Es exactamente el error que esta descomposicion existe para impedir."""
    with pytest.raises(DistributionError, match="al menos 2 caminos"):
        decompose_variance("exceso", [[0.1] * 10])


def test_un_camino_sin_valores_validos_falla() -> None:
    with pytest.raises(DistributionError, match="ni un valor valido"):
        decompose_variance("x", [[0.1, 0.2], [None, None]])


def test_el_t_del_drift_detecta_un_drift_grande_y_no_uno_chico() -> None:
    """El numero que dice si "converger a estar invertido" era alcanzable.

    Oraculo: con media/desvio conocidos, ``t = media/(desvio/sqrt(n))``.
    """
    rng = np.random.default_rng(5)
    ruido = rng.standard_normal(4_800)
    ruido = (ruido - ruido.mean()) / ruido.std(ddof=1)

    # Drift del regimen medium: mu=8% anual sobre 252 barras, sigma 30% anual.
    debil = 0.08 / 252 + ruido * 0.30 / math.sqrt(252)
    assert abs(drift_t_statistic(debil)) < 2.0

    fuerte = 0.60 / 252 + ruido * 0.10 / math.sqrt(252)
    assert abs(drift_t_statistic(fuerte)) > 4.0


def test_el_t_del_drift_coincide_con_la_formula_a_mano() -> None:
    serie = np.array([0.01, -0.005, 0.02, 0.0, 0.015])
    esperado = float(serie.mean()) / (float(serie.std(ddof=1)) / math.sqrt(5))
    assert drift_t_statistic(serie) == pytest.approx(esperado, rel=1e-12)


def test_el_t_del_drift_rechaza_series_degeneradas() -> None:
    with pytest.raises(DistributionError, match="al menos 2 retornos"):
        drift_t_statistic(np.array([0.01]))
    with pytest.raises(DistributionError, match="dispersion nula"):
        drift_t_statistic(np.full(50, 0.01))
