"""Metricas sobre series de equity construidas a mano.

Los valores esperados se calculan desde la definicion de cada metrica, no
llamando a la implementacion. Donde el numero no es redondo se deja escrita la
aritmetica que lo produce, para que la revision pueda seguirla sin ejecutar
nada.

Serie de referencia de casi todo el modulo:

    equity      = [100, 120,  90, 108]
    retornos    = [0.20, -0.25, 0.20]      (120/100-1, 90/120-1, 108/90-1)
    bars_per_year = 4  ->  3 periodos = 0.75 anos
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from eval.metrics import (
    MetricError,
    annual_volatility,
    cagr,
    calmar,
    excess_returns,
    first_ruin_index,
    max_drawdown,
    periodic_rate,
    sharpe,
    simple_returns,
    sortino,
    total_return,
    truncate_at_ruin,
    years_elapsed,
)
from sim.engine import SimConfig

BARRAS_POR_ANO = 4.0
SERIE = np.array([100.0, 120.0, 90.0, 108.0])

# Desvio muestral de [0.20, -0.25, 0.20]: media 0.05, desvios [0.15, -0.30, 0.15],
# suma de cuadrados 0.0225 + 0.09 + 0.0225 = 0.135, dividido por (3 - 1).
DESVIO_MUESTRAL = math.sqrt(0.135 / 2)
MEDIA_RETORNOS = 0.05
# Segundo momento parcial inferior: solo -0.25 aporta, sobre las 3 observaciones.
DESVIO_A_LA_BAJA = math.sqrt(0.0625 / 3)


class TestRetornos:
    def test_retornos_simples(self) -> None:
        np.testing.assert_allclose(simple_returns(SERIE), [0.20, -0.25, 0.20])

    def test_retorno_total(self) -> None:
        assert total_return(SERIE) == pytest.approx(0.08)

    def test_una_serie_de_n_puntos_tiene_n_menos_1_periodos(self) -> None:
        assert years_elapsed(4, BARRAS_POR_ANO) == pytest.approx(0.75)
        # Un ano de barras diarias son 253 observaciones, no 252.
        assert years_elapsed(253, 252.0) == pytest.approx(1.0)

    def test_cagr(self) -> None:
        # (108/100) ** (1 / 0.75) - 1
        assert cagr(SERIE, BARRAS_POR_ANO) == pytest.approx(1.08 ** (4 / 3) - 1)

    def test_retornos_en_exceso_descuentan_la_tasa_por_barra(self) -> None:
        tasa_barra = (1.05) ** (1 / BARRAS_POR_ANO) - 1
        np.testing.assert_allclose(
            excess_returns(SERIE, BARRAS_POR_ANO, 0.05),
            np.array([0.20, -0.25, 0.20]) - tasa_barra,
        )


class TestTasaPorBarra:
    def test_coincide_con_la_del_motor(self) -> None:
        """La tasa que descuenta el Sharpe es la que el motor devenga.

        Si divergieran, una estrategia que pasa tiempo en cash quedaria premiada
        o castigada por una inconsistencia contable, no por su desempeno.
        """
        for anual in (0.0, 0.01, 0.05, 0.10):
            config = SimConfig(
                initial_cash=1_000.0, cash_rate=anual, bars_per_year=252.0
            )
            assert periodic_rate(anual, 252.0) == pytest.approx(config.rate_per_bar)

    def test_tasa_cero_es_cero_exacto(self) -> None:
        assert periodic_rate(0.0, 252.0) == 0.0


class TestRiesgo:
    def test_volatilidad_anualizada(self) -> None:
        assert annual_volatility(SERIE, BARRAS_POR_ANO) == pytest.approx(
            DESVIO_MUESTRAL * 2.0  # sqrt(4)
        )

    def test_sharpe(self) -> None:
        assert sharpe(SERIE, BARRAS_POR_ANO) == pytest.approx(
            MEDIA_RETORNOS / DESVIO_MUESTRAL * 2.0
        )

    def test_sortino_solo_castiga_la_dispersion_a_la_baja(self) -> None:
        assert sortino(SERIE, BARRAS_POR_ANO) == pytest.approx(
            MEDIA_RETORNOS / DESVIO_A_LA_BAJA * 2.0
        )

    def test_el_sortino_supera_al_sharpe_con_una_sola_caida(self) -> None:
        """Oraculo cualitativo: el denominador a la baja es menor que el total."""
        assert sortino(SERIE, BARRAS_POR_ANO) > sharpe(SERIE, BARRAS_POR_ANO)

    def test_sharpe_con_tasa_libre_de_riesgo(self) -> None:
        tasa_barra = (1.20) ** (1 / BARRAS_POR_ANO) - 1
        exceso = np.array([0.20, -0.25, 0.20]) - tasa_barra
        media = float(np.mean(exceso))
        desvio = math.sqrt(float(np.sum((exceso - media) ** 2)) / 2)
        assert sharpe(SERIE, BARRAS_POR_ANO, 0.20) == pytest.approx(
            media / desvio * 2.0
        )


class TestDrawdown:
    def test_geometria_completa(self) -> None:
        dd = max_drawdown(SERIE)
        # Maximos previos: [100, 120, 120, 120]. Caidas: [0, 0, 0.25, 0.10].
        assert dd.depth == pytest.approx(0.25)
        assert dd.peak_index == 1
        assert dd.trough_index == 2
        assert dd.recovery_index is None  # nunca vuelve a 120
        assert dd.duration_bars == 1
        assert dd.underwater_bars == 2  # del pico (1) al final (3)

    def test_recuperacion_cuando_vuelve_al_pico(self) -> None:
        dd = max_drawdown(np.array([100.0, 120.0, 90.0, 130.0]))
        assert dd.depth == pytest.approx(0.25)
        assert dd.recovery_index == 3
        assert dd.underwater_bars == 2  # del pico (1) a la recuperacion (3)

    def test_calmar(self) -> None:
        assert calmar(SERIE, BARRAS_POR_ANO) == pytest.approx(
            (1.08 ** (4 / 3) - 1) / 0.25
        )

    def test_serie_monotona_creciente_no_tiene_drawdown(self) -> None:
        dd = max_drawdown(np.array([100.0, 110.0, 120.0]))
        assert dd.depth == 0.0
        assert dd.duration_bars == 0
        assert dd.underwater_bars == 0

    def test_calmar_sin_drawdown_es_infinito(self) -> None:
        assert calmar(np.array([100.0, 110.0, 120.0]), BARRAS_POR_ANO) == math.inf


class TestSerieConstante:
    """Caso borde: nada se mueve. Ninguna metrica debe devolver nan."""

    CONSTANTE = np.array([100.0, 100.0, 100.0, 100.0])

    def test_retorno_y_cagr_son_cero(self) -> None:
        assert total_return(self.CONSTANTE) == 0.0
        assert cagr(self.CONSTANTE, BARRAS_POR_ANO) == pytest.approx(0.0)

    def test_volatilidad_cero(self) -> None:
        assert annual_volatility(self.CONSTANTE, BARRAS_POR_ANO) == 0.0

    def test_sharpe_y_sortino_son_cero_no_nan(self) -> None:
        assert sharpe(self.CONSTANTE, BARRAS_POR_ANO) == 0.0
        assert sortino(self.CONSTANTE, BARRAS_POR_ANO) == 0.0

    def test_calmar_es_cero(self) -> None:
        assert calmar(self.CONSTANTE, BARRAS_POR_ANO) == 0.0

    def test_sin_drawdown(self) -> None:
        assert max_drawdown(self.CONSTANTE).depth == 0.0


class TestCrecimientoSinVolatilidad:
    """Retorno constante y positivo: el Sharpe no esta acotado.

    El test no es trivial. Esta serie crece exactamente 10% por barra, pero en
    float64 los retornos NO salen identicos (110/100 y 133.1/121 difieren en el
    ultimo bit), asi que el desvio muestral es del orden de 1e-17 en vez de
    cero. Una comparacion contra cero exacto devolveria un Sharpe finito de
    ~1.5e15: un numero que gana cualquier ranking de semillas y regimenes sin
    significar nada.
    """

    RECTA = np.array([100.0, 110.0, 121.0, 133.1])

    def test_los_retornos_no_son_exactamente_iguales_en_float64(self) -> None:
        """Premisa del test siguiente. Si esto cambia, el otro deja de probar algo."""
        retornos = simple_returns(self.RECTA)
        assert len(set(retornos.tolist())) > 1
        assert float(np.std(retornos, ddof=1)) > 0.0

    def test_sharpe_infinito_no_un_numero_gigante(self) -> None:
        assert sharpe(self.RECTA, BARRAS_POR_ANO) == math.inf

    def test_sortino_infinito(self) -> None:
        assert sortino(self.RECTA, BARRAS_POR_ANO) == math.inf

    def test_una_dispersion_real_no_se_confunde_con_ruido(self) -> None:
        """El umbral es relativo: una serie con volatilidad de verdad la mide."""
        casi_recta = np.array([100.0, 110.0, 121.5, 133.0])
        assert math.isfinite(sharpe(casi_recta, BARRAS_POR_ANO))


class TestEquityQueTocaCero:
    def test_detecta_la_barra_de_la_ruina(self) -> None:
        serie = np.array([100.0, 50.0, 0.0, 30.0])
        assert first_ruin_index(serie) == 2

    def test_trunca_incluyendo_la_barra_de_la_ruina(self) -> None:
        serie = np.array([100.0, 50.0, 0.0, 30.0])
        cortada, ruina = truncate_at_ruin(serie)
        assert ruina == 2
        np.testing.assert_allclose(cortada, [100.0, 50.0, 0.0])

    def test_sin_ruina_devuelve_la_serie_entera(self) -> None:
        cortada, ruina = truncate_at_ruin(SERIE)
        assert ruina is None
        np.testing.assert_allclose(cortada, SERIE)

    def test_metricas_sobre_la_serie_truncada(self) -> None:
        cortada, _ = truncate_at_ruin(np.array([100.0, 50.0, 0.0, 30.0]))
        assert total_return(cortada) == pytest.approx(-1.0)
        assert cagr(cortada, BARRAS_POR_ANO) == -1.0
        assert max_drawdown(cortada).depth == pytest.approx(1.0)

    def test_equity_negativo_pierde_mas_que_todo(self) -> None:
        """La liquidacion puede ser negativa: cerrar cuesta mas de lo que hay."""
        serie = np.array([100.0, 50.0, -10.0])
        assert total_return(serie) == pytest.approx(-1.10)
        # 1 - (-10 / 100) = 1.10
        assert max_drawdown(serie).depth == pytest.approx(1.10)
        # No se inventa un numero peor que "todo".
        assert cagr(serie, BARRAS_POR_ANO) == -1.0


class TestSeriesInvalidas:
    def test_una_sola_observacion(self) -> None:
        with pytest.raises(MetricError, match="al menos 2 observaciones"):
            total_return(np.array([100.0]))

    def test_equity_inicial_no_positivo(self) -> None:
        with pytest.raises(MetricError, match="equity inicial debe ser positivo"):
            total_return(np.array([0.0, 100.0]))

    def test_valores_no_finitos(self) -> None:
        with pytest.raises(MetricError, match="no finitos"):
            total_return(np.array([100.0, np.nan]))

    def test_serie_bidimensional(self) -> None:
        with pytest.raises(MetricError, match="1-D"):
            total_return(np.ones((2, 2)))

    def test_sharpe_necesita_tres_observaciones(self) -> None:
        """Con dos puntos hay un solo retorno y no existe varianza muestral."""
        with pytest.raises(MetricError, match="al menos 3 observaciones"):
            sharpe(np.array([100.0, 110.0]), BARRAS_POR_ANO)

    def test_bars_per_year_no_positivo(self) -> None:
        with pytest.raises(MetricError, match="bars_per_year"):
            years_elapsed(10, 0.0)
