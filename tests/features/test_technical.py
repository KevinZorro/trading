"""Indicadores tecnicos con oraculos calculados a mano.

Donde el numero no es redondo, la aritmetica queda escrita en el test para que
la revision pueda seguirla sin ejecutar nada. Ninguna cifra sale de llamar a la
implementacion.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from features.technical import atr, bollinger, macd, normalized_volume, rsi


class TestRSI:
    def test_serie_estrictamente_creciente_da_100(self) -> None:
        """Sin una sola perdida, el RSI esta topeado."""
        assert rsi(np.arange(1.0, 16.0), period=14) == 100.0

    def test_serie_estrictamente_decreciente_da_0(self) -> None:
        assert rsi(np.arange(15.0, 0.0, -1.0), period=14) == 0.0

    def test_serie_plana_da_50(self) -> None:
        """Ni sobrecomprada ni sobrevendida, que es lo que una serie plana es."""
        assert rsi(np.full(15, 42.0), period=14) == 50.0

    def test_caso_calculado_a_mano(self) -> None:
        # closes = [10, 11, 10.5, 11.5], period = 2
        # variaciones      = [+1.0, -0.5, +1.0]
        # ganancias        = [ 1.0,  0.0,  1.0]
        # perdidas         = [ 0.0,  0.5,  0.0]
        # Wilder(ganancias, 2): media([1, 0]) = 0.5; luego (0.5*1 + 1)/2 = 0.75
        # Wilder(perdidas, 2):  media([0, 0.5]) = 0.25; luego (0.25*1 + 0)/2 = 0.125
        # RS = 0.75 / 0.125 = 6  ->  RSI = 100 - 100/7
        esperado = 100.0 - 100.0 / 7.0
        assert rsi(np.array([10.0, 11.0, 10.5, 11.5]), period=2) == pytest.approx(
            esperado
        )

    def test_necesita_period_mas_uno_cierres(self) -> None:
        with pytest.raises(ValueError, match="al menos 15 barras"):
            rsi(np.arange(1.0, 15.0), period=14)

    def test_period_invalido(self) -> None:
        with pytest.raises(ValueError, match="period debe ser positivo"):
            rsi(np.arange(1.0, 16.0), period=0)


class TestMACD:
    def test_serie_plana_da_todo_cero(self) -> None:
        """El EMA de una constante es la constante: las dos lineas coinciden."""
        linea, senal, hist = macd(np.full(30, 7.0))
        assert linea == pytest.approx(0.0)
        assert senal == pytest.approx(0.0)
        assert hist == pytest.approx(0.0)

    def test_serie_creciente_da_macd_positivo(self) -> None:
        """La media rapida sigue al precio mas de cerca que la lenta."""
        linea, _, _ = macd(np.arange(1.0, 41.0))
        assert linea > 0.0

    def test_serie_decreciente_da_macd_negativo(self) -> None:
        linea, _, _ = macd(np.arange(40.0, 0.0, -1.0))
        assert linea < 0.0

    def test_caso_calculado_con_la_recursion_del_ema(self) -> None:
        """El oraculo re-deriva el EMA desde su definicion, no llama a macd."""
        closes = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)

        def ema(valores, period):
            alpha = 2.0 / (period + 1.0)
            fuera = [valores[0]]
            for x in valores[1:]:
                fuera.append(alpha * x + (1 - alpha) * fuera[-1])
            return fuera

        rapida = ema(closes, 2)
        lenta = ema(closes, 3)
        linea_esperada = [r - le for r, le in zip(rapida, lenta, strict=True)]
        senal_esperada = ema(np.array(linea_esperada), 2)

        linea, senal, hist = macd(closes, fast=2, slow=3, signal=2)
        assert linea == pytest.approx(linea_esperada[-1])
        assert senal == pytest.approx(senal_esperada[-1])
        assert hist == pytest.approx(linea_esperada[-1] - senal_esperada[-1])

    def test_fast_debe_ser_menor_que_slow(self) -> None:
        with pytest.raises(ValueError, match="menor que slow"):
            macd(np.arange(1.0, 41.0), fast=26, slow=12)


class TestATR:
    def test_caso_calculado_a_mano(self) -> None:
        # high  = [10, 12, 11]; low = [8, 9, 9]; close = [9, 11, 10]; period = 2
        # barra 1: h-l = 3 ; |h - c_prev| = |12-9| = 3 ; |l - c_prev| = |9-9| = 0
        #          TR = 3
        # barra 2: h-l = 2 ; |11-11| = 0 ; |9-11| = 2
        #          TR = 2
        # Wilder([3, 2], 2) = media([3, 2]) = 2.5
        valor = atr(
            np.array([10.0, 12.0, 11.0]),
            np.array([8.0, 9.0, 9.0]),
            np.array([9.0, 11.0, 10.0]),
            period=2,
        )
        assert valor == pytest.approx(2.5)

    def test_el_rango_verdadero_incluye_el_gap(self) -> None:
        """Una barra que abre lejos tiene TR mayor que su propio rango.

        Importa en este proyecto: el gap entre close[t] y open[t+1] es un
        termino de primer nivel del desglose de costos.
        """
        sin_gap = atr(
            np.array([10.0, 10.5]), np.array([9.5, 10.0]), np.array([10.0, 10.2]), 1
        )
        con_gap = atr(
            np.array([10.0, 20.5]), np.array([9.5, 20.0]), np.array([10.0, 20.2]), 1
        )
        assert con_gap > sin_gap
        # El rango de la segunda barra es 0.5 en los dos casos; la diferencia es
        # solo el gap: |20.5 - 10.0| = 10.5
        assert con_gap == pytest.approx(10.5)

    def test_longitudes_inconsistentes(self) -> None:
        with pytest.raises(ValueError, match="al menos"):
            atr(np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0]), period=2)


class TestBollinger:
    def test_caso_calculado_a_mano(self) -> None:
        # closes = [1, 2, 3, 4, 5], period = 5, k = 2
        # media = 3 ; desvio poblacional = sqrt(2)
        # banda inferior = 3 - 2*sqrt(2) ; ancho total = 4*sqrt(2)
        # %B = (5 - (3 - 2*sqrt(2))) / (4*sqrt(2))
        # ancho relativo = 4*sqrt(2) / 3
        raiz2 = math.sqrt(2.0)
        pct_b, ancho = bollinger(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), period=5, k=2.0)
        assert pct_b == pytest.approx((5.0 - (3.0 - 2 * raiz2)) / (4 * raiz2))
        assert ancho == pytest.approx(4 * raiz2 / 3.0)

    def test_serie_plana_da_medio_y_cero(self) -> None:
        """Sin desvio no hay banda: el precio esta exactamente en la media."""
        pct_b, ancho = bollinger(np.full(20, 5.0), period=20)
        assert pct_b == 0.5
        assert ancho == 0.0

    def test_el_ancho_es_adimensional(self) -> None:
        """Dividido por la media: comparable entre instrumentos y regimenes."""
        base = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        _, ancho_bajo = bollinger(base, period=5)
        _, ancho_alto = bollinger(base * 1000.0, period=5)
        assert ancho_bajo == pytest.approx(ancho_alto)

    def test_period_minimo(self) -> None:
        with pytest.raises(ValueError, match="al menos 2"):
            bollinger(np.array([1.0]), period=1)


class TestVolumenNormalizado:
    def test_caso_calculado_a_mano(self) -> None:
        # volumenes = [1, 1, 1, 1, 2], period = 5
        # media = 6/5 = 1.2 ; actual = 2 ; log(2 / 1.2) = log(5/3)
        valor = normalized_volume(np.array([1.0, 1.0, 1.0, 1.0, 2.0]), period=5)
        assert valor == pytest.approx(math.log(5.0 / 3.0))

    def test_volumen_igual_a_su_media_da_cero(self) -> None:
        assert normalized_volume(np.full(20, 7.0), period=20) == pytest.approx(0.0)

    def test_es_simetrico_en_el_logaritmo(self) -> None:
        """Cinco veces lo normal y un quinto pesan igual con signo opuesto."""
        arriba = normalized_volume(np.array([1.0, 1.0, 1.0, 1.0, 5.0]), period=5)
        media_arriba = (4 + 5) / 5
        assert arriba == pytest.approx(math.log(5.0 / media_arriba))
        abajo = normalized_volume(np.array([5.0, 5.0, 5.0, 5.0, 1.0]), period=5)
        media_abajo = (20 + 1) / 5
        assert abajo == pytest.approx(math.log(1.0 / media_abajo))
        assert abajo < 0 < arriba

    def test_barra_sin_volumen_no_devuelve_menos_infinito(self) -> None:
        """El simulador la rechaza con NO_VOLUME; un -inf envenenaria la
        observacion entera."""
        valor = normalized_volume(np.array([1.0, 1.0, 1.0, 1.0, 0.0]), period=5)
        assert valor == 0.0
        assert math.isfinite(valor)


class TestValidacionComun:
    def test_valores_no_finitos(self) -> None:
        with pytest.raises(ValueError, match="no finitos"):
            rsi(np.array([1.0, 2.0, np.nan, 4.0]), period=2)

    def test_array_bidimensional(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            bollinger(np.ones((3, 3)), period=2)

    def test_ninguna_funcion_guarda_estado(self) -> None:
        """Dos evaluaciones sobre la misma ventana dan el mismo numero."""
        closes = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 5.5, 7.0])
        assert rsi(closes, 3) == rsi(closes, 3)
        assert macd(closes, 2, 4, 2) == macd(closes, 2, 4, 2)
        assert bollinger(closes, 4) == bollinger(closes, 4)
