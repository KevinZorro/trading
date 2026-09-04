"""Reporte comparativo de las dos series de equity.

Los casos con series construidas a mano llevan su aritmetica escrita. Los de
integracion corren el simulador de verdad y verifican invariantes estructurales,
no numeros concretos: fijar un numero que sale del propio motor lo convertiria
en su propio oraculo.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from agents.baselines import BuyAndHold
from data.instruments import InstrumentSpec
from eval.metrics import MetricError, max_drawdown
from eval.report import evaluate_run, evaluate_series, friction_gap
from sim.costs import FixedBpsSpread, LinearSlippage
from sim.engine import SimConfig, Simulator

from ..sim.conftest import make_series

BARRAS_POR_ANO = 4.0

# Dos series pensadas para que la posicion sea grande en el pico y chica en el
# valle: la friccion de salida es 10 en el pico y 1 en el resto.
MARCA = np.array([100.0, 120.0, 90.0, 100.0])
LIQUIDACION = np.array([99.0, 110.0, 89.0, 99.0])


class TestBrechaEntreSeries:
    def test_metricas_de_la_brecha(self) -> None:
        # brecha = [1, 10, 1, 1]; media = 13/4 = 3.25
        f = friction_gap(MARCA, LIQUIDACION)
        assert f.mean_gap == pytest.approx(3.25)
        assert f.max_gap == pytest.approx(10.0)
        assert f.min_gap == pytest.approx(1.0)
        assert f.final_gap == pytest.approx(1.0)
        assert f.max_gap_pct == pytest.approx(10.0 / 120.0)
        assert f.final_gap_pct == pytest.approx(1.0 / 100.0)

    def test_series_de_distinta_longitud(self) -> None:
        with pytest.raises(MetricError, match="longitudes distintas"):
            friction_gap(MARCA, LIQUIDACION[:-1])


class TestDrawdownDivergente:
    """El caso donde las dos series dejan de contar la misma historia.

    ``equity_liquidation <= equity_mark`` en toda barra, pero eso NO implica que
    su drawdown sea mayor. Si la posicion es grande en el pico y chica en el
    valle, la friccion deprime el pico mas que el valle y el drawdown de la
    serie de liquidacion resulta *menor*. Escribir la asercion al reves seria
    una premisa falsa que ningun test de series suaves detecta.
    """

    def test_la_liquidacion_nunca_supera_a_la_marca(self) -> None:
        assert np.all(LIQUIDACION <= MARCA)

    def test_el_drawdown_de_liquidacion_es_menor(self) -> None:
        # Marca: pico 120 en t=1, valle 90 en t=2  ->  1 - 90/120 = 0.25
        assert max_drawdown(MARCA).depth == pytest.approx(0.25)
        # Liquidacion: pico 110 en t=1, valle 89 en t=2  ->  1 - 89/110
        assert max_drawdown(LIQUIDACION).depth == pytest.approx(1 - 89 / 110)
        assert max_drawdown(LIQUIDACION).depth < max_drawdown(MARCA).depth

    def test_el_reporte_muestra_las_dos_sin_ordenarlas(self) -> None:
        m = evaluate_series(MARCA, label="mark", bars_per_year=BARRAS_POR_ANO)
        lq = evaluate_series(
            LIQUIDACION, label="liquidation", bars_per_year=BARRAS_POR_ANO
        )
        assert lq.final_equity < m.final_equity
        assert lq.drawdown.depth < m.drawdown.depth


class TestAnualizacion:
    def test_periodo_corto_queda_marcado(self) -> None:
        # 4 observaciones a 4 barras por ano = 0.75 anos.
        m = evaluate_series(MARCA, label="mark", bars_per_year=BARRAS_POR_ANO)
        assert m.years == pytest.approx(0.75)
        assert m.annualization_reliable is False
        # Se calcula igual: en walk-forward todas las ventanas son cortas.
        assert math.isfinite(m.cagr)

    def test_periodo_de_un_ano_o_mas_es_confiable(self) -> None:
        serie = np.linspace(100.0, 130.0, 253)
        m = evaluate_series(serie, label="mark", bars_per_year=252.0)
        assert m.years == pytest.approx(1.0)
        assert m.annualization_reliable is True

    def test_dos_observaciones_no_tienen_sharpe(self) -> None:
        m = evaluate_series(
            np.array([100.0, 110.0]), label="mark", bars_per_year=BARRAS_POR_ANO
        )
        assert m.sharpe is None
        assert m.sortino is None
        assert m.total_return == pytest.approx(0.10)


class TestSerieQueTocaCero:
    def test_se_reporta_la_barra_de_la_ruina(self) -> None:
        m = evaluate_series(
            np.array([100.0, 50.0, 0.0, 30.0]),
            label="liquidation",
            bars_per_year=BARRAS_POR_ANO,
        )
        assert m.ruined_at == 2
        assert m.n_bars == 3  # la serie se corta en la ruina
        assert m.total_return == pytest.approx(-1.0)

    def test_ruina_en_la_segunda_barra_todavia_es_medible(self) -> None:
        """Dos observaciones bastan para el retorno y el drawdown, no para el Sharpe."""
        m = evaluate_series(
            np.array([100.0, 0.0]), label="liquidation", bars_per_year=BARRAS_POR_ANO
        )
        assert m.ruined_at == 1
        assert m.n_bars == 2
        assert m.total_return == pytest.approx(-1.0)
        assert m.drawdown.depth == pytest.approx(1.0)
        assert m.sharpe is None

    def test_ruina_en_la_primera_barra_no_deja_nada_que_medir(self) -> None:
        with pytest.raises(MetricError, match="observaciones utiles"):
            evaluate_series(
                np.array([0.0, 100.0]),
                label="liquidation",
                bars_per_year=BARRAS_POR_ANO,
            )


def _corrida(spec: InstrumentSpec, *, cash: float = 100_000.0):
    """Corrida con costos reales: spread, slippage y comision distintos de cero."""
    precios = [100.0, 102.0, 101.0, 105.0, 103.0, 108.0, 106.0, 110.0]
    aperturas = [p * 1.001 for p in precios]
    series = make_series(spec, precios, opens=aperturas, volume=50_000.0)
    config = SimConfig(
        initial_cash=cash,
        spread=FixedBpsSpread(bps=10.0),
        slippage=LinearSlippage(k=0.5),
        bars_per_year=252.0,
    )
    return Simulator(series, config).run(BuyAndHold())


class TestCostosYGap:
    def test_el_gap_no_esta_dentro_de_los_costos(
        self, equity_spec: InstrumentSpec
    ) -> None:
        """El invariante central del desglose, verificado sobre el reporte.

        ``total_costs_paid`` es exactamente spread + slippage + comision. El gap
        vive en su propia linea y solo se suma en ``implementation_shortfall``,
        que se llama por su nombre.
        """
        reporte = evaluate_run(_corrida(equity_spec))
        c = reporte.costs
        assert c.total_costs_paid == pytest.approx(c.spread + c.slippage + c.commission)
        assert c.implementation_shortfall == pytest.approx(c.total_costs_paid + c.gap)

    def test_los_costos_son_no_negativos_y_el_gap_lleva_signo(
        self, equity_spec: InstrumentSpec
    ) -> None:
        c = evaluate_run(_corrida(equity_spec)).costs
        assert c.spread >= 0.0
        assert c.slippage >= 0.0
        assert c.commission >= 0.0
        # Con open > close de la barra anterior el gap de una compra es adverso;
        # lo que importa es que exista como magnitud propia y con signo.
        assert c.gap != 0.0

    def test_un_gap_favorable_no_se_esconde_dentro_de_los_costos(
        self, sin_comision: InstrumentSpec
    ) -> None:
        """Abre por debajo del cierre previo: la compra sale mas barata.

        Es el efecto que se midio en el tier de riesgo alto de la Etapa 1. Si el
        gap se agregara a los costos, este caso apareceria como "costos
        negativos", que es un sinsentido, o quedaria cancelado contra el spread.
        """
        precios = [100.0, 100.0, 100.0, 100.0]
        aperturas = [100.0, 95.0, 100.0, 100.0]  # el fill de t=1 abre 5% abajo
        series = make_series(sin_comision, precios, opens=aperturas, volume=50_000.0)
        result = Simulator(series, SimConfig(initial_cash=10_000.0)).run(BuyAndHold())
        c = evaluate_run(result).costs
        assert c.gap < 0.0  # favorable: se pago menos que el precio de decision
        assert c.total_costs_paid >= 0.0
        assert c.implementation_shortfall == pytest.approx(c.total_costs_paid + c.gap)

    def test_costos_en_bps_del_nocional(self, equity_spec: InstrumentSpec) -> None:
        reporte = evaluate_run(_corrida(equity_spec))
        c, t = reporte.costs, reporte.turnover
        assert c.costs_bps_of_turnover == pytest.approx(
            c.total_costs_paid / t.notional * 1e4
        )


class TestReporteIntegrado:
    def test_la_brecha_nunca_es_negativa_con_long_only(
        self, equity_spec: InstrumentSpec
    ) -> None:
        reporte = evaluate_run(_corrida(equity_spec))
        assert reporte.friction.min_gap >= 0.0
        assert reporte.friction.max_gap > 0.0  # hay posicion y salir cuesta

    def test_la_tasa_libre_de_riesgo_sale_del_cash_rate(
        self, sin_comision: InstrumentSpec
    ) -> None:
        series = make_series(sin_comision, [100.0] * 6, volume=50_000.0)
        config = SimConfig(initial_cash=10_000.0, cash_rate=0.05, bars_per_year=252.0)
        reporte = evaluate_run(Simulator(series, config).run(BuyAndHold()))
        assert reporte.risk_free == pytest.approx(0.05)

    def test_la_tasa_se_puede_forzar(self, sin_comision: InstrumentSpec) -> None:
        series = make_series(sin_comision, [100.0] * 6, volume=50_000.0)
        config = SimConfig(initial_cash=10_000.0, cash_rate=0.05, bars_per_year=252.0)
        reporte = evaluate_run(
            Simulator(series, config).run(BuyAndHold()), risk_free=0.0
        )
        assert reporte.risk_free == 0.0

    def test_buy_and_hold_hace_un_solo_trade_y_lo_deja_abierto(
        self, equity_spec: InstrumentSpec
    ) -> None:
        reporte = evaluate_run(_corrida(equity_spec))
        assert reporte.trades.n_round_trips == 0
        assert reporte.trades.win_rate is None
        assert reporte.trades.open_position is not None

    def test_el_reporte_se_serializa(self, equity_spec: InstrumentSpec) -> None:
        d = evaluate_run(_corrida(equity_spec)).as_dict()
        assert set(d) >= {
            "strategy_name",
            "mark",
            "liquidation",
            "friction",
            "costs",
            "turnover",
            "trades",
            "config",
        }
        assert isinstance(d["mark"], dict)
        assert isinstance(d["costs"], dict)

    def test_el_render_separa_costos_de_gap(self, equity_spec: InstrumentSpec) -> None:
        texto = evaluate_run(_corrida(equity_spec)).render()
        assert "equity_mark" in texto
        assert "equity_liquid." in texto
        assert "Friccion no realizada" in texto
        assert "Costos de transaccion (desglosados, nunca agregados)" in texto
        assert "Gap de ejecucion (valuacion, NO es un costo" in texto
        # La linea del gap esta despues de la del total de costos, en su propio
        # bloque: no hay una sola cifra de "costos" que los agregue.
        assert texto.index("TOTAL costos pagados") < texto.index("gap close[t]")

    def test_el_render_avisa_cuando_el_periodo_es_corto(
        self, equity_spec: InstrumentSpec
    ) -> None:
        texto = evaluate_run(_corrida(equity_spec)).render()
        assert "menos de un ano" in texto
