"""Invariantes contables y de costos del motor."""

from __future__ import annotations

import numpy as np
import pytest

from agents.baselines import BuyAndHold
from data.instruments import CommissionSchema, InstrumentSpec
from data.synthetic import generate_gbm_sv
from sim.costs import (
    CorwinSchultzSpread,
    FixedBpsSpread,
    LinearSlippage,
    SchemaCommission,
    SqrtSlippage,
)
from sim.engine import SimConfig, Simulator
from sim.orders import MarketOrder, OrderStatus, RejectReason
from sim.portfolio import ACCOUNTING_TOL

from .conftest import make_series


class CompraUnaVez:
    """Compra ``qty`` en la barra ``t_entrada`` y no hace nada mas."""

    name = "compra_una_vez"

    def __init__(self, qty: float, t_entrada: int = 0) -> None:
        self.qty = qty
        self.t_entrada = t_entrada

    def reset(self, seed: int | None = None) -> None:
        return None

    def on_bar(self, view, account):
        if view.t == self.t_entrada:
            return MarketOrder(qty=self.qty)
        return None


class TestSinFricciones:
    def test_buy_and_hold_replica_el_retorno_del_activo(
        self, fractional: InstrumentSpec
    ) -> None:
        """Con costos en cero, el capital invertido rinde exactamente el activo.

        Matiz que no es un fallo del test: buy-and-hold no puede estar al 100%
        invertido sin lookahead, porque dimensiona con ``close[t]`` y ejecuta al
        ``open[t+1]``. Por eso la igualdad exacta se afirma sobre el capital
        efectivamente invertido, y aparte se comprueba que el equity cuadra
        exactamente con cash + posicion.
        """
        closes = [100.0, 103.0, 99.0, 107.0, 112.0]
        series = make_series(fractional, closes)
        # safety=0.9: la barra de ejecucion abre un 3% por encima del precio de
        # dimensionamiento. Con un margen menor la orden se rechazaria por cash
        # insuficiente, que es exactamente el fenomeno descrito arriba.
        result = Simulator(series, SimConfig(initial_cash=10_000.0)).run(
            BuyAndHold(safety=0.9)
        )

        fill = result.fills[0]
        assert fill.transaction_costs == 0.0

        retorno_activo = closes[-1] / fill.fill_price - 1.0
        pnl = result.equity[-1] - result.equity[0]
        capital_invertido = fill.qty_filled * fill.fill_price
        assert pnl / capital_invertido == pytest.approx(retorno_activo, rel=1e-12)

        cash_final = 10_000.0 - capital_invertido
        assert result.equity[-1] == pytest.approx(
            cash_final + fill.qty_filled * closes[-1], rel=1e-12
        )

    def test_sin_operar_el_equity_es_constante(
        self, fractional: InstrumentSpec
    ) -> None:
        series = make_series(fractional, [100.0, 120.0, 80.0, 95.0])

        class NoOpera:
            name = "no_opera"

            def reset(self, seed: int | None = None) -> None:
                return None

            def on_bar(self, view, account):
                return None

        result = Simulator(series, SimConfig(initial_cash=5_000.0)).run(NoOpera())
        assert np.all(result.equity == 5_000.0)
        assert result.fills == []


class TestMonotoniaDeCostos:
    """Mas costos nunca puede significar mas retorno."""

    @pytest.fixture
    def series(self, fractional: InstrumentSpec):
        return generate_gbm_sv(
            250, seed=3, instrument=fractional, params="medium", s0=100.0
        )

    def _retorno(self, series, config: SimConfig) -> float:
        result = Simulator(series, config).run(BuyAndHold())
        return result.equity[-1] / result.equity[0] - 1.0

    @pytest.mark.parametrize("bps", [0.0, 1.0, 5.0, 20.0, 100.0, 500.0])
    def test_spread_creciente_reduce_el_retorno(self, series, bps: float) -> None:
        base = self._retorno(series, SimConfig(initial_cash=100_000.0))
        con_spread = self._retorno(
            series,
            SimConfig(initial_cash=100_000.0, spread=FixedBpsSpread(bps=bps)),
        )
        assert con_spread <= base + 1e-12

    def test_secuencia_de_spreads_es_monotona(self, series) -> None:
        retornos = [
            self._retorno(
                series,
                SimConfig(initial_cash=100_000.0, spread=FixedBpsSpread(bps=bps)),
            )
            for bps in (0.0, 1.0, 5.0, 20.0, 100.0)
        ]
        assert retornos == sorted(retornos, reverse=True)

    def test_comision_creciente_reduce_el_retorno(self, series) -> None:
        retornos = [
            self._retorno(
                series,
                SimConfig(
                    initial_cash=100_000.0,
                    commission=SchemaCommission(
                        CommissionSchema(kind="percent", value=rate)
                    ),
                ),
            )
            for rate in (0.0, 0.0005, 0.001, 0.005, 0.02)
        ]
        assert retornos == sorted(retornos, reverse=True)

    def test_slippage_creciente_reduce_el_retorno(self, series) -> None:
        retornos = [
            self._retorno(
                series,
                SimConfig(initial_cash=100_000.0, slippage=LinearSlippage(k=k)),
            )
            for k in (0.0, 0.01, 0.1, 1.0)
        ]
        assert retornos == sorted(retornos, reverse=True)

    def test_monotonia_con_una_estrategia_que_rota(self, series) -> None:
        """Con turnover alto el efecto es mayor, pero el orden es el mismo."""
        from agents.baselines import MovingAverageCross

        retornos = []
        for bps in (0.0, 10.0, 50.0):
            config = SimConfig(initial_cash=100_000.0, spread=FixedBpsSpread(bps=bps))
            result = Simulator(series, config).run(
                MovingAverageCross(fast=5, slow=20)
            )
            retornos.append(result.equity[-1])
        assert retornos == sorted(retornos, reverse=True)


class TestConservacion:
    """cash + valor de posicion == equity, en todo momento."""

    def _verificar(self, result, series) -> None:
        esperado = result.cash + result.position * np.asarray(series.close)
        np.testing.assert_allclose(result.equity, esperado, rtol=ACCOUNTING_TOL)

    def test_con_fricciones_completas(self, fractional: InstrumentSpec) -> None:
        series = generate_gbm_sv(
            200, seed=11, instrument=fractional, params="high", s0=100.0
        )
        config = SimConfig(
            initial_cash=50_000.0,
            spread=CorwinSchultzSpread(),
            slippage=SqrtSlippage(k=0.05),
            commission=SchemaCommission(CommissionSchema(kind="percent", value=0.001)),
        )
        from agents.baselines import RandomAgent

        result = Simulator(series, config).run(RandomAgent(seed=7, trade_prob=0.4))
        self._verificar(result, series)
        assert len(result.executed_fills) > 10

    def test_tras_llenados_parciales(self, fractional: InstrumentSpec) -> None:
        """El caso que mas facilmente descuadra la contabilidad."""
        series = make_series(
            fractional, [100.0] * 6, volume=1_000.0
        )  # capacidad: 100 unidades por barra
        config = SimConfig(initial_cash=1_000_000.0, max_participation=0.10)
        result = Simulator(series, config).run(CompraUnaVez(qty=5_000.0))

        fill = result.fills[0]
        assert fill.status is OrderStatus.PARTIAL
        assert fill.qty_filled == pytest.approx(100.0)
        assert fill.qty_requested == 5_000.0
        self._verificar(result, series)

    def test_el_remanente_parcial_se_cancela(self, fractional: InstrumentSpec) -> None:
        """No se arrastra a la barra siguiente: una sola ejecucion."""
        series = make_series(fractional, [100.0] * 6, volume=1_000.0)
        result = Simulator(
            series, SimConfig(initial_cash=1_000_000.0)
        ).run(CompraUnaVez(qty=5_000.0))
        assert len(result.fills) == 1
        assert result.position[-1] == pytest.approx(100.0)

    def test_el_cash_baja_exactamente_por_nocional_mas_comision(
        self, fractional: InstrumentSpec
    ) -> None:
        series = make_series(fractional, [100.0, 100.0, 100.0])
        config = SimConfig(
            initial_cash=10_000.0,
            spread=FixedBpsSpread(bps=20.0),
            commission=SchemaCommission(CommissionSchema(kind="percent", value=0.001)),
        )
        result = Simulator(series, config).run(CompraUnaVez(qty=10.0))
        fill = result.fills[0]
        assert result.cash[1] == pytest.approx(
            10_000.0 - fill.qty_filled * fill.fill_price - fill.commission
        )

    def test_identidad_verificada_en_cada_barra(
        self, fractional: InstrumentSpec
    ) -> None:
        """La verificacion vive dentro del bucle, no solo en los tests."""
        series = make_series(fractional, [100.0, 101.0, 102.0])
        config = SimConfig(initial_cash=1_000.0, check_accounting=True)
        Simulator(series, config).run(CompraUnaVez(qty=1.0))


class TestGapSeparadoDeCostos:
    """El gap es valuacion, no costo. El log debe distinguirlos."""

    def test_gap_adverso_no_entra_en_los_costos(
        self, fractional: InstrumentSpec
    ) -> None:
        # close[0]=100, open[1]=110: la ejecucion ocurre 10 por encima del
        # precio que la estrategia vio al decidir.
        series = make_series(
            fractional, closes=[100.0, 110.0, 110.0], opens=[100.0, 110.0, 110.0]
        )
        config = SimConfig(initial_cash=10_000.0, spread=FixedBpsSpread(bps=10.0))
        result = Simulator(series, config).run(CompraUnaVez(qty=10.0))
        fill = result.fills[0]

        assert fill.gap == pytest.approx(10.0 * (110.0 - 100.0))
        assert fill.spread_cost == pytest.approx(10.0 * 110.0 * 10.0 / 2e4)
        assert fill.transaction_costs == pytest.approx(fill.spread_cost)
        assert fill.implementation_shortfall == pytest.approx(
            fill.gap + fill.spread_cost
        )

        desglose = result.cost_breakdown()
        assert desglose["gap"] == pytest.approx(100.0)
        assert desglose["total_costs_paid"] == pytest.approx(fill.spread_cost)
        assert desglose["total_costs_paid"] != desglose["implementation_shortfall"]

    def test_el_gap_puede_ser_favorable(self, fractional: InstrumentSpec) -> None:
        """Un gap a favor es negativo. Un costo nunca lo es."""
        series = make_series(
            fractional, closes=[100.0, 90.0, 90.0], opens=[100.0, 90.0, 90.0]
        )
        result = Simulator(
            series, SimConfig(initial_cash=10_000.0, spread=FixedBpsSpread(bps=10.0))
        ).run(CompraUnaVez(qty=10.0))
        fill = result.fills[0]
        assert fill.gap < 0
        assert fill.spread_cost > 0
        assert fill.transaction_costs > 0

    def test_gap_de_venta_tiene_el_signo_contrario(
        self, fractional: InstrumentSpec
    ) -> None:
        series = make_series(
            fractional,
            closes=[100.0, 100.0, 110.0, 110.0],
            opens=[100.0, 100.0, 110.0, 110.0],
        )

        class CompraYVende:
            name = "compra_y_vende"

            def reset(self, seed: int | None = None) -> None:
                return None

            def on_bar(self, view, account):
                if view.t == 0:
                    return MarketOrder(qty=10.0)
                if view.t == 1:
                    return MarketOrder(qty=-10.0)
                return None

        result = Simulator(series, SimConfig(initial_cash=10_000.0)).run(
            CompraYVende()
        )
        venta = result.fills[1]
        # Vender con el mercado en gap al alza es favorable: gap negativo.
        assert venta.qty_filled == -10.0
        assert venta.gap == pytest.approx(-10.0 * (110.0 - 100.0))

    def test_sin_gap_el_shortfall_son_solo_costos(
        self, fractional: InstrumentSpec
    ) -> None:
        series = make_series(fractional, [100.0, 100.0, 100.0])
        result = Simulator(
            series, SimConfig(initial_cash=10_000.0, spread=FixedBpsSpread(bps=10.0))
        ).run(CompraUnaVez(qty=10.0))
        fill = result.fills[0]
        assert fill.gap == pytest.approx(0.0)
        assert fill.implementation_shortfall == pytest.approx(fill.transaction_costs)


class TestSlippageAdverso:
    """El precio ejecutado nunca es mejor que el de referencia."""

    def test_compras_pagan_mas_y_ventas_cobran_menos(
        self, fractional: InstrumentSpec
    ) -> None:
        series = generate_gbm_sv(
            150, seed=5, instrument=fractional, params="high", s0=100.0
        )
        config = SimConfig(
            initial_cash=100_000.0,
            spread=CorwinSchultzSpread(),
            slippage=SqrtSlippage(k=0.1),
        )
        from agents.baselines import RandomAgent

        result = Simulator(series, config).run(RandomAgent(seed=1, trade_prob=0.5))
        assert len(result.executed_fills) > 5
        for fill in result.executed_fills:
            if fill.qty_filled > 0:
                assert fill.fill_price >= fill.ref_price
            else:
                assert fill.fill_price <= fill.ref_price
            assert fill.spread_cost >= 0.0
            assert fill.slippage_cost >= 0.0
            assert fill.commission >= 0.0

    def test_mayor_participacion_implica_mayor_slippage(
        self, fractional: InstrumentSpec
    ) -> None:
        series = make_series(fractional, [100.0] * 4, volume=100_000.0)
        config = SimConfig(initial_cash=1_000_000.0, slippage=LinearSlippage(k=0.5))
        precios = []
        for qty in (10.0, 100.0, 1_000.0, 5_000.0):
            result = Simulator(series, config).run(CompraUnaVez(qty=qty))
            precios.append(result.fills[0].fill_price)
        assert precios == sorted(precios)

    def test_modelos_con_k_negativo_son_rechazados(self) -> None:
        with pytest.raises(ValueError, match="adverso"):
            LinearSlippage(k=-0.1)
        with pytest.raises(ValueError, match="adverso"):
            SqrtSlippage(k=-0.1)


class TestInteresSobreCashOcioso:
    def test_cero_por_defecto(self, fractional: InstrumentSpec) -> None:
        series = make_series(fractional, [100.0] * 10)
        result = Simulator(series, SimConfig(initial_cash=1_000.0)).run(
            CompraUnaVez(qty=0.0)
        )
        assert result.interest.sum() == 0.0
        assert result.equity[-1] == 1_000.0

    def test_el_cash_ocioso_acumula_interes(self, fractional: InstrumentSpec) -> None:
        series = make_series(fractional, [100.0] * 253)

        class NoOpera:
            name = "no_opera"

            def reset(self, seed: int | None = None) -> None:
                return None

            def on_bar(self, view, account):
                return None

        config = SimConfig(initial_cash=10_000.0, cash_rate=0.05, bars_per_year=252.0)
        result = Simulator(series, config).run(NoOpera())
        # 252 barras de devengo sobre 253 marcas.
        assert result.equity[-1] == pytest.approx(10_500.0, rel=1e-3)
        assert result.cost_breakdown()["interest_earned"] > 0

    def test_estar_invertido_no_devenga_interes_sobre_lo_invertido(
        self, fractional: InstrumentSpec
    ) -> None:
        series = make_series(fractional, [100.0] * 60)
        config = SimConfig(initial_cash=10_000.0, cash_rate=0.05)
        invertido = Simulator(series, config).run(BuyAndHold())
        parado = Simulator(series, config).run(CompraUnaVez(qty=0.0))
        assert parado.interest.sum() > invertido.interest.sum()


class TestEquityDeLiquidacion:
    def test_liquidacion_es_menor_o_igual_que_la_marca_a_close(
        self, fractional: InstrumentSpec
    ) -> None:
        series = generate_gbm_sv(
            120, seed=13, instrument=fractional, params="high", s0=100.0
        )
        config = SimConfig(
            initial_cash=50_000.0,
            spread=FixedBpsSpread(bps=50.0),
            slippage=LinearSlippage(k=0.2),
            commission=SchemaCommission(CommissionSchema(kind="percent", value=0.001)),
        )
        result = Simulator(series, config).run(BuyAndHold())
        assert np.all(result.equity_liquidation <= result.equity + 1e-9)
        assert np.any(result.equity_liquidation < result.equity - 1e-9)

    def test_sin_posicion_ambas_series_coinciden(
        self, fractional: InstrumentSpec
    ) -> None:
        series = make_series(fractional, [100.0] * 5)
        result = Simulator(
            series, SimConfig(initial_cash=1_000.0, spread=FixedBpsSpread(bps=50.0))
        ).run(CompraUnaVez(qty=0.0))
        np.testing.assert_allclose(result.equity_liquidation, result.equity)

    def test_el_descuento_es_exactamente_el_costo_de_salir(
        self, fractional: InstrumentSpec
    ) -> None:
        """La serie de liquidacion vale lo que se cobraria al cerrar de verdad."""
        series = make_series(fractional, [100.0] * 4, volume=10_000.0)
        config = SimConfig(
            initial_cash=100_000.0,
            spread=FixedBpsSpread(bps=40.0),
            slippage=LinearSlippage(k=0.5),
            commission=SchemaCommission(CommissionSchema(kind="percent", value=0.002)),
        )
        result = Simulator(series, config).run(CompraUnaVez(qty=500.0))

        posicion, cash, close = result.position[-1], result.cash[-1], 100.0
        half_spread = close * 40.0 / 2e4
        impacto = close * 0.5 * (posicion / 10_000.0)
        precio_salida = close - half_spread - impacto
        esperado = cash + posicion * precio_salida - posicion * precio_salida * 0.002
        assert result.equity_liquidation[-1] == pytest.approx(esperado)

    def test_el_drawdown_de_liquidacion_no_es_mecanicamente_peor(
        self, fractional: InstrumentSpec
    ) -> None:
        """Con costos proporcionales, la serie de liquidacion es casi un
        reescalado del equity marcado a close, asi que su drawdown *relativo*
        puede ser incluso algo menor.

        Donde si difiere de forma material es cuando el tamano de la posicion
        cambia entre el pico y el valle: una estrategia invertida en el pico y
        plana en el valle paga el descuento solo en el pico. Ese es el caso que
        el tier alto debe reportar, no un "siempre peor" generico.
        """
        series = generate_gbm_sv(
            250, seed=21, instrument=fractional, params="high", s0=100.0
        )
        config = SimConfig(
            initial_cash=50_000.0,
            spread=FixedBpsSpread(bps=80.0),
            slippage=LinearSlippage(k=0.3),
        )
        result = Simulator(series, config).run(BuyAndHold())

        def max_dd(equity: np.ndarray) -> float:
            return float(np.max(1.0 - equity / np.maximum.accumulate(equity)))

        # Siempre por debajo en nivel...
        assert np.all(result.equity_liquidation <= result.equity + 1e-9)
        # ...pero el drawdown relativo es practicamente el mismo con posicion
        # constante. Afirmar lo contrario seria un test que pasa por casualidad.
        assert max_dd(result.equity_liquidation) == pytest.approx(
            max_dd(result.equity), abs=1e-3
        )


class TestConfiguracion:
    def test_allow_short_lanza_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="allow_short"):
            SimConfig(initial_cash=1_000.0, allow_short=True)

    def test_latencia_distinta_de_una_barra_no_soportada(self) -> None:
        with pytest.raises(NotImplementedError, match="latencia"):
            SimConfig(initial_cash=1_000.0, latency_bars=2)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"initial_cash": 0.0}, "initial_cash debe ser positivo"),
            ({"max_participation": 0.0}, "max_participation debe estar"),
            ({"max_participation": 1.5}, "max_participation debe estar"),
            ({"cash_rate": -0.01}, "cash_rate no puede ser negativo"),
        ],
    )
    def test_parametros_invalidos(
        self, kwargs: dict[str, object], match: str
    ) -> None:
        base: dict[str, object] = {"initial_cash": 1_000.0}
        with pytest.raises(ValueError, match=match):
            SimConfig(**{**base, **kwargs})  # type: ignore[arg-type]

    def test_la_config_se_serializa_con_el_resultado(
        self, fractional: InstrumentSpec
    ) -> None:
        series = make_series(fractional, [100.0, 101.0, 102.0])
        config = SimConfig(initial_cash=1_000.0, spread=FixedBpsSpread(bps=5.0))
        result = Simulator(series, config).run(CompraUnaVez(qty=1.0))
        assert result.config["spread"] == "fixed_bps"
        assert result.config["initial_cash"] == 1_000.0
        assert result.series_meta["symbol"] == "FRAC"
        assert result.strategy_name == "compra_una_vez"

    def test_serie_demasiado_corta(self, fractional: InstrumentSpec) -> None:
        series = make_series(fractional, [100.0, 101.0])
        Simulator(series, SimConfig(initial_cash=100.0))  # 2 barras: minimo
        with pytest.raises(ValueError, match="al menos 2 barras"):
            Simulator(series.slice(0, 1), SimConfig(initial_cash=100.0))

    def test_on_bar_debe_devolver_market_order(
        self, fractional: InstrumentSpec
    ) -> None:
        series = make_series(fractional, [100.0, 101.0, 102.0])

        class Rota:
            name = "rota"

            def reset(self, seed: int | None = None) -> None:
                return None

            def on_bar(self, view, account):
                return 42

        with pytest.raises(TypeError, match="MarketOrder"):
            Simulator(series, SimConfig(initial_cash=100.0)).run(Rota())


class TestVentaSinPosicion:
    def test_vender_sin_posicion_se_rechaza(self, fractional: InstrumentSpec) -> None:
        series = make_series(fractional, [100.0, 100.0, 100.0])
        result = Simulator(series, SimConfig(initial_cash=1_000.0)).run(
            CompraUnaVez(qty=-5.0)
        )
        fill = result.fills[0]
        assert fill.status is OrderStatus.REJECTED
        assert fill.reject_reason is RejectReason.SHORT_NOT_ALLOWED
        assert result.position[-1] == 0.0

    def test_vender_mas_de_lo_que_se_tiene_se_rechaza(
        self, fractional: InstrumentSpec
    ) -> None:
        series = make_series(fractional, [100.0] * 5)

        class CompraYSobrevende:
            name = "sobrevende"

            def reset(self, seed: int | None = None) -> None:
                return None

            def on_bar(self, view, account):
                if view.t == 0:
                    return MarketOrder(qty=1.0)
                if view.t == 2:
                    return MarketOrder(qty=-10.0)
                return None

        result = Simulator(series, SimConfig(initial_cash=1_000.0)).run(
            CompraYSobrevende()
        )
        assert result.fills[1].reject_reason is RejectReason.INSUFFICIENT_POSITION
        assert result.position[-1] == pytest.approx(1.0)

    def test_barra_sin_volumen_rechaza(self, fractional: InstrumentSpec) -> None:
        series = make_series(
            fractional, [100.0, 100.0, 100.0], volume=np.array([1e6, 0.0, 1e6])
        )
        result = Simulator(series, SimConfig(initial_cash=1_000.0)).run(
            CompraUnaVez(qty=1.0)
        )
        assert result.fills[0].reject_reason is RejectReason.NO_VOLUME
