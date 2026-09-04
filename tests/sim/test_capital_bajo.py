"""Capital bajo contra minimos de orden.

Es el escenario que el barrido de la Etapa 6 mide, y el que mas facilmente
produce un backtest mentiroso: si el simulador redimensiona en silencio o
redondea hacia arriba, aparecen ejecuciones fantasma que ningun broker habria
aceptado y el estudio concluye que 10 USD operan bien.
"""

from __future__ import annotations

import numpy as np
import pytest

from agents.baselines import BuyAndHold
from data.instruments import CommissionSchema, binance_spot_spec, us_equity_spec
from sim.costs import SchemaCommission
from sim.engine import SimConfig, Simulator
from sim.orders import MarketOrder, OrderStatus, RejectReason

from .conftest import make_series


class Intento:
    """Envia una unica orden de ``qty`` en la primera barra."""

    name = "intento"

    def __init__(self, qty: float) -> None:
        self.qty = qty

    def reset(self, seed: int | None = None) -> None:
        return None

    def on_bar(self, view, account):
        return MarketOrder(qty=self.qty) if view.t == 0 else None


class TestAccionesEnteras:
    def test_diez_dolares_contra_una_accion_de_cuatrocientos(self) -> None:
        """No alcanza para una unidad: rechazo, no una fraccion fantasma."""
        spec = us_equity_spec("CARA")
        series = make_series(spec, [400.0, 400.0, 400.0])
        result = Simulator(series, SimConfig(initial_cash=10.0)).run(BuyAndHold())

        assert result.position[-1] == 0.0
        assert result.cash[-1] == 10.0
        assert result.equity[-1] == 10.0
        assert result.executed_fills == []
        # La orden de la ultima barra expira (no hay barra donde ejecutarla);
        # todas las demas se rechazan por no alcanzar una accion entera.
        assert {f.status for f in result.fills} == {
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }
        assert {f.reject_reason for f in result.rejected_fills} == {
            RejectReason.ZERO_AFTER_ROUNDING
        }

    def test_la_orden_rechazada_queda_en_el_log_con_su_motivo(self) -> None:
        """Un rechazo silencioso es peor que un rechazo: no se puede auditar."""
        spec = us_equity_spec("CARA")
        series = make_series(spec, [400.0, 400.0])
        result = Simulator(series, SimConfig(initial_cash=10.0)).run(Intento(qty=0.5))
        assert len(result.fills) == 1
        fill = result.fills[0]
        assert fill.qty_requested == 0.5
        assert fill.qty_filled == 0.0
        assert fill.reject_reason is RejectReason.ZERO_AFTER_ROUNDING
        assert fill.transaction_costs == 0.0

    def test_nunca_se_redondea_hacia_arriba(self) -> None:
        """1.9 acciones se ejecutan como 1, jamas como 2."""
        spec = us_equity_spec("TEST")
        series = make_series(spec, [100.0, 100.0, 100.0])
        result = Simulator(series, SimConfig(initial_cash=10_000.0)).run(
            Intento(qty=1.9)
        )
        assert result.fills[0].qty_filled == 1.0

    def test_capital_justo_por_debajo_de_una_accion(self) -> None:
        spec = us_equity_spec(
            "TEST", commission=CommissionSchema(kind="fixed", value=0.0)
        )
        series = make_series(spec, [100.0, 100.0, 100.0])
        result = Simulator(series, SimConfig(initial_cash=99.0)).run(BuyAndHold())
        assert result.position[-1] == 0.0
        assert result.equity[-1] == 99.0

    def test_capital_justo_por_encima_ejecuta_una_accion(self) -> None:
        spec = us_equity_spec(
            "TEST", commission=CommissionSchema(kind="fixed", value=0.0)
        )
        series = make_series(spec, [100.0, 100.0, 100.0])
        result = Simulator(series, SimConfig(initial_cash=105.0)).run(BuyAndHold())
        assert result.position[-1] == 1.0
        assert result.cash[-1] == pytest.approx(5.0)


class TestCriptoNocionalMinimo:
    """En cripto el binding constraint es el nocional, no la cantidad."""

    def test_cinco_dolares_en_bitcoin_se_rechaza_por_nocional(self) -> None:
        spec = binance_spot_spec("BTCUSDT", min_notional=10.0)
        series = make_series(spec, [50_000.0, 50_000.0, 50_000.0])
        result = Simulator(
            series,
            SimConfig(
                initial_cash=5.0,
                commission=SchemaCommission(
                    CommissionSchema(kind="percent", value=0.001)
                ),
            ),
        ).run(BuyAndHold())

        assert result.position[-1] == 0.0
        assert result.executed_fills == []
        motivos = {f.reject_reason for f in result.rejected_fills}
        assert motivos == {RejectReason.MIN_NOTIONAL}

    def test_el_motivo_del_rechazo_no_es_la_cantidad_minima(self) -> None:
        """La cantidad supera de sobra el step size: si solo se modelara
        ``min_order_qty``, esta orden se habria ejecutado."""
        spec = binance_spot_spec("BTCUSDT", step_size=1e-5, min_notional=10.0)
        series = make_series(spec, [50_000.0, 50_000.0])
        result = Simulator(series, SimConfig(initial_cash=6.0)).run(
            Intento(qty=6.0 / 50_000.0)
        )
        fill = result.fills[0]
        assert fill.reject_reason is RejectReason.MIN_NOTIONAL
        # 0.00012 BTC == 12x el step size del venue.
        assert abs(spec.round_qty(6.0 / 50_000.0)) > 10 * spec.min_order_qty

    def test_cien_dolares_en_bitcoin_si_opera(self) -> None:
        """El tier de riesgo alto es testeable con 100 USD gracias a los
        fraccionales; sin ellos el regimen entero seria intestable."""
        spec = binance_spot_spec("BTCUSDT", min_notional=10.0)
        series = make_series(spec, [50_000.0, 50_000.0, 51_000.0])
        result = Simulator(
            series,
            SimConfig(
                initial_cash=100.0,
                commission=SchemaCommission(
                    CommissionSchema(kind="percent", value=0.001)
                ),
            ),
        ).run(BuyAndHold())

        assert result.position[-1] > 0
        assert result.equity[-1] > 100.0

    def test_frontera_exacta_del_nocional(self) -> None:
        spec = binance_spot_spec("BTCUSDT", min_notional=10.0)
        series = make_series(spec, [1_000.0, 1_000.0])
        justo_debajo = Simulator(series, SimConfig(initial_cash=10_000.0)).run(
            Intento(qty=0.00999)
        )
        justo_encima = Simulator(series, SimConfig(initial_cash=10_000.0)).run(
            Intento(qty=0.01001)
        )
        assert justo_debajo.fills[0].reject_reason is RejectReason.MIN_NOTIONAL
        assert justo_encima.fills[0].status is OrderStatus.FILLED


class TestCashInsuficiente:
    def test_la_comision_puede_hacer_que_la_orden_no_quepa(self) -> None:
        """Sin redimensionado silencioso: si no cabe con comision, se rechaza."""
        spec = us_equity_spec(
            "TEST", commission=CommissionSchema(kind="fixed", value=5.0)
        )
        series = make_series(spec, [100.0, 100.0, 100.0])
        result = Simulator(series, SimConfig(initial_cash=101.0)).run(Intento(qty=1.0))
        assert result.fills[0].reject_reason is RejectReason.INSUFFICIENT_CASH
        assert result.cash[-1] == 101.0

    def test_un_gap_al_alza_puede_dejar_la_orden_sin_cash(self) -> None:
        """Se dimensiona con close[t] y se ejecuta al open[t+1], que puede ser
        mas alto. Es la razon por la que buy-and-hold no puede ir al 100%."""
        spec = us_equity_spec(
            "TEST", commission=CommissionSchema(kind="fixed", value=0.0)
        )
        series = make_series(
            spec, closes=[100.0, 200.0, 200.0], opens=[100.0, 200.0, 200.0]
        )
        result = Simulator(series, SimConfig(initial_cash=150.0)).run(Intento(qty=1.0))
        assert result.fills[0].reject_reason is RejectReason.INSUFFICIENT_CASH

    def test_no_hay_ejecuciones_fantasma_en_ningun_rechazo(self) -> None:
        """Invariante global: un rechazo no mueve cash ni posicion."""
        spec = us_equity_spec("CARA")
        series = make_series(spec, [400.0] * 20)
        result = Simulator(series, SimConfig(initial_cash=10.0)).run(BuyAndHold())
        assert np.all(result.cash == 10.0)
        assert np.all(result.position == 0.0)
        assert result.cost_breakdown()["total_costs_paid"] == 0.0
        assert result.cost_breakdown()["n_fills"] == 0
        assert result.cost_breakdown()["n_rejected"] == len(result.fills) - 1


class TestBarridoDeCapital:
    """Forma reducida del barrido de la Etapa 6."""

    @pytest.mark.parametrize(
        ("capital", "opera"),
        [(10.0, False), (100.0, False), (500.0, True), (10_000.0, True)],
    )
    def test_umbral_de_operatividad_en_acciones(
        self, capital: float, opera: bool
    ) -> None:
        spec = us_equity_spec(
            "CARA", commission=CommissionSchema(kind="fixed", value=1.0)
        )
        series = make_series(spec, [400.0] * 10)
        result = Simulator(series, SimConfig(initial_cash=capital)).run(BuyAndHold())
        assert bool(result.position[-1] > 0) is opera

    def test_los_costos_fijos_pesan_mas_con_capital_bajo(self) -> None:
        """La comision minima es la que destruye la ventaja con poco capital."""
        spec = us_equity_spec(
            "TEST",
            commission=CommissionSchema(
                kind="per_share", value=0.005, minimum=1.0
            ),
        )
        series = make_series(spec, [10.0] * 10)
        config = SimConfig(initial_cash=0.0 + 1.0)  # placeholder, se reemplaza abajo

        pesos = []
        for capital in (200.0, 2_000.0, 20_000.0):
            config = SimConfig(initial_cash=capital)
            result = Simulator(series, config).run(BuyAndHold())
            costos = result.cost_breakdown()["total_costs_paid"]
            pesos.append(costos / capital)
        assert pesos == sorted(pesos, reverse=True)
