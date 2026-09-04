"""Reglas de ejecucion por instrumento.

El test que mas importa aqui es el que separa cantidad minima de nocional
minimo: es la diferencia entre concluir que 10 USD opera bien en BTC y saber
que el venue rechaza la orden.
"""

from __future__ import annotations

import pytest

from data.instruments import (
    CommissionSchema,
    InstrumentSpec,
    RejectReason,
    binance_spot_spec,
    us_equity_spec,
)


class TestRedondeo:
    def test_acciones_enteras_redondean_hacia_cero(self) -> None:
        spec = us_equity_spec("AAPL")
        assert spec.round_qty(3.9) == 3.0
        assert spec.round_qty(0.9) == 0.0
        assert spec.round_qty(-3.9) == -3.0  # hacia cero, no hacia abajo

    def test_nunca_redondea_hacia_arriba(self) -> None:
        spec = us_equity_spec("AAPL")
        for qty in (0.999, 1.999, 10.9999):
            assert abs(spec.round_qty(qty)) <= abs(qty)

    def test_cripto_respeta_step_size(self) -> None:
        spec = binance_spot_spec("BTCUSDT", step_size=1e-5)
        assert spec.round_qty(0.000123456) == pytest.approx(0.00012, abs=1e-12)

    def test_tolerancia_en_multiplos_exactos(self) -> None:
        # 0.1 + 0.2 == 0.30000000000000004: sin tolerancia esto devolveria 0.2
        spec = binance_spot_spec("ETHUSDT", step_size=0.1)
        assert spec.round_qty(0.1 + 0.2) == pytest.approx(0.3, abs=1e-12)

    def test_precio_al_tick(self) -> None:
        spec = us_equity_spec("AAPL")
        assert spec.round_price(10.004) == pytest.approx(10.00)
        assert spec.round_price(10.006) == pytest.approx(10.01)


class TestAdmisibilidad:
    def test_accion_con_capital_bajo_rechaza_por_cantidad(self) -> None:
        """10 USD contra una accion de 400 USD: no alcanza para una unidad."""
        spec = us_equity_spec("AAPL")
        qty, reason = spec.check_tradable(10.0 / 400.0, price=400.0)
        assert qty == 0.0
        assert reason is RejectReason.ZERO_AFTER_ROUNDING

    def test_cripto_con_capital_bajo_rechaza_por_nocional_no_por_cantidad(self) -> None:
        """El binding constraint en cripto es el nocional, y hay que verlo.

        0.0001 BTC a 50k USD son 5 USD: supera de sobra el step size de 1e-5,
        asi que un modelo que solo mirara min_order_qty la aceptaria.
        """
        spec = binance_spot_spec("BTCUSDT", min_notional=10.0)
        qty, reason = spec.check_tradable(0.0001, price=50_000.0)
        assert qty == pytest.approx(0.0001)
        assert abs(qty) > spec.min_order_qty  # la cantidad minima NO es la que muerde
        assert reason is RejectReason.MIN_NOTIONAL

    def test_cripto_por_encima_del_nocional_minimo_se_acepta(self) -> None:
        spec = binance_spot_spec("BTCUSDT", min_notional=10.0)
        qty, reason = spec.check_tradable(0.0005, price=50_000.0)
        assert reason is None
        assert qty == pytest.approx(0.0005)

    def test_ambas_restricciones_se_evaluan_por_separado(self) -> None:
        spec = InstrumentSpec(
            symbol="X",
            venue="TEST",
            tick_size=0.01,
            lot_size=0.001,
            allow_fractional=True,
            qty_precision=3,
            min_order_qty=0.5,
            min_notional=100.0,
            commission_schema=CommissionSchema(),
            asset_class="crypto",
        )
        # Cantidad insuficiente: muerde min_order_qty aunque el nocional sobre.
        _, reason = spec.check_tradable(0.1, price=10_000.0)
        assert reason is RejectReason.MIN_QTY
        # Cantidad suficiente, nocional insuficiente.
        _, reason = spec.check_tradable(1.0, price=50.0)
        assert reason is RejectReason.MIN_NOTIONAL
        # Ambas satisfechas.
        _, reason = spec.check_tradable(1.0, price=200.0)
        assert reason is None

    def test_precio_no_positivo_es_error_de_programacion(self) -> None:
        spec = us_equity_spec("AAPL")
        with pytest.raises(ValueError, match="precio no positivo"):
            spec.check_tradable(1.0, price=0.0)


class TestSpecInvalida:
    def test_sin_fraccionales_exige_lote_entero(self) -> None:
        with pytest.raises(ValueError, match="allow_fractional"):
            InstrumentSpec(
                symbol="X",
                venue="TEST",
                tick_size=0.01,
                lot_size=0.001,
                allow_fractional=False,
                qty_precision=0,
                min_order_qty=1.0,
                min_notional=0.0,
                commission_schema=CommissionSchema(),
            )

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"tick_size": 0.0}, "tick_size debe ser positivo"),
            ({"lot_size": 0.0}, "lot_size debe ser positivo"),
            ({"qty_precision": -1}, "qty_precision no puede ser negativa"),
            ({"min_notional": -1.0}, "los minimos no pueden ser negativos"),
        ],
    )
    def test_parametros_invalidos(self, kwargs: dict[str, object], match: str) -> None:
        base: dict[str, object] = {
            "symbol": "X",
            "venue": "TEST",
            "tick_size": 0.01,
            "lot_size": 1.0,
            "allow_fractional": False,
            "qty_precision": 0,
            "min_order_qty": 1.0,
            "min_notional": 0.0,
            "commission_schema": CommissionSchema(),
        }
        # El match no es cosmetico: sin el, un cambio que hiciera fallar la
        # construccion por otro motivo dejaria el test en verde igual.
        with pytest.raises(ValueError, match=match):
            InstrumentSpec(**{**base, **kwargs})  # type: ignore[arg-type]

    def test_comision_invalida(self) -> None:
        with pytest.raises(ValueError, match="kind de comision desconocido"):
            CommissionSchema(kind="porcentual")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="value de comision no puede ser"):
            CommissionSchema(kind="percent", value=-0.1)
