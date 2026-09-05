"""El sizer por peso objetivo. Vive en sim/ porque dimensionar es ejecucion.

Los dos comportamientos que este archivo fija son los que impiden que el entorno
de la Etapa 2 se autocensure: no se pre-redondea a cero y no se recorta contra
el cash. Si alguna de las dos se relajara, "no quise operar" y "no pude" se
volverian indistinguibles en el log.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.instruments import CommissionSchema, InstrumentSpec, us_equity_spec
from sim.engine import LONG_ONLY_ACTION_RANGE
from sim.sizing import TargetWeightSizer
from sim.view import AccountSnapshot, MarketView

from .conftest import make_series


@pytest.fixture
def vista() -> MarketView:
    spec = us_equity_spec("TEST", commission=CommissionSchema(kind="fixed", value=0.0))
    return MarketView(make_series(spec, [100.0] * 10), 5)


def cuenta(*, equity: float = 10_000.0, position: float = 0.0) -> AccountSnapshot:
    return AccountSnapshot(
        t=5,
        cash=equity - position * 100.0,
        position=position,
        mark_price=100.0,
        equity=equity,
        pending_qty=0.0,
    )


class TestConversionDePeso:
    def test_peso_cero_desde_cero_no_hace_nada(self, vista) -> None:
        assert TargetWeightSizer().order_for(0.0, vista, cuenta()) is None

    def test_peso_calculado_a_mano(self, vista) -> None:
        # equity 10_000, precio 100, safety 0.98, peso 0.5
        # objetivo = 0.5 * 10_000 * 0.98 / 100 = 49 unidades
        # posicion actual 0  ->  delta = 49
        orden = TargetWeightSizer(safety=0.98).order_for(0.5, vista, cuenta())
        assert orden is not None
        assert orden.qty == pytest.approx(49.0)

    def test_el_delta_descuenta_la_posicion_existente(self, vista) -> None:
        # objetivo = 1.0 * 10_000 * 0.98 / 100 = 98 ; posicion 30 -> delta 68
        orden = TargetWeightSizer().order_for(1.0, vista, cuenta(position=30.0))
        assert orden is not None
        assert orden.qty == pytest.approx(68.0)

    def test_bajar_el_peso_produce_una_venta(self, vista) -> None:
        # objetivo = 0.2 * 10_000 * 0.98 / 100 = 19.6 ; posicion 50 -> delta -30.4
        orden = TargetWeightSizer().order_for(0.2, vista, cuenta(position=50.0))
        assert orden is not None
        assert orden.qty == pytest.approx(-30.4)

    def test_peso_cero_con_posicion_la_cierra_entera(self, vista) -> None:
        """El safety no toca la salida: un peso de 0 vende todo."""
        orden = TargetWeightSizer().order_for(0.0, vista, cuenta(position=42.0))
        assert orden is not None
        assert orden.qty == pytest.approx(-42.0)

    def test_el_safety_deja_margen_para_el_gap_y_la_comision(self, vista) -> None:
        """Un peso de 1.0 sin margen se rechazaria por cash casi siempre.

        Estar exactamente all-in exigiria conocer el precio de ejecucion antes
        de enviar la orden, o sea lookahead.
        """
        sin_margen = TargetWeightSizer(safety=1.0).order_for(1.0, vista, cuenta())
        con_margen = TargetWeightSizer(safety=0.98).order_for(1.0, vista, cuenta())
        assert sin_margen is not None
        assert con_margen is not None
        assert con_margen.qty < sin_margen.qty
        assert con_margen.qty * 100.0 == pytest.approx(9_800.0)


class TestNoSeAutocensura:
    def test_no_pre_redondea_a_cero(self, vista) -> None:
        """Un delta menor que el lote se envia igual.

        ``round_qty`` redondea hacia cero, asi que redondear aca convertiria la
        orden en ``qty=0`` y desapareceria del log. El rechazo por
        ZERO_AFTER_ROUNDING lo pone el venue, y ahi queda registrado.
        """
        spec = InstrumentSpec(
            symbol="ENTERO",
            venue="TEST",
            tick_size=0.01,
            lot_size=1.0,
            allow_fractional=False,
            qty_precision=0,
            min_order_qty=1.0,
            min_notional=0.0,
            commission_schema=CommissionSchema(kind="fixed", value=0.0),
        )
        vista_entera = MarketView(make_series(spec, [100.0] * 10), 5)
        orden = TargetWeightSizer().order_for(0.001, vista_entera, cuenta())
        assert orden is not None
        # 0.001 * 10_000 * 0.98 / 100 = 0.098 unidades: por debajo del lote.
        assert orden.qty == pytest.approx(0.098)
        assert spec.round_qty(orden.qty) == 0.0

    def test_no_recorta_contra_el_cash(self, vista) -> None:
        """Si no entra, lo rechaza el venue con INSUFFICIENT_CASH.

        Recortarlo aca seria redimensionar en silencio, que es lo que
        ``check_tradable`` y la ``RiskLayer`` tienen prohibido.
        """
        # Cuenta con equity alto pero casi todo en posicion: el cash no alcanza.
        apretada = AccountSnapshot(
            t=5,
            cash=10.0,
            position=99.9,
            mark_price=100.0,
            equity=10_000.0,
            pending_qty=0.0,
        )
        orden = TargetWeightSizer(safety=1.0).order_for(1.0, vista, apretada)
        assert orden is not None
        # Pide 0.1 unidades = 10 de nocional, pero la comision no entra en 10 de
        # cash. El sizer manda igual; el venue decide.
        assert orden.qty > 0.0


class TestBandaMuerta:
    def test_por_defecto_no_hay_banda(self, vista) -> None:
        """deadband=0.0: cualquier delta distinto de cero se envia."""
        sizer = TargetWeightSizer()
        assert sizer.deadband == 0.0
        orden = sizer.order_for(1e-9, vista, cuenta())
        assert orden is not None
        assert orden.qty > 0.0

    def test_la_banda_suprime_el_micro_rebalanceo(self, vista) -> None:
        # deadband 0.01 = 1% del equity = 100 de nocional.
        # objetivo 0.5 -> 49 unidades = 4_900 ; posicion 48.7 -> delta 0.3 = 30
        sizer = TargetWeightSizer(deadband=0.01)
        assert sizer.order_for(0.5, vista, cuenta(position=48.7)) is None

    def test_la_banda_deja_pasar_lo_que_la_supera(self, vista) -> None:
        sizer = TargetWeightSizer(deadband=0.01)
        # posicion 40 -> delta 9 unidades = 900 de nocional, sobre los 100.
        orden = sizer.order_for(0.5, vista, cuenta(position=40.0))
        assert orden is not None

    def test_la_banda_se_serializa_con_la_corrida(self) -> None:
        """Una banda escondida en el codigo seria autocensura; en la config, no."""
        d = TargetWeightSizer(safety=0.95, deadband=0.02).describe()
        assert d == {"safety": 0.95, "deadband": 0.02, "tag": "target_weight"}


class TestValidacion:
    def test_peso_fuera_del_rango_del_motor(self, vista) -> None:
        sizer = TargetWeightSizer()
        bajo, alto = LONG_ONLY_ACTION_RANGE
        with pytest.raises(ValueError, match="fuera de"):
            sizer.order_for(alto + 0.01, vista, cuenta())
        with pytest.raises(ValueError, match="fuera de"):
            sizer.order_for(bajo - 0.01, vista, cuenta())

    def test_el_recorte_es_responsabilidad_de_quien_llama(self, vista) -> None:
        """El sizer no recorta en silencio: lanza y obliga a registrar el recorte."""
        with pytest.raises(ValueError, match="tiene que quedar registrado"):
            TargetWeightSizer().order_for(2.0, vista, cuenta())

    def test_safety_invalido(self) -> None:
        with pytest.raises(ValueError, match="safety debe estar"):
            TargetWeightSizer(safety=0.0)
        with pytest.raises(ValueError, match="safety debe estar"):
            TargetWeightSizer(safety=1.5)

    def test_deadband_negativo(self) -> None:
        with pytest.raises(ValueError, match="deadband no puede ser negativo"):
            TargetWeightSizer(deadband=-0.01)

    def test_precio_no_positivo(self) -> None:
        """Defensa en profundidad contra datos que esquivaron la validacion.

        Una ``BarSeries`` construida por el camino normal no puede tener precios
        no positivos -``data.validation`` los rechaza- asi que el caso se fuerza
        con un doble. El guard existe igual: dividir por cero aca produciria una
        cantidad infinita que el venue intentaria ejecutar.
        """

        class VistaCorrupta:
            def close(self, lookback: int = 0) -> float:
                return 0.0

        with pytest.raises(ValueError, match="precio no positivo"):
            TargetWeightSizer().order_for(0.5, VistaCorrupta(), cuenta())


class TestEtiqueta:
    def test_la_orden_lleva_la_etiqueta_configurada(self, vista) -> None:
        orden = TargetWeightSizer(tag="agente_a").order_for(0.5, vista, cuenta())
        assert orden is not None
        assert orden.tag == "agente_a"

    def test_la_cantidad_es_finita(self, vista) -> None:
        orden = TargetWeightSizer().order_for(0.5, vista, cuenta())
        assert orden is not None
        assert np.isfinite(orden.qty)
