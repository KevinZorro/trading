"""La capa de riesgo aplicada dentro del motor.

Verifica el punto de aplicacion, no los limites en si (eso es
``test_layer.py``): que el veto ocurra entre la decision y el envio al venue,
que la orden vetada no produzca fill y que quede registrada.
"""

from __future__ import annotations

import numpy as np
import pytest

from agents.baselines import BuyAndHold
from data.instruments import InstrumentSpec
from risk.layer import RiskLayer, RiskLimits, RiskRejectReason
from sim.clock import SimulatedClock
from sim.engine import SimConfig, SimResult, Simulator
from sim.orders import MarketOrder

from ..sim.conftest import make_series


class CompraFija:
    """Pide la misma cantidad en cada barra. Sin autocensura."""

    name = "compra_fija"

    def __init__(self, qty: float) -> None:
        self.qty = qty

    def reset(self, seed: int | None = None) -> None:
        return None

    def on_bar(self, view, account):
        return MarketOrder(qty=self.qty, tag="fija")


@pytest.fixture
def serie(fractional: InstrumentSpec):
    return make_series(fractional, [100.0] * 6, volume=1_000_000.0)


def correr(serie, gate: RiskLayer | None, qty: float = 10.0) -> SimResult:
    sim = Simulator(serie, SimConfig(initial_cash=100_000.0))
    return sim.run(CompraFija(qty), gate=gate)


class TestSinCapaDeRiesgo:
    def test_el_comportamiento_por_defecto_no_cambia(self, serie) -> None:
        resultado = correr(serie, gate=None)
        assert resultado.gate_rejections == []
        assert len(resultado.executed_fills) > 0


class TestVetoDentroDelMotor:
    def _gate(self, **kwargs: object) -> RiskLayer:
        return RiskLayer(
            RiskLimits(**kwargs),  # type: ignore[arg-type]
            SimulatedClock(np.datetime64("2020-01-01T21:00:00", "ns")),
        )

    def test_la_orden_vetada_no_llega_al_venue(self, serie) -> None:
        # 10 unidades a 100 = 1000 de nocional, sobre un limite de 500.
        resultado = correr(serie, self._gate(max_position_notional=500.0))
        assert resultado.fills == []
        assert len(resultado.gate_rejections) > 0

    def test_el_veto_queda_registrado_con_motivo(self, serie) -> None:
        resultado = correr(serie, self._gate(max_position_notional=500.0))
        veto = resultado.gate_rejections[0]
        assert veto.reason == RiskRejectReason.MAX_POSITION_NOTIONAL
        assert veto.qty_requested == pytest.approx(10.0)
        assert veto.symbol == "FRAC"
        assert veto.tag == "fija"
        assert veto.client_order_id != ""

    def test_los_vetos_no_se_mezclan_con_los_rechazos_del_venue(self, serie) -> None:
        """Dos diagnosticos distintos, dos listas distintas.

        "el venue no pudo llenarla" y "nuestra capa de riesgo no la dejo salir"
        son problemas diferentes; agregarlos en una sola cifra los volveria
        indistinguibles.
        """
        resultado = correr(serie, self._gate(max_position_notional=500.0))
        assert resultado.rejected_fills == []
        assert resultado.gate_rejections != []
        desglose = resultado.cost_breakdown()
        assert desglose["n_rejected"] == 0
        assert desglose["n_gate_rejected"] == len(resultado.gate_rejections)

    def test_el_identificador_se_consume_aunque_la_orden_no_salga(self, serie) -> None:
        """El emisor genera el ID antes del veto, como en vivo: la capa de
        riesgo vetа una orden que ya existe, no un borrador."""
        resultado = correr(serie, self._gate(max_position_notional=500.0))
        ids = [v.client_order_id for v in resultado.gate_rejections]
        assert len(set(ids)) == len(ids)
        assert ids[0] == "sim-00000000"

    def test_una_orden_dentro_del_limite_pasa(self, serie) -> None:
        resultado = correr(serie, self._gate(max_position_notional=5_000.0), qty=1.0)
        assert len(resultado.executed_fills) > 0
        assert resultado.gate_rejections == []


class TestKillSwitchEnLaCorrida:
    def test_una_perdida_grande_corta_la_operacion(self, fractional) -> None:
        """El precio se desploma: el limite diario dispara el kill switch."""
        serie = make_series(
            fractional, [100.0, 100.0, 50.0, 50.0, 50.0], volume=1_000_000.0
        )
        gate = RiskLayer(
            RiskLimits(max_daily_loss_pct=0.10),
            SimulatedClock(np.datetime64("2020-01-01T21:00:00", "ns")),
        )
        sim = Simulator(serie, SimConfig(initial_cash=100_000.0))
        resultado = sim.run(BuyAndHold(), gate=gate)

        assert gate.kill_switch_active
        assert any(e.kind == "kill_switch_tripped" for e in gate.events)
        # BuyAndHold no vuelve a operar, asi que el corte se ve en el log de la
        # capa, no en ordenes vetadas.
        assert isinstance(resultado.equity[-1], float)

    def test_tras_el_corte_solo_pasan_las_ventas(self, fractional) -> None:
        serie = make_series(
            fractional, [100.0, 100.0, 50.0, 50.0, 50.0], volume=1_000_000.0
        )
        gate = RiskLayer(
            RiskLimits(max_daily_loss_pct=0.10),
            SimulatedClock(np.datetime64("2020-01-01T21:00:00", "ns")),
        )
        sim = Simulator(serie, SimConfig(initial_cash=100_000.0))
        resultado = sim.run(CompraFija(900.0), gate=gate)

        assert gate.kill_switch_active
        motivos = {v.reason for v in resultado.gate_rejections}
        assert RiskRejectReason.KILL_SWITCH in motivos
        # Ninguna compra se ejecuto despues del corte.
        despues = [
            f for f in resultado.executed_fills if f.t_fill > 2 and f.qty_filled > 0
        ]
        assert despues == []
