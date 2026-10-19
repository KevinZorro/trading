"""AgentStep sobre las grabaciones reales de Binance. Oraculos calculados a mano."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from agents.ppo import SB3Policy
from data.instruments import CommissionSchema
from envs.observation import ObservationBuilder
from features.scaler import FeatureScaler
from live.agent_step import AgentStep, StaleBarsError, liquidation_equity
from live.fillstore import FillStore, venue_header
from live.frozen import FrozenPolicy
from live.venue import ShadowVenue
from sim.clock import SimulatedClock
from sim.costs import SchemaCommission
from sim.gate import GateDecision
from sim.orders import MarketOrder, OrderStatus
from sim.sizing import TargetWeightSizer
from sim.view import AccountSnapshot

from .conftest import RESPUESTAS, Servidor, cliente, grabacion

# Cierre de la ultima barra cerrada grabada: open_time 2026-10-16 + 1 dia.
CIERRE = np.datetime64("2026-10-17T00:00:00", "ns")
AHORA = CIERRE + np.timedelta64(30, "s")


class Modelo:
    def __init__(self, accion: float = 1.0) -> None:
        self.accion = accion
        self.vistas: list[Any] = []

    def predict(self, obs: Any, deterministic: bool) -> tuple[Any, None]:
        self.vistas.append(np.array(obs))
        return np.array([self.accion]), None


class Puerta:
    def __init__(self, veto: str | None = None) -> None:
        self.veto = veto
        self.fills: list[Any] = []
        self.barras = 0

    def check(self, order: MarketOrder, account: AccountSnapshot) -> GateDecision:
        if self.veto:
            return GateDecision(False, self.veto, "prueba")
        return GateDecision(True)

    def observe_bar(self, account: AccountSnapshot) -> None:
        self.barras += 1

    def observe_fill(self, fill: Any) -> None:
        self.fills.append(fill)


def armar(
    tmp: Path,
    modelo: Modelo,
    *,
    gate: Puerta | None = None,
    servidor: Servidor | None = None,
    ahora: np.datetime64 = AHORA,
) -> tuple[AgentStep, ShadowVenue, Puerta]:
    reloj = SimulatedClock(ahora)
    mercado = cliente(servidor, ahora)
    spec = mercado.instrument("BTCUSDT")
    builder = ObservationBuilder()
    agente = FrozenPolicy(
        seed=11,
        policy=SB3Policy(modelo, recurrent=False),
        scaler=FeatureScaler.identity(builder.names),
        config={},
    )
    cabecera = venue_header(
        initial_cash=100.0, spec_sha256="a" * 64, spec_received_at=AHORA
    )
    store = FillStore(tmp, cabecera)
    venue = ShadowVenue(
        market=mercado, instrument=spec, clock=reloj, store=store, initial_cash=100.0
    )
    puerta = gate or Puerta()
    paso = AgentStep(
        agent=agente,
        builder=builder,
        sizer=TargetWeightSizer(),
        gate=puerta,
        venue=venue,
        store=store,
        market=mercado,
        symbol="BTCUSDT",
        commission=SchemaCommission(schema=spec.commission_schema),
        clock=reloj,
        run_id="A11",
    )
    return paso, venue, puerta


def mejor_ask() -> float:
    return float(grabacion("depth.json")["asks"][0][0])


def cierre_grabado() -> float:
    return float(grabacion("klines_1d.json")[-2][4])


def test_compra_con_el_98_por_ciento_y_llena_contra_el_ask(tmp_path: Path) -> None:
    paso, venue, _ = armar(tmp_path, Modelo(1.0))
    salida = paso.run()
    # 1.0 * 100 USDT * 0.98 / close, redondeado hacia cero al lote de 1e-5.
    esperado = int(0.98 * 100.0 / cierre_grabado() / 1e-5) * 1e-5
    assert salida.order is not None
    assert salida.order.qty == pytest.approx(0.98 * 100.0 / cierre_grabado())
    [fill] = salida.fills
    assert fill.status is OrderStatus.FILLED
    assert fill.qty_filled == pytest.approx(esperado)
    assert fill.fill_price == mejor_ask()  # 0.0012 BTC cabe en el primer nivel
    assert fill.decision_price == cierre_grabado()
    assert venue.get_positions()["BTCUSDT"] == pytest.approx(esperado)
    assert salida.bar_timestamp == CIERRE


def test_accion_cero_sin_posicion_no_envia_nada(tmp_path: Path) -> None:
    paso, venue, _ = armar(tmp_path, Modelo(0.0))
    salida = paso.run()
    assert salida.order is None
    assert salida.fills == ()
    assert venue.get_positions() == {}


def test_el_veto_no_llega_al_venue_y_queda_registrado(tmp_path: Path) -> None:
    paso, venue, _ = armar(tmp_path, Modelo(1.0), gate=Puerta(veto="KILL_SWITCH"))
    salida = paso.run()
    assert salida.veto is not None
    assert salida.veto.reason == "KILL_SWITCH"
    assert salida.fills == ()
    assert venue.get_positions() == {}
    assert venue.reconcile().open_order_ids == ()
    registro = [r for r in venue._store.replay() if r["type"] == "decision"]
    assert registro[0]["veto_reason"] == "KILL_SWITCH"


def test_reiniciar_retoma_la_misma_orden_sin_consultar_a_la_politica(
    tmp_path: Path,
) -> None:
    modelo = Modelo(1.0)
    paso, venue, _ = armar(tmp_path, modelo)
    primera = paso.run()
    pos = venue.get_positions()["BTCUSDT"]

    modelo2 = Modelo(0.0)  # si la consultara, vendería
    paso2, venue2, _ = armar(tmp_path, modelo2)
    segunda = paso2.run()
    assert segunda.resumed
    assert modelo2.vistas == []
    assert segunda.order == primera.order
    assert venue2.get_positions()["BTCUSDT"] == pos
    assert len(venue2.get_fills()) == 1


def test_barra_vieja_se_rechaza(tmp_path: Path) -> None:
    paso, _, _ = armar(tmp_path, Modelo(), ahora=AHORA + np.timedelta64(2, "h"))
    with pytest.raises(StaleBarsError, match="edad"):
        paso.run()


@pytest.mark.leakage
def test_la_barra_en_curso_no_cambia_la_observacion(tmp_path: Path) -> None:
    """Anti-leakage: lo que el venue publica de la vela abierta no entra."""
    m1 = Modelo(1.0)
    armar(tmp_path / "a", m1)[0].run()
    klines = json.loads((RESPUESTAS / "klines_1d.json").read_text())
    klines[-1][1:6] = ["1", "2", "3", "4", "5"]  # la vela en curso, alterada
    m2 = Modelo(1.0)
    armar(tmp_path / "b", m2, servidor=Servidor(klines_1d=klines))[0].run()
    assert m1.vistas[0].tolist() == m2.vistas[0].tolist()


def test_liquidation_equity_oraculo_a_mano() -> None:
    bids = np.array([[100.0, 1.0], [99.0, 2.0]])
    com = SchemaCommission(schema=CommissionSchema(kind="percent", value=0.001))
    # 2 BTC: 100 + 99 = 199; vwap 99.5; comision 2 * 99.5 * 0.001 = 0.199
    assert liquidation_equity(10.0, 2.0, bids, com) == pytest.approx(208.801)
    # 4 BTC con libro de 3: el remanente se valua al peor nivel (99).
    # ingreso 100 + 198 + 99 = 397; vwap 99.25; comision 4 * 99.25 * 0.001 = 0.397
    assert liquidation_equity(10.0, 4.0, bids, com) == pytest.approx(406.603)
    assert liquidation_equity(10.0, 0.0, bids, com) == 10.0
