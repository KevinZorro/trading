"""Capa de riesgo: cada limite bloquea y queda registrado con su motivo.

Los limites se prueban sobre ``AccountSnapshot`` construidos a mano, con los
numeros elegidos para que el umbral se cruce por un margen calculable a ojo.
"""

from __future__ import annotations

import numpy as np
import pytest

from risk.layer import RiskLayer, RiskLimits, RiskRejectReason
from sim.clock import SimulatedClock
from sim.gate import GateDecision, OrderGate
from sim.orders import Fill, MarketOrder, OrderStatus
from sim.view import AccountSnapshot

T0 = np.datetime64("2020-01-01T12:00:00", "ns")


def cuenta(
    *,
    position: float = 0.0,
    mark_price: float = 100.0,
    cash: float = 10_000.0,
    equity: float = 10_000.0,
    t: int = 0,
) -> AccountSnapshot:
    return AccountSnapshot(
        t=t,
        cash=cash,
        position=position,
        mark_price=mark_price,
        equity=equity,
        pending_qty=0.0,
    )


def fill_ejecutado(*, qty: float, price: float) -> Fill:
    return Fill(
        t_decision=0,
        t_fill=1,
        timestamp_decision=T0,
        timestamp_fill=T0,
        symbol="TEST",
        status=OrderStatus.FILLED,
        qty_requested=qty,
        qty_filled=qty,
        decision_price=price,
        ref_price=price,
        fill_price=price,
        gap=0.0,
        spread_cost=0.0,
        slippage_cost=0.0,
        commission=0.0,
        participation=0.0,
    )


def capa(**kwargs: object) -> RiskLayer:
    return RiskLayer(RiskLimits(**kwargs), SimulatedClock(T0))  # type: ignore[arg-type]


class TestProtocolo:
    def test_la_capa_satisface_order_gate(self) -> None:
        assert isinstance(capa(), OrderGate)

    def test_sin_limites_todo_pasa(self) -> None:
        gate = capa()
        assert gate.check(MarketOrder(qty=1_000.0), cuenta()).approved
        assert gate.events == []


class TestLimiteDeNocional:
    def test_bloquea_la_orden_que_supera_el_limite(self) -> None:
        # Posicion resultante: 0 + 21 unidades a 100 = 2100 > 2000.
        gate = capa(max_position_notional=2_000.0)
        decision = gate.check(MarketOrder(qty=21.0), cuenta())
        assert not decision.approved
        assert decision.reason == RiskRejectReason.MAX_POSITION_NOTIONAL

    def test_deja_pasar_la_que_llega_justo_al_limite(self) -> None:
        # 20 unidades a 100 = 2000, exactamente el limite.
        gate = capa(max_position_notional=2_000.0)
        assert gate.check(MarketOrder(qty=20.0), cuenta()).approved

    def test_mide_la_posicion_resultante_no_la_orden(self) -> None:
        """Con 15 en cartera, comprar 10 mas lleva a 2500: se rechaza aunque la
        orden sola (1000) este por debajo del limite."""
        gate = capa(max_position_notional=2_000.0)
        decision = gate.check(MarketOrder(qty=10.0), cuenta(position=15.0))
        assert not decision.approved
        assert decision.reason == RiskRejectReason.MAX_POSITION_NOTIONAL

    def test_queda_registrado_con_motivo(self) -> None:
        gate = capa(max_position_notional=2_000.0)
        gate.check(MarketOrder(qty=21.0), cuenta())
        assert len(gate.events) == 1
        evento = gate.events[0]
        assert evento.kind == "order_rejected"
        assert evento.reason == RiskRejectReason.MAX_POSITION_NOTIONAL
        assert "2100.00" in evento.detail
        assert evento.timestamp == T0


class TestLimiteDePorcentajeDeEquity:
    def test_bloquea_por_encima_del_porcentaje(self) -> None:
        # 60 unidades a 100 = 6000 sobre equity 10000 = 60% > 50%.
        gate = capa(max_position_pct_equity=0.50)
        decision = gate.check(MarketOrder(qty=60.0), cuenta())
        assert not decision.approved
        assert decision.reason == RiskRejectReason.MAX_POSITION_PCT_EQUITY

    def test_deja_pasar_justo_en_el_porcentaje(self) -> None:
        gate = capa(max_position_pct_equity=0.50)
        assert gate.check(MarketOrder(qty=50.0), cuenta()).approved


class TestLimiteDePerdidaDiaria:
    def _gate_con_dia_abierto(self, limite: float) -> RiskLayer:
        gate = capa(max_daily_loss_pct=limite, trip_on_daily_loss=False)
        gate.observe_bar(cuenta(equity=10_000.0))
        return gate

    def test_bloquea_cuando_la_perdida_del_dia_supera_el_limite(self) -> None:
        gate = self._gate_con_dia_abierto(0.02)
        # Equity cae a 9_700: perdida del 3% sobre el equity de apertura.
        gate.observe_bar(cuenta(equity=9_700.0))
        decision = gate.check(MarketOrder(qty=1.0), cuenta(equity=9_700.0))
        assert not decision.approved
        assert decision.reason == RiskRejectReason.MAX_DAILY_LOSS

    def test_no_bloquea_por_debajo_del_limite(self) -> None:
        gate = self._gate_con_dia_abierto(0.05)
        gate.observe_bar(cuenta(equity=9_700.0))
        assert gate.check(MarketOrder(qty=1.0), cuenta(equity=9_700.0)).approved

    def test_el_ancla_es_el_equity_de_apertura_no_el_maximo(self) -> None:
        """Es un limite de perdida diaria, no un drawdown intradiario.

        Sube a 12000 y vuelve a 10000: contra el maximo seria -16.7%, contra la
        apertura es 0%. La eleccion importa y esta fijada aca.
        """
        gate = self._gate_con_dia_abierto(0.05)
        gate.observe_bar(cuenta(equity=12_000.0))
        gate.observe_bar(cuenta(equity=10_000.0))
        assert gate.check(MarketOrder(qty=1.0), cuenta(equity=10_000.0)).approved

    def test_con_barras_diarias_el_ancla_es_el_cierre_del_dia_anterior(self) -> None:
        """El caso que hace util al limite en vez de decorativo.

        Con barras diarias una barra es un dia entero. Si el ancla fuera el
        equity de la propia barra, el limite compararia el equity contra si
        mismo y no morderia nunca. El ancla es el cierre del dia anterior.
        """
        reloj = SimulatedClock(T0)
        gate = RiskLayer(
            RiskLimits(max_daily_loss_pct=0.10, trip_on_daily_loss=False), reloj
        )
        gate.observe_bar(cuenta(equity=10_000.0))  # dia 1
        reloj.advance_to(T0 + np.timedelta64(1, "D"))
        gate.observe_bar(cuenta(equity=8_000.0))  # dia 2: -20% contra el cierre
        decision = gate.check(MarketOrder(qty=1.0), cuenta(equity=8_000.0))
        assert not decision.approved
        assert decision.reason == RiskRejectReason.MAX_DAILY_LOSS

    def test_el_ancla_se_reinicia_al_cambiar_de_dia(self) -> None:
        reloj = SimulatedClock(T0)
        gate = RiskLayer(
            RiskLimits(max_daily_loss_pct=0.02, trip_on_daily_loss=False), reloj
        )
        gate.observe_bar(cuenta(equity=10_000.0))
        gate.observe_bar(cuenta(equity=9_000.0))  # -10%, bloqueado
        assert not gate.check(MarketOrder(qty=1.0), cuenta(equity=9_000.0)).approved

        reloj.advance_to(T0 + np.timedelta64(1, "D"))
        gate.observe_bar(cuenta(equity=9_000.0))  # dia nuevo: ancla en 9000
        assert gate.check(MarketOrder(qty=1.0), cuenta(equity=9_000.0)).approved


class TestPerdidaSinAncla:
    """Sin observe_bar previo no hay dia abierto: no se puede medir la perdida.

    Es el estado del arranque, antes de la primera barra. Devolver 0 en vez de
    inventar un ancla evita que el limite bloquee la primera orden de la corrida
    por un dato que todavia no existe.
    """

    def test_sin_barras_observadas_no_bloquea(self) -> None:
        gate = capa(max_daily_loss_pct=0.01)
        assert gate.check(MarketOrder(qty=1.0), cuenta(equity=1.0)).approved

    def test_expone_los_limites_configurados(self) -> None:
        gate = capa(max_position_notional=123.0)
        assert gate.limits.max_position_notional == 123.0


class TestLimiteDeTurnover:
    def test_bloquea_cuando_el_turnover_proyectado_supera_el_limite(self) -> None:
        gate = capa(max_daily_turnover_ratio=0.50)
        gate.observe_bar(cuenta())
        # Ya se operaron 4000 de nocional; esta orden agrega 2000 -> 0.6x.
        gate.observe_fill(fill_ejecutado(qty=40.0, price=100.0))
        decision = gate.check(MarketOrder(qty=20.0), cuenta())
        assert not decision.approved
        assert decision.reason == RiskRejectReason.MAX_DAILY_TURNOVER

    def test_no_bloquea_por_debajo_del_limite(self) -> None:
        gate = capa(max_daily_turnover_ratio=0.50)
        gate.observe_bar(cuenta())
        gate.observe_fill(fill_ejecutado(qty=40.0, price=100.0))
        assert gate.check(MarketOrder(qty=9.0), cuenta()).approved

    def test_una_venta_cuenta_su_nocional_en_valor_absoluto(self) -> None:
        gate = capa(max_daily_turnover_ratio=0.50)
        gate.observe_bar(cuenta())
        gate.observe_fill(fill_ejecutado(qty=-40.0, price=100.0))
        decision = gate.check(MarketOrder(qty=20.0), cuenta())
        assert not decision.approved

    def test_el_turnover_se_reinicia_al_cambiar_de_dia(self) -> None:
        reloj = SimulatedClock(T0)
        gate = RiskLayer(RiskLimits(max_daily_turnover_ratio=0.50), reloj)
        gate.observe_bar(cuenta())
        gate.observe_fill(fill_ejecutado(qty=40.0, price=100.0))
        assert not gate.check(MarketOrder(qty=20.0), cuenta()).approved

        reloj.advance_to(T0 + np.timedelta64(1, "D"))
        gate.observe_bar(cuenta())
        assert gate.check(MarketOrder(qty=20.0), cuenta()).approved


class TestKillSwitch:
    def test_se_dispara_solo_al_superar_la_perdida_diaria(self) -> None:
        gate = capa(max_daily_loss_pct=0.02)
        gate.observe_bar(cuenta(equity=10_000.0))
        assert not gate.kill_switch_active
        gate.observe_bar(cuenta(equity=9_700.0))
        assert gate.kill_switch_active

    def test_no_se_dispara_si_trip_on_daily_loss_es_falso(self) -> None:
        gate = capa(max_daily_loss_pct=0.02, trip_on_daily_loss=False)
        gate.observe_bar(cuenta(equity=10_000.0))
        gate.observe_bar(cuenta(equity=9_700.0))
        assert not gate.kill_switch_active

    def test_bloquea_las_compras(self) -> None:
        gate = capa()
        gate.trip_kill_switch("prueba")
        decision = gate.check(MarketOrder(qty=1.0), cuenta(position=5.0))
        assert not decision.approved
        assert decision.reason == RiskRejectReason.KILL_SWITCH

    def test_deja_pasar_las_ordenes_que_cierran(self) -> None:
        """Un kill switch que bloquea todo deja la posicion abierta e indefensa,
        que suele ser peor que el riesgo que lo disparo."""
        gate = capa()
        gate.trip_kill_switch("prueba")
        assert gate.check(MarketOrder(qty=-5.0), cuenta(position=5.0)).approved

    def test_una_venta_sin_posicion_no_es_una_reduccion(self) -> None:
        gate = capa()
        gate.trip_kill_switch("prueba")
        decision = gate.check(MarketOrder(qty=-1.0), cuenta(position=0.0))
        assert not decision.approved
        assert decision.reason == RiskRejectReason.KILL_SWITCH

    def test_queda_registrado_con_motivo(self) -> None:
        gate = capa(max_daily_loss_pct=0.02)
        gate.observe_bar(cuenta(equity=10_000.0))
        gate.observe_bar(cuenta(equity=9_700.0))
        disparos = [e for e in gate.events if e.kind == "kill_switch_tripped"]
        assert len(disparos) == 1
        assert "3.0000%" in disparos[0].detail
        assert "2.0000%" in disparos[0].detail

    def test_no_se_registra_dos_veces(self) -> None:
        gate = capa(max_daily_loss_pct=0.02)
        gate.observe_bar(cuenta(equity=10_000.0))
        gate.observe_bar(cuenta(equity=9_700.0))
        gate.observe_bar(cuenta(equity=9_000.0))
        assert len([e for e in gate.events if e.kind == "kill_switch_tripped"]) == 1

    def test_disparar_dos_veces_es_idempotente(self) -> None:
        gate = capa()
        gate.trip_kill_switch("primera")
        gate.trip_kill_switch("segunda")
        disparos = [e for e in gate.events if e.kind == "kill_switch_tripped"]
        assert len(disparos) == 1
        assert disparos[0].detail == "primera"

    def test_reponer_sin_haber_disparado_no_hace_nada(self) -> None:
        gate = capa()
        gate.reset_kill_switch()
        assert gate.events == []

    def test_la_reposicion_es_manual(self) -> None:
        """Si se repusiera solo no seria un kill switch sino una pausa."""
        gate = capa(max_daily_loss_pct=0.02)
        gate.observe_bar(cuenta(equity=10_000.0))
        gate.observe_bar(cuenta(equity=9_700.0))
        assert gate.kill_switch_active

        gate.reset_kill_switch("revisado por un humano")
        assert not gate.kill_switch_active
        assert any(e.kind == "kill_switch_reset" for e in gate.events)


class TestNuncaRedimensiona:
    """La regla que hace auditable el log: rechaza o aprueba, no recorta."""

    def test_la_decision_no_puede_devolver_una_orden(self) -> None:
        campos = set(GateDecision.__dataclass_fields__)
        assert campos == {"approved", "reason", "detail"}

    def test_una_orden_bloqueada_no_se_reduce_al_limite(self) -> None:
        gate = capa(max_position_notional=2_000.0)
        orden = MarketOrder(qty=21.0)
        decision = gate.check(orden, cuenta())
        assert not decision.approved
        # La orden original sigue intacta: quien la envio pidio 21, no 20.
        assert orden.qty == 21.0

    def test_un_rechazo_siempre_declara_el_motivo(self) -> None:
        with pytest.raises(ValueError, match="debe declarar el motivo"):
            GateDecision(approved=False)


class TestLimitesInvalidos:
    @pytest.mark.parametrize(
        "campo",
        [
            "max_position_notional",
            "max_position_pct_equity",
            "max_daily_loss_pct",
            "max_daily_turnover_ratio",
        ],
    )
    def test_no_pueden_ser_negativos(self, campo: str) -> None:
        with pytest.raises(ValueError, match=f"{campo} no puede ser negativo"):
            RiskLimits(**{campo: -1.0})  # type: ignore[arg-type]

    def test_describe_serializa_los_limites(self) -> None:
        limites = RiskLimits(max_position_notional=1_000.0)
        d = limites.describe()
        assert d["max_position_notional"] == 1_000.0
        assert d["max_daily_loss_pct"] is None
