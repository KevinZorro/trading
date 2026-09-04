"""El protocolo ExecutionVenue implementado por el simulador.

El test central del archivo es el de idempotencia: reenviar el mismo
``client_order_id`` no debe ejecutar dos veces. Ese es el escenario que en vivo
separa "reintenté tras un timeout" de "abrí el doble de posición".
"""

from __future__ import annotations

import numpy as np
import pytest

from agents.baselines import BuyAndHold
from data.instruments import InstrumentSpec
from sim.engine import SimConfig, Simulator
from sim.ids import SequentialIds
from sim.orders import MarketOrder, OrderStatus, RejectReason
from sim.venue import CancelAck, ExecutionVenue, OrderAck, VenueState

from .conftest import make_series


@pytest.fixture
def serie(fractional: InstrumentSpec):
    return make_series(fractional, [100.0, 101.0, 102.0, 103.0, 104.0])


@pytest.fixture
def simulador(serie) -> Simulator:
    return Simulator(serie, SimConfig(initial_cash=10_000.0))


class TestProtocolo:
    def test_el_simulador_satisface_execution_venue(self, simulador) -> None:
        assert isinstance(simulador, ExecutionVenue)

    def test_reconcile_devuelve_el_estado_de_arranque(self, simulador) -> None:
        estado = simulador.reconcile()
        assert isinstance(estado, VenueState)
        assert estado.cash == pytest.approx(10_000.0)
        assert estado.positions == {}
        assert estado.open_order_ids == ()
        assert estado.position("FRAC") == 0.0

    def test_get_positions_devuelve_una_copia(self, simulador) -> None:
        posiciones = simulador.get_positions()
        posiciones["FRAC"] = 999.0
        assert simulador.get_positions() == {}

    def test_get_fills_rechaza_indices_negativos(self, simulador) -> None:
        with pytest.raises(ValueError, match="since no puede ser negativo"):
            simulador.get_fills(since=-1)


class TestAcusesInconsistentes:
    """El acuse no puede mentir sobre si acepto o rechazo."""

    def test_aceptada_con_motivo_de_rechazo(self) -> None:
        with pytest.raises(ValueError, match="no puede llevar motivo de rechazo"):
            OrderAck(
                client_order_id="a",
                accepted=True,
                reject_reason=RejectReason.NO_VOLUME,
            )

    def test_rechazada_sin_motivo(self) -> None:
        with pytest.raises(ValueError, match="debe declarar el motivo"):
            OrderAck(client_order_id="a", accepted=False)


class TestAdmision:
    """Validacion al enviar, distinta de la validacion al llenar."""

    def test_sin_client_order_id_se_rechaza(self, simulador) -> None:
        ack = simulador.submit_order(MarketOrder(qty=1.0))
        assert isinstance(ack, OrderAck)
        assert not ack.accepted
        assert ack.reject_reason is RejectReason.MISSING_CLIENT_ORDER_ID

    def test_fuera_del_bucle_se_rechaza(self, simulador) -> None:
        ack = simulador.submit_order(MarketOrder(qty=1.0, client_order_id="a"))
        assert not ack.accepted
        assert ack.reject_reason is RejectReason.VENUE_NOT_RUNNING

    def test_el_acuse_no_es_un_fill(self, serie, fractional) -> None:
        """submit_order acusa recepcion; el fill llega despues, al open de t+1."""
        vistos: list[OrderAck] = []

        class Emisor:
            name = "emisor"

            def __init__(self, venue: Simulator) -> None:
                self.venue = venue

            def reset(self, seed: int | None = None) -> None:
                return None

            def on_bar(self, view, account):
                if view.t == 0:
                    orden = MarketOrder(qty=1.0, client_order_id="propio-1")
                    vistos.append(self.venue.submit_order(orden))
                    # En el instante del envio todavia no hay fill.
                    assert self.venue.get_fills() == []
                return None

        sim = Simulator(serie, SimConfig(initial_cash=10_000.0))
        resultado = sim.run(Emisor(sim))
        assert len(vistos) == 1
        assert vistos[0].accepted
        assert vistos[0].client_order_id == "propio-1"
        # El fill existe, pero recien en la barra siguiente.
        ejecutados = resultado.executed_fills
        assert len(ejecutados) == 1
        assert ejecutados[0].t_decision == 0
        assert ejecutados[0].t_fill == 1
        assert ejecutados[0].client_order_id == "propio-1"


class TestIdempotencia:
    """Reenviar el mismo identificador no ejecuta dos veces."""

    def _correr(self, serie, veces: int):
        class Reenviador:
            name = "reenviador"

            def __init__(self, venue: Simulator) -> None:
                self.venue = venue
                self.acks: list[OrderAck] = []

            def reset(self, seed: int | None = None) -> None:
                self.acks = []

            def on_bar(self, view, account):
                if view.t == 0:
                    orden = MarketOrder(qty=2.0, client_order_id="duplicado")
                    for _ in range(veces):
                        self.acks.append(self.venue.submit_order(orden))
                return None

        sim = Simulator(serie, SimConfig(initial_cash=10_000.0))
        estrategia = Reenviador(sim)
        return sim.run(estrategia), estrategia

    def test_un_solo_envio_produce_un_solo_fill(self, serie) -> None:
        resultado, _ = self._correr(serie, veces=1)
        assert len(resultado.executed_fills) == 1

    def test_reenviar_el_mismo_id_no_duplica_la_ejecucion(self, serie) -> None:
        resultado, estrategia = self._correr(serie, veces=3)
        # Tres envios, un solo fill: exactamente lo que evita la posicion doble.
        assert len(estrategia.acks) == 3
        assert len(resultado.executed_fills) == 1
        assert resultado.executed_fills[0].qty_filled == pytest.approx(2.0)

    def test_el_reenvio_se_marca_como_duplicado(self, serie) -> None:
        _, estrategia = self._correr(serie, veces=3)
        assert estrategia.acks[0].accepted
        assert not estrategia.acks[0].is_duplicate
        assert all(a.is_duplicate for a in estrategia.acks[1:])
        assert all(a.accepted for a in estrategia.acks[1:])

    def test_la_posicion_final_es_la_de_un_solo_envio(self, serie) -> None:
        una, _ = self._correr(serie, veces=1)
        tres, _ = self._correr(serie, veces=3)
        np.testing.assert_allclose(una.position, tres.position)
        np.testing.assert_allclose(una.equity, tres.equity)


class TestCancelacion:
    def test_cancelar_la_orden_encolada_evita_el_fill(self, serie) -> None:
        class Cancelador:
            name = "cancelador"

            def __init__(self, venue: Simulator) -> None:
                self.venue = venue
                self.ack: CancelAck | None = None

            def reset(self, seed: int | None = None) -> None:
                return None

            def on_bar(self, view, account):
                if view.t == 0:
                    self.venue.submit_order(
                        MarketOrder(qty=1.0, client_order_id="a-cancelar")
                    )
                    self.ack = self.venue.cancel_order("a-cancelar")
                return None

        sim = Simulator(serie, SimConfig(initial_cash=10_000.0))
        estrategia = Cancelador(sim)
        resultado = sim.run(estrategia)
        assert estrategia.ack is not None
        assert estrategia.ack.cancelled
        assert resultado.fills == []

    def test_cancelar_algo_inexistente_no_es_un_error(self, simulador) -> None:
        ack = simulador.cancel_order("no-existe")
        assert not ack.cancelled
        assert "no hay una orden viva" in ack.detail


class TestIdentificadoresEnLaCorrida:
    def test_el_runner_asigna_id_a_las_ordenes_sin_uno(self, serie) -> None:
        resultado = Simulator(serie, SimConfig(initial_cash=10_000.0)).run(BuyAndHold())
        assert resultado.executed_fills[0].client_order_id == "sim-00000000"

    def test_el_run_id_de_la_config_prefija_los_identificadores(self, serie) -> None:
        config = SimConfig(initial_cash=10_000.0, run_id="tier_alto_seed7")
        resultado = Simulator(serie, config).run(BuyAndHold())
        assert resultado.executed_fills[0].client_order_id.startswith(
            "tier_alto_seed7-"
        )
        assert resultado.config["run_id"] == "tier_alto_seed7"

    def test_dos_corridas_producen_los_mismos_identificadores(self, serie) -> None:
        """Reproducibilidad: los logs de dos corridas deben ser comparables."""
        sim = Simulator(serie, SimConfig(initial_cash=10_000.0))
        primera = [f.client_order_id for f in sim.run(BuyAndHold()).fills]
        segunda = [f.client_order_id for f in sim.run(BuyAndHold()).fills]
        assert primera == segunda
        assert primera != []

    def test_el_generador_se_puede_inyectar(self, serie) -> None:
        sim = Simulator(
            serie,
            SimConfig(initial_cash=10_000.0),
            order_ids=SequentialIds("inyectado"),
        )
        assert sim.run(BuyAndHold()).fills[0].client_order_id == "inyectado-00000000"

    def test_la_orden_expirada_conserva_su_identificador(self, fractional) -> None:
        serie = make_series(fractional, [100.0, 100.0, 100.0])

        class CompraSiempre:
            name = "compra_siempre"

            def reset(self, seed: int | None = None) -> None:
                return None

            def on_bar(self, view, account):
                return MarketOrder(qty=0.001, tag="tarde")

        resultado = Simulator(serie, SimConfig(initial_cash=10_000.0)).run(
            CompraSiempre()
        )
        expiradas = [f for f in resultado.fills if f.status is OrderStatus.EXPIRED]
        assert len(expiradas) == 1
        assert expiradas[0].client_order_id != ""
        assert expiradas[0].tag == "tarde"


class TestReconciliacionDeArranque:
    def test_el_estado_se_reinicia_entre_corridas(self, serie) -> None:
        """Correr dos veces el mismo simulador no arrastra posiciones."""
        sim = Simulator(serie, SimConfig(initial_cash=10_000.0))
        primera = sim.run(BuyAndHold())
        segunda = sim.run(BuyAndHold())
        np.testing.assert_allclose(primera.equity, segunda.equity)
        np.testing.assert_allclose(primera.position, segunda.position)
        assert len(primera.fills) == len(segunda.fills)
