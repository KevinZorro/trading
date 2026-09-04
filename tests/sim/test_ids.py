"""Generacion de client_order_id: determinismo e idempotencia."""

from __future__ import annotations

import numpy as np
import pytest

from sim.clock import SimulatedClock
from sim.ids import OrderIdGenerator, PrefixedSequentialIds, SequentialIds

T0 = np.datetime64("2020-01-01T00:00:00", "ns")


class TestSequentialIds:
    def test_secuencia_deterministica(self) -> None:
        gen = SequentialIds("corrida")
        assert [gen.next_id() for _ in range(3)] == [
            "corrida-00000000",
            "corrida-00000001",
            "corrida-00000002",
        ]

    def test_dos_generadores_iguales_dan_la_misma_secuencia(self) -> None:
        """Reproducibilidad total: dos corridas del mismo backtest, mismos IDs."""
        a = [SequentialIds("x").next_id() for _ in range(1)]
        b = [SequentialIds("x").next_id() for _ in range(1)]
        assert a == b

    def test_reset_vuelve_a_cero(self) -> None:
        gen = SequentialIds("x")
        gen.next_id()
        gen.reset()
        assert gen.next_id() == "x-00000000"

    def test_run_id_vacio(self) -> None:
        with pytest.raises(ValueError, match="run_id no puede ser vacio"):
            SequentialIds("")

    def test_satisface_el_protocolo(self) -> None:
        assert isinstance(SequentialIds("x"), OrderIdGenerator)


class TestPrefixedSequentialIds:
    def test_el_prefijo_sale_del_reloj_inyectado(self) -> None:
        gen = PrefixedSequentialIds(SimulatedClock(T0), "live")
        esperado = f"live-{T0.astype('datetime64[ns]').astype(int)}-00000000"
        assert gen.next_id() == esperado

    def test_dos_arranques_en_instantes_distintos_no_colisionan(self) -> None:
        """El caso que SequentialIds no cubre: reinicio del proceso en vivo."""
        a = PrefixedSequentialIds(SimulatedClock(T0), "live").next_id()
        b = PrefixedSequentialIds(
            SimulatedClock(T0 + np.timedelta64(1, "s")), "live"
        ).next_id()
        assert a != b

    def test_dentro_de_una_instancia_sigue_siendo_deterministico(self) -> None:
        """El prefijo se calcula una vez; no vuelve a consultar el reloj."""
        reloj = SimulatedClock(T0)
        gen = PrefixedSequentialIds(reloj, "live")
        primero = gen.next_id()
        reloj.advance_to(T0 + np.timedelta64(5, "D"))
        segundo = gen.next_id()
        assert primero.rsplit("-", 1)[0] == segundo.rsplit("-", 1)[0]
        assert segundo.endswith("-00000001")

    def test_satisface_el_protocolo(self) -> None:
        assert isinstance(PrefixedSequentialIds(SimulatedClock(T0)), OrderIdGenerator)
